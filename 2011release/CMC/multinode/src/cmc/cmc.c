/* 
 * Created on July 26, 2011, 10:51 AM
 */

#include "emmintrin.h"
#include "cmc.h"
#include "cmc_utils.h"
#include "cmc_gen_tally.h"
#include "../ecodes.h"
#ifdef DO_PAPI
#include "papi.h"
#endif
#ifdef DO_MPI
#include "mpi.h"
#endif

#define VECTOR_WIDTH 2
#ifndef SSE_SWITCH
#define SSE_SWITCH 1000
#endif


colData allocate_tallies(int);
void deallocate_tallies(colData);

/*
 * Main Function
 */
int run(int rank) {


  long long NP;		                                        // Number of particles
  pfloat sysLeng;	                                        // Length of system
  int numCells;	                                              	// Number of cells in system
  pfloat tol_std;                                      		// Tolerance for standard deviation calculations
  
  //Random number generator 
  dsfmt_t * dsfmt = (dsfmt_t*)malloc(sizeof(dsfmt_t));

  //stores user input
  NP = opt.NP;
  sysLeng = opt.sysLen;
  numCells = opt.numCells;
  tol_std = opt.tol_std;

  int nprocs;
#ifdef DO_MPI
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
#else
  nprocs = 1;
#endif

  //Initialization of vital simulation parameters
  pfloat dx = sysLeng / numCells;                             	//cell width
  pfloat eps = 0.1;                                          	//the threshold angle for tallying
  pfloat sig_t = opt.sig_t;                                   	//total collision cross-section
  pfloat sig_s = opt.sig_s;                                     //scattering cross-section

  //The source term uniform for now	
  pfloat q0_avg = 1.0;

  //Form data structure for simulation parameters
  struct data data;
  data.NP = NP;							//Number of Particles
  data.lx = sysLeng;						//Length of System
  data.nx = numCells;						//Number of Cells
  data.dx = dx;						        //Cell Width
  data.dx2 = dx*dx;						// Cell Width SQUARED
  data.dx_recip = 1.0/dx;					// Pre-computed reciprocal
  data.sig_t = sig_t;						//Total Collision Cross-Section
  data.sig_t_recip = 1.0/sig_t;					// Pre-computed reciprocal
  data.sig_s = sig_s;						//Scattering Collision Cross-Section
  data.q0_avg = q0_avg;                                         //Weight of each particle
  data.eps = eps;						//Threshold Angle for Tallying
  
  long long NPc = (long long )((pfloat)NP / numCells);          // number of particles in cell
  long long NP_tot = NP; 					//total number of particles for the CMC
  data.NP_tot = NP_tot;						// Total Number of Particles
  data.NPc = NPc;						// Number of Particles per Cell

  //Initialize the average values
  int iter = 0;     
  int iter_avg = 0 ;                                            //tally for iteration for convergence
  int i;

  // MT RNG
  dsfmt_init_gen_rand(dsfmt, (int)time(NULL));
  
  //time keeping structures
  struct timeval start_time, end_time, startPerIter, endPerIter;

  afloat * phi_n_avg = NULL;                                     //Vector to hold avg value of phi_n across iterations
  afloat * phi_n_tot = NULL;                                     //Vector to hold sum of phi_n across iterations
  afloat * phiS2_n_tot = NULL;
#if defined(L2_1) || defined(L2_2) || defined(L2_3)
  afloat * anal_soln = NULL;                                     //Holds the analytical solution if comparing with it
  pfloat l2;                                                    //Stores the l2 norm while comparing with analytical solution
#else
  meanDev * phi_nStats = NULL;                                   //Mean and standard deviation
#endif

  colData all_tallies;                                           //Stores the tallies for the current iteration
  
  if( rank == 0 )
    {
      phi_n_avg = (afloat *) calloc(numCells, sizeof(afloat));
      phi_n_tot  = (afloat*) calloc(numCells,sizeof(afloat));
      phiS2_n_tot = (afloat*) calloc(numCells,sizeof(afloat));
  
#if defined(L2_1) || defined(L2_2) || defined(L2_3)
      anal_soln = (afloat *) calloc(numCells, sizeof (afloat)); 			// Reference solution
#if defined(L2_1)
      init_L2_norm_1(anal_soln);
#elif defined(L2_2)
      init_L2_norm_2(anal_soln);
#elif defined(L2_3)
      init_L2_norm_3(anal_soln);
#endif
#else
      //Initialize vector to hold stats of the monte carlo
      phi_nStats = (meanDev*) malloc(sizeof(meanDev)*opt.numCells);
#endif
    }

  //Plot and print initial condition as well as simulation parameters
  if (rank == 0 && !opt.silent) sim_param_print(data);
  
  int flag_sim = 0;                                                                     //Flag to singal convergence
  long long samp_cnt = 0;               						//total history tally

  
  //Allocate tallies on each process. Aligned to cache line ( not really necessary for MPI )
  colData tallies = allocate_tallies(numCells);

#ifdef DO_MPI
  //In MPI mode, allocate on process 0, all_tallies to hold sum of tallies across processes
  if( rank == 0 )
    all_tallies = allocate_tallies(numCells);
  else
    {
      all_tallies.phi_n = NULL;
      all_tallies.phi_n2 = NULL;
      all_tallies.tot_col = NULL;
      all_tallies.full_buffer = NULL;
      all_tallies.buffer_size = tallies.buffer_size;
    }
#else
  //For sequential mode, alias all_tallies to tallies
  all_tallies.phi_n = tallies.phi_n;
  all_tallies.phi_n2 = tallies.phi_n2;
  all_tallies.tot_col = tallies.tot_col;
  all_tallies.buffer_size = tallies.buffer_size;
#endif

  double avg_col = 0;                                                                //Average collisions across all the particles streamed across all processes
  /* double tot_col = 0; */
  /* double all_col =  0; */
  double running_col = 0.0;
  
  //Calculation starts here
  gettimeofday(&start_time, NULL);
	
#ifdef DO_PAPI    
    PAPI_library_init(PAPI_VER_CURRENT);
    int EventSet = PAPI_NULL;
    long long start[4],stop[4];
    PAPI_create_eventset(&EventSet);
    PAPI_add_event(EventSet,PAPI_TOT_CYC);
    PAPI_add_event(EventSet,PAPI_TOT_INS);
    PAPI_add_event(EventSet,PAPI_FP_OPS); 
    PAPI_add_event(EventSet,PAPI_FP_INS); 
    PAPI_start(EventSet);
    PAPI_read(EventSet,start);
#endif
  // while not converged and total number of iterations < opt.numiters
  while (flag_sim == 0 && iter < opt.numIters) {

    /* //start time for iteration */
    /* gettimeofday(&startPerIter, NULL); */

    iter += 1;
    if(!opt.silent && rank==0) printf("This is the %dth iteration\n", iter);
    
    //calls collision and tally to stream particles and collect statistics
    *(tallies.tot_col) = collision_and_tally(data, &tallies,rank,nprocs,dsfmt);

#ifdef DO_MPI
    //In MPI mode, need to reduce phi_n and phi_n2 across all processes
    MPI_Reduce(tallies.full_buffer,all_tallies.full_buffer,tallies.buffer_size,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
#endif

    //Do the averaging and convergence calculations on process 0
    if( rank == 0 )
      {
	running_col += *(all_tallies.tot_col);
	avg_col = running_col / iter / NP;

	//Normalizing the values of phi_n
	for( i = 0 ; i < numCells ; i++ )
	  {
	    all_tallies.phi_n[i] /= (afloat)data.dx*NP;
	    all_tallies.phi_n[i] *= data.lx;
	  }
	
	/***************************************************
	  Calculates the averages 
	  **************************************************\/ */
	iter_avg += 1;
	samp_cnt = NP_tot * ((long long) iter_avg);

	//accumulate phase -- add new data to the running averages
	for (i = 0; i < numCells; i++) {
	  phi_n_tot[i] += all_tallies.phi_n[i];
	  phiS2_n_tot[i] += all_tallies.phi_n2[i];
	}

	// for each cell, calculate the average for phi_n, phi_lo and E_n
	for (i = 0; i < numCells; i++) {
	  phi_n_avg[i] = phi_n_tot[i] / iter_avg;
	}

	//prints out the necessary data
	if(!opt.silent && rank == 0){	    
	  printf("Running Average number of collisions = %lf\n",avg_col);
	}

	//check for convergence
#if !defined(L2_1) && !defined(L2_2) && !defined(L2_3)
	//Use mean and standard deviation for convergence 
	mean_std_calc(phi_n_tot, phiS2_n_tot, samp_cnt, opt.NP, opt.numCells, opt.dx, phi_nStats);
	if (maxFAS(&phi_nStats[0].stdDev, numCells, 2) <= tol_std) {
	  flag_sim = 1;
	}
	  
	if( !opt.silent ){
	  printf("The maximum standard deviation of flux at node is, max (sig_phi) = %f\n", maxFAS(&phi_nStats[0].stdDev, numCells, 2));
	}
#else
	//Use L2 norm of the analytical solution
	l2 = l2_norm_cmp(phi_n_avg, anal_soln, numCells, dx);
	flag_sim = l2 <= tol_std;
	if(rank == 0){
	  gettimeofday(&end_time, NULL);
	  printf("L2: %f, Sofar: %ldu_sec\n", l2, (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));
	}
#endif	  
       
      }	
#ifdef DO_MPI
    //Broadcast to other processes if convergence is reached
    MPI_Bcast(&flag_sim,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
    
    /* //end time per iteration */
    /* gettimeofday(&endPerIter, NULL); */
    /* printf("ID = %d, Time per Iteration: %ldu_sec, flags sim = %d\n\n", rank,(endPerIter.tv_sec - startPerIter.tv_sec)*1000000 + (endPerIter.tv_usec - startPerIter.tv_usec),flag_sim); */
  }
#ifdef DO_PAPI
    PAPI_read(EventSet,stop);
    printf("%lld %lld %lld %lld\n",stop[0] - start[0],stop[1] - start[1],stop[2] - start[2],stop[3] - start[3]);
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(EventSet);
#endif

  
  gettimeofday(&end_time, NULL);
  if( rank == 0 )
    printf("Elapsed Time: %ldu_sec\n", (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));

  if(rank == 0 && !opt.silent){
    printf("NODEAVG\n");
    for (i = 0; i < numCells; i++) {
      printf("%d %lf\n", i, phi_n_avg[i]);
    }
  }

  /************************************************
   * Free memory
   *************************************************/
  deallocate_tallies(tallies);
  free(dsfmt);
  if( rank == 0 )
    {
#ifdef DO_MPI
      deallocate_tallies(all_tallies);
#endif
      free(phi_n_tot);
      free(phiS2_n_tot);
      free(phi_n_avg);
#if defined(L2_1) || defined(L2_2) || defined(L2_3)
      free(anal_soln);
#else
      free(phi_nStats);
#endif
    }

  return (EXIT_SUCCESS);
}

/*********************************************************************************************
 * sim_param_print
 *
 * Function that prints the initialization data and simulation parameters
 *
 * @param data data structure that holds initialization and simulation parameters
 *
 * @return void
 *
 * TODO better naming convention for data and struct data
 **********************************************************************************************/
void sim_param_print(struct data data) {
  pfloat lx = data.lx;
  pfloat dx = data.dx;
  int nx = data.nx;
  long long NP = data.NP;
  pfloat sig_t = data.sig_t;
  pfloat sig_s = data.sig_s;

  printf("***********************************************************\n");
  printf("******THE SIMULATION PARAMETERS ARE PRINTED OUT BELOW******\n");
  printf("The system length is, lx = %f\n", lx);
  printf("The cell width is, dx = %f\n", dx);
  printf("The number of cell is, nx = %d\n", nx);
  printf("The reference number of particle is, NP = %lld\n", NP);
  printf("The total cross section is, sig_t = %f\n", sig_t);
  printf("The scattering cross section is, sig_s = %f\n", sig_s);
  printf("Floating point data representation is, %lu byte\n", sizeof (pfloat));
  printf("Number of particles = %lld, NPc = %lld\n",data.NP,data.NPc);
  printf("***********************************************************\n");
}


/**********************************************************************************************************
 * collision_and_tally
 * Input
 * data : the physical dimensions fo the problem
 * tallies : to store the phi_n and phi_n2 ofr the current iteration
 * rank : the rank of the process
 * procs : the number of processes
 * Output
 * Number of collision across all particles streamed
 *********************************************************************************************************/
double collision_and_tally(struct data data, colData * tallies, int rank, int nprocs, dsfmt_t * dsfmt) {
  //initialize simulation parameters
  int nx = data.nx;
  long long NP = data.NP;

  //initialize tallies and corresponding variables
  memset(tallies->full_buffer,0,tallies->buffer_size);

  struct timeval startBoth, endBoth; 
  
  double tot_num_col = 0;
  
  if (NP != 0) { //weight per particl3e
    //start time for montecarlo
    /* gettimeofday(&startBoth, NULL); */
    
    //in MPI, each process simulates particles originating in a subset of the cells
    int lo_cell = nx/nprocs;
    int hi_cell = lo_cell + 1;
    int switch_rank = hi_cell * nprocs - nx;
    int startcell,endcell;
    if( rank < switch_rank )
      {
	startcell = rank * lo_cell;
	endcell = startcell + lo_cell;
      }
    else
      {
	startcell = switch_rank * lo_cell + ( rank - switch_rank ) * hi_cell;
	endcell = startcell + hi_cell;
      }

    //Generate particles and tally its statistics
    tot_num_col = gen_particles_and_tally(data, tallies,startcell,endcell,dsfmt);

    /* //end time for montecarlo */
    /* gettimeofday(&endBoth, NULL); */
    /* printf("Time for Monte Carlo per Iteration: %ld u_sec\n", (endBoth.tv_sec - startBoth.tv_sec)*1000000 + (endBoth.tv_usec - startBoth.tv_usec)); */
  }

  return tot_num_col;
}
	

/* merged particle generation and tally */
double gen_particles_and_tally(struct data data, colData *tallies, int start_iter_cell, int end_iter_cell, dsfmt_t* dsfmt){

  const pfloat lx = data.lx;                                                                 //System length
  const pfloat eps = data.eps;                                                               //mu cut-off threshold
  const pfloat sig_t_recip = data.sig_t_recip      ;                                         //Reciprocal of scattering cross-section
  const pfloat cellsize = data.dx, cellsize_recip = data.dx_recip, cellsizeS2 = data.dx2;    //Cellsize inormations
  const long long NPc = data.NPc;                                                            //Number of particles per cell
  const pfloat weight = data.q0_avg;                                                         //weight of each particle
  const pfloat weightS2 = weight*weight;			                             // for STDEV
  const int numCells = data.nx;

  //temporary variables
  pfloat fx, mu, efmu, efmu_recip, start, end, next,do_collision;                         
  int i,k;
  long long j;
  int collided;
  pfloat efmu_recipS2;
  pfloat absdist;
  int startcell, endcell, tempcell;
  pfloat begin_face, end_face;
  // pre-compute useful cell values
  pfloat weight_efmu_recip, weightS2_efmu_recipS2;
  // for inner loop
  pfloat weight_cellsize_efmu_recip, weightS2_cellsizeS2_efmu_recipS2;
  
#ifndef NDEBUG
  pfloat  mu_avg = 0.0; 
  pfloat  fx_avg = 0.0;
  int counter = 0;
#endif
  double col_tot = 0.0;
  
  /* create and tally all particles */
  for (i = start_iter_cell; i < end_iter_cell; i++) {
    // all particles are weighted from their starting (not left-most) cell
	
    for (j = 0; j < NPc; j++) {
      //The initial position of the particle
      next = (i  +  unirandMT_0_1(dsfmt)) * cellsize;
	  
      double num_col = 0.0;
      do {
	//initialize mu, fx, start
	start = next;

	mu = unirandMT_1_1(dsfmt);		      
	fx = (-log(unirandMT_0_1(dsfmt)) * sig_t_recip); 

#ifndef NDEBUG
	mu_avg += mu;
	fx_avg += ( fx * mu > 0 ? fx * mu : -fx*mu );
	counter++;
#endif

	end = start + mu * fx;	// done with unmodified mu

	startcell  = (int)(start * cellsize_recip);

	//Check if particle goea out left boundary
	if( end <= 0.0 )
	  {
	    end = 0.0;
	    collided = 0;
	    endcell = 0;
	  }
	//Check if it goes out right boundary
	else if( end >= lx )
	  {
	    end = lx;
	    collided = 0;
	    endcell = numCells - 1;
	  }
	else
	  {
	    collided = 1;
	    num_col++;
	    endcell = (int)( end * cellsize_recip );
	  }
	next = end;
	
	//amke start <= end and startcell <= endcell
	if( mu < 0.0 )
	  {
	    end = start;
	    start = next;
	    mu = -mu;
	    tempcell = startcell;
	    startcell = endcell;
	    endcell = tempcell;
	  }
	assert(end >= start );
		
	// precompute per-loop constants, to be used below
	/* put mu in a nicer format (reciprocal and respecting epsilon boundaries*/ 
	efmu = (mu > eps) ? mu : eps/2;					  			/* with epsilon boundaries */ 
	efmu_recip = 1.0 / efmu;												/* reciprocal */ 
	efmu_recipS2 = efmu_recip*efmu_recip;						/* for STDEV */ 
	/* pre-compute repeatedly used measurements*/    
	weight_efmu_recip = weight*efmu_recip;
	weightS2_efmu_recipS2 = weightS2*efmu_recipS2; 
	// precompute values for inner loop
	weight_cellsize_efmu_recip = (afloat)weight_efmu_recip*cellsize; 
	weightS2_cellsizeS2_efmu_recipS2 = (afloat)weightS2_efmu_recipS2*cellsizeS2; 

	
	// tally up the information from the corner cells
	if (startcell == endcell){ 
	  /* tally once, with difference of particles */ 
	  absdist = end-start; 
	  tallies->phi_n[startcell] += absdist * weight_efmu_recip; 
	  tallies->phi_n2[startcell] += absdist * absdist * weightS2_efmu_recipS2; 
	} 
	else{ 
	  /* starting cell */ 
	  begin_face = (startcell + 1) * cellsize; 
	  absdist = begin_face - start; /* otherwise -0.0f can mess things up */ 
	  assert(absdist >= 0); 
	  tallies->phi_n[startcell] += absdist * weight_efmu_recip; 
	  tallies->phi_n2[startcell] += absdist * absdist *  weightS2_efmu_recipS2; 
	  /* ending cell */ 
	  end_face = endcell * cellsize; 
	  absdist = end - end_face;	/* otherwise -0.0f can mess things up */ 
	  assert(absdist >= 0); 
	  tallies->phi_n[endcell] += absdist * weight_efmu_recip; 
	  tallies->phi_n2[endcell] += absdist * absdist *  weightS2_efmu_recipS2; 
	  
	  /* tally "internal" cells (should be vectorizable!) */
	  for (k = startcell+1; k < endcell; k++) { 
	    tallies->phi_n[k] += weight_cellsize_efmu_recip; 
	    tallies->phi_n2[k] += weightS2_cellsizeS2_efmu_recipS2; 
	  }
	  
	}
      }
      while( collided && ( ( do_collision = unirandMT_0_1(dsfmt) ) < (data.sig_s * data.sig_t_recip) ) );

      col_tot += ( do_collision < data.sig_s * data.sig_t_recip ? num_col : num_col - 1 );
    }
  }
  
  //printf("Time for SubTally accumulation per Iteration: %lld u_sec\n", totInner);
#ifndef NDEBUG
  printf("Counter = %d, average mu = %lf, average fx = %lf\n",counter,mu_avg/counter,fx_avg/counter);   
#endif
  return col_tot;
}


// calculate the mean and stdev of phi
// assumes that phi^2 was pre-calculated
// value is phi, value2 is phi^2
// NP is used for scaling, along with scale
void mean_std_calc(afloat * value, afloat * value2, long long int samp_cnt, long long NP, int arrLen, pfloat scale, meanDev* retvals){
  int i;
  for (i = 0; i < arrLen; i++) {
    retvals[i].mean = (value[i] * NP * scale) / samp_cnt;
    retvals[i].stdDev = sqrt(fabs((value2[i] / samp_cnt) - pow(retvals[i].mean, 2.0f)) / (samp_cnt - 1));
  }
}


//Function to allocate tallies data structure. 
colData allocate_tallies(int ind_size)
{
  const int cache_line_size = 16;
  const int log_line_size = 4;
  colData to_ret;
  int mod_ind_size = ( (ind_size & ( cache_line_size - 1)) == 0 ? ind_size : ( (ind_size >> log_line_size) + 1) << log_line_size ) ;
  posix_memalign((void**)&(to_ret.full_buffer),cache_line_size,sizeof(afloat)*(2*mod_ind_size+1));
  to_ret.phi_n = to_ret.full_buffer;
  to_ret.phi_n2 = to_ret.full_buffer + mod_ind_size;
  to_ret.tot_col = to_ret.full_buffer + 2*mod_ind_size;
  to_ret.buffer_size = (mod_ind_size * 2 + 1);
  return to_ret;
}

//Function to deallocate the tallies data structure
void deallocate_tallies(colData to_ret)
{
  free(to_ret.full_buffer);
}
