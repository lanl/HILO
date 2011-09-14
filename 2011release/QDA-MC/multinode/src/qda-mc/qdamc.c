/* 
 * Monte Carlo simulation, first draft from Matlab source
 *
 * we use NAN to represent the sqrt(-1) flag present in the matlab QDAMC code
 *
 * Created on May 26, 2011, 10:51 AM
 */

#include "qdamc.h"
#include "qdamc_utils.h"
#include "../ecodes.h"
#include "../dSFMT-2.1/dSFMT.h"
#ifdef DO_PAPI
#include "papi.h"
#endif
#ifdef DO_OMP
#include "omp.h"
#endif
#ifdef DO_MPI
#include "mpi.h"
#endif

#define CACHE_LINE_SIZE 64

/* Main Function */
int run(int rank) {

  //stores user input
  long long NP = opt.NP;                                                		// Number of particles
  pfloat sysLeng = opt.sysLen;                                                        	// Length of system
  int numCells = opt.numCells;                                                         	// Number of cells in system
  pfloat tol_std = opt.tol_std;                                                        	// Tolerance for standard deviation calculations
  int runningWidth = opt.runWidth;                                                     	// Number of iterations to skip before starting averaging
  long long NPc = (long long)((pfloat)NP / numCells);                                   // number of particles in cell
  long long NP_tot = NP; 								//total number of particles for the QDAMC
  
  int nprocs;            
#ifdef DO_MPI
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
#else
  nprocs = 1;
#endif

  //Initialization of vital simulation parameters
  pfloat dx = sysLeng / numCells;                                                 	//cell width
  pfloat eps = 0.001;                                                            	//the threshold angle for tallying
  pfloat sig_t = opt.sig_t;                                                             //Total cross-section 
  pfloat sig_s = opt.sig_s;                                                             //Scattering cross-section
  pfloat D = 1.0 / (3 * sig_t );                                                	//diffusion coefficient

  //The source term uniform for now	
  pfloat * q0 = NULL; 
  int i;
  if( rank == 0 ) {
    q0 = (pfloat *) malloc(sizeof (pfloat) * numCells);	
    for (i = 0; i < numCells; i++) {
      q0[i] = 1.0f;
    }
  }
  pfloat q0_lo = 1.0f;

  //Form data structure for simulation parameters
  struct data data;
  data.NP = NP;										//Number of Particles
  data.lx = sysLeng;									//Length of System
  data.nx = numCells;									//Number of Cells
  data.dx = dx;										//Cell Width
  data.dx2 = dx*dx;									// Cell Width SQUARED
  data.dx_recip = 1.0/dx;								// Pre-computed reciprocal
  data.sig_t = sig_t;									//Total Collision Cross-Section
  data.sig_t_recip = 1.0/sig_t;			                           		// Pre-computed reciprocal
  data.sig_s = sig_s;									//Scattering Collision Cross-Section
  data.q0 = q0;                                                                         //Source term per cell ( Uniform )
  data.eps = eps;									//Threshold Angle for Tallying
  data.q0_lo = q0_lo;									//Source Term
  data.D = D;						                 		//Diffusion Coefficient
  data.NP_tot = NP_tot;									// Total Number of Particles
  data.NPc = NPc;									// Number of Particles per Cell

  //Initialize the average values
  int iter = 0;               								//tally for iteration for convergence
  int iter_avg;         								// iteration for averaging
  if( runningWidth == 0 )
    iter_avg = 0;  
  else
    iter_avg = 1;
  afloat phi_left_tot = 0.0; 								//tally for average phi on left wall
  afloat phi_right_tot = 0.0;   							//tally for average phi on right wall
  afloat J_left_tot = 0.0;  								//tally for average J on left wall
  afloat J_right_tot = 0.0; 								//tally for average J on right wall
  afloat E_left_tot = 0.0;  								//tally for average E on left wall
  afloat E_right_tot = 0.0; 								//tally for average E on right wall

  // Tally for averaging the parameters above
  pfloat phi_left_avg = 0.0;
  pfloat phi_right_avg = 0.0;
  pfloat J_left_avg = 0.0;
  pfloat J_right_avg = 0.0;
  pfloat E_left_avg = 0.0;
  pfloat E_right_avg = 0.0;

  //time keeping structures
  struct timeval start_time, end_time, startPerIter, endPerIter;

  afloat * phi_n_tot = NULL;                                                           //tally of phi from the higher order                                                            
  afloat * phiS2_n_tot = NULL;
  afloat * phi_lo_tot = NULL;                                                          //tally of phi from the LO solver   
  afloat * E_n_tot = NULL;                                                             //Tally of eddington tensor from higher order
  pfloat * phi_n_avg = NULL;                                                           //Average of phi from higher order
  pfloat * E_n_avg = NULL;                                                             //Average of eddington tensor from higher order 
  pfloat * phi_lo_avg = NULL;                                                          //Average of eddington tensor from higher order
  pfloat * E_ho_n = NULL;                                                              //Values of eddington tensor from one iteration
  pfloat * phi_lo = NULL;                                                              //Values of phi returned by lower order solver
  triMat A_lo;                                                                         //Data structure to hold the tri-daiagonal matrix of the lower order sovler
  pfloat * b_lo = NULL;                                                                //Array to hold the right hand side of the lower order sovler
  pfloat * Qbar = NULL;                                                                //Input to the higher order system
#if defined(L2_1) || defined(L2_2) || defined(L2_3)
  pfloat * anal_soln = NULL;                                                           //Array to hold the analytical solution 
  pfloat l2;
#else
  meanDev * phi_nStats = NULL;                                                         //Values returned from mean and std dev calculations
  unsigned long samp_cnt = 0;               					       //total history tally
#endif

  //ALlocate data structure for averaging on process 0 only
  if(rank == 0){
    phi_n_tot = (afloat *) calloc(numCells, sizeof (afloat));
    phiS2_n_tot = (afloat*) calloc(numCells, sizeof(afloat));
    phi_lo_tot = (afloat *) calloc(numCells, sizeof (afloat)); 	
    E_n_tot = (afloat *) calloc(numCells, sizeof (afloat));
    phi_n_avg = (pfloat *) calloc(numCells, sizeof (pfloat));
    E_n_avg = (pfloat *) calloc(numCells, sizeof (pfloat));
    phi_lo_avg = (pfloat *) calloc(numCells, sizeof (pfloat));

    // Initialize lower order scalar flux using diffusion approximation
    E_ho_n = (pfloat *) malloc(sizeof (pfloat) * numCells);
    for (i = 0; i < numCells; i++) {
      E_ho_n[i] = 1.0f / 3.0f;
    }
    phi_lo = (pfloat*) malloc(data.nx*sizeof(pfloat));
    A_lo.a = (pfloat*) malloc(sizeof(pfloat)*(data.nx-1));
    A_lo.b = (pfloat*) malloc(sizeof(pfloat)*data.nx);
    A_lo.c = (pfloat*) malloc(sizeof(pfloat)*(data.nx-1));
    b_lo = (pfloat*) malloc(sizeof(pfloat)*data.nx);

    
#if defined(L2_1) || defined(L2_2) || defined(L3_2)
    anal_soln = (pfloat *) calloc(numCells, sizeof(pfloat));	
#if defined(L2_1)
    init_L2_norm_1(anal_soln);
#elif defined(L2_2)
    init_L2_norm_2(anal_soln);
#elif defined(L2_3)
    init_L2_norm_3(anal_soln);
#endif
#else
    phi_nStats = (meanDev*) malloc(sizeof(meanDev)*opt.numCells);
#endif

  }
  /************************************************
   * Calculation of Initial Condition
   * TODO account for isotropic and beam source
   *************************************************/
  int flag_sim = 0;
  
  Qbar = malloc(sizeof (pfloat) * numCells);

  //Tallies per threads
  colData * thread_tallies;
#ifdef DO_OMP
  int max_threads = omp_get_max_threads();
#else
  int max_threads = 1;
#endif
  thread_tallies = (colData*) malloc(max_threads*sizeof(colData));

  //Structure to hold the random number generator data structures for each thread
  dsfmt_t** thread_rand;
  thread_rand = (dsfmt_t**)calloc(max_threads,sizeof(dsfmt_t*));
 
  //Allocating data structures to be used by each thread
  for( i = 0 ; i < max_threads ; i++ ) {
    //Allocate all data that is written by the thread to cache line size
    posix_memalign( (void**)&(thread_tallies[i].phi_n), CACHE_LINE_SIZE, numCells*sizeof(afloat));
    posix_memalign( (void**)&(thread_tallies[i].phi_n2), CACHE_LINE_SIZE, numCells*sizeof(afloat));
    posix_memalign( (void**)&(thread_tallies[i].E_n), CACHE_LINE_SIZE, numCells*sizeof(afloat));
    posix_memalign( (void**)(thread_rand+i), CACHE_LINE_SIZE, sizeof(dsfmt_t));
  }
 
  //Tallies across all processes.
  colData * all_tallies;
  //For MPI, allocate the finall tally on process 0
#ifdef DO_MPI
  all_tallies = (colData*) malloc(sizeof(colData));
  if( rank == 0 ) {
    all_tallies->phi_n = (afloat*) malloc(numCells*sizeof(afloat));
    all_tallies->phi_n2 = (afloat*) malloc(numCells*sizeof(afloat));
    all_tallies->E_n = (afloat*) malloc(numCells*sizeof(afloat));
  }
  else{
    all_tallies->phi_n = NULL;
    all_tallies->phi_n2 = NULL;
    all_tallies->E_n = NULL;
  }
#else
  //Otherwise alias all_tallies to tallies of thread 0. 
  all_tallies = &(thread_tallies[0]);
#endif
  
  //Calculation starts here
  //START TIMING
  gettimeofday(&start_time, NULL);

  if( rank == 0){
    //Plot and print initial condition as well as simulation parameters
    if (!opt.silent) sim_param_print(data);

    //calls low order solver for initial condition
    lo_solver(data, 0.0, 0.0, 0.0, 0.0, 1.0f / 3.0f, 1.0f / 3.0f, E_ho_n, 1,phi_lo,A_lo,b_lo);
    free(E_ho_n);
  }

#ifdef DO_OMP
#pragma omp parallel default(shared) private(iter,i)
  {
    int nthreads = omp_get_num_threads();
#else
    int nthreads  = 1;
#endif 
    iter = 0;                   								//tally for iteration for convergence
    
    int nstages = 0;
    i = 1;
    
    while( i < nthreads )
      {
	nstages++;
	i <<= 1 ;
      }
    
#ifdef DO_OMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif

    // MT RNG
    dsfmt_init_gen_rand(thread_rand[thread_id], (int)time(NULL) + thread_id);

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
      
      //start time for iteration
      gettimeofday(&startPerIter, NULL);
      
      iter += 1;
      if(!opt.silent && rank == 0 && thread_id == 0 ) printf("This is the %dth iteration\n",iter);
      
      if( rank == 0 && thread_id == 0 ){
	if( iter == 1 )
	  for (i = 0; i < numCells; i++) {
	    Qbar[i] = fabs(q0[i] + sig_s * phi_lo[i]) * data.lx;	
	  }
	else
	  for (i = 0; i < numCells; i++) {
	    Qbar[i] = fabs(q0[i] + sig_s * phi_lo_avg[i]) * data.lx;	
	  }
      }
	
#ifdef DO_MPI
      if( thread_id == 0 ){
	MPI_Bcast((void*)Qbar,numCells,MPI_DOUBLE,0,MPI_COMM_WORLD);
      }
#endif

#ifdef DO_OMP 
#pragma omp barrier
#endif 
      //Simulate the Higher Order System using Monte Carlo
      collision_and_tally(data, Qbar,thread_tallies+ thread_id,thread_id,nthreads,thread_rand[thread_id],rank,nprocs);

#ifdef DO_OMP 
      //COMBINE TALLIES HERE - TREE REDUCTION
      int factor = 1;
      int k;
      for( i = 0 ; i < nstages ; i++ )
	{
#pragma omp barrier
	  if( ( (thread_id % (factor << 1) ) == 0 ) && ( thread_id + factor < nthreads ) )
	    {
	      for( k = 0 ; k < numCells ; k++ )
		{
		  thread_tallies[thread_id].phi_n[k] += thread_tallies[thread_id+factor].phi_n[k];
		  thread_tallies[thread_id].phi_n2[k] += thread_tallies[thread_id+factor].phi_n2[k];
		  thread_tallies[thread_id].E_n[k] += thread_tallies[thread_id+factor].E_n[k];
		}
	      thread_tallies[thread_id].phi_left += thread_tallies[thread_id+factor].phi_left;
	      thread_tallies[thread_id].phi_right += thread_tallies[thread_id+factor].phi_right;
	      thread_tallies[thread_id].J_left += thread_tallies[thread_id+factor].J_left;
	      thread_tallies[thread_id].J_right += thread_tallies[thread_id+factor].J_right;
	      thread_tallies[thread_id].E_left += thread_tallies[thread_id+factor].E_left;
	      thread_tallies[thread_id].E_right += thread_tallies[thread_id+factor].E_right;
	    }
	  factor <<= 1;
	}
#endif 
      
#ifdef DO_MPI
      //AFTER THREAD reduction, reduction across MPI processes
      if( thread_id == 0 )
	{
	  MPI_Reduce((void*)thread_tallies[0].phi_n,(void*)(all_tallies->phi_n),numCells,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	  MPI_Reduce((void*)thread_tallies[0].phi_n2,(void*)(all_tallies->phi_n2),numCells,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	  MPI_Reduce((void*)thread_tallies[0].E_n,(void*)(all_tallies->E_n),numCells,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	  MPI_Reduce((void*)&(thread_tallies[0].phi_left),(void*)&(all_tallies->phi_left),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	  MPI_Reduce((void*)&(thread_tallies[0].phi_right),(void*)&(all_tallies->phi_right),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	  MPI_Reduce((void*)&(thread_tallies[0].J_left),(void*)&(all_tallies->J_left),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	  MPI_Reduce((void*)&(thread_tallies[0].J_right),(void*)&(all_tallies->J_right),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	  MPI_Reduce((void*)&(thread_tallies[0].E_left),(void*)&(all_tallies->E_left),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	  MPI_Reduce((void*)&(thread_tallies[0].E_right),(void*)&(all_tallies->E_right),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	}
#endif

      if( rank == 0 && thread_id == 0 ){

	//Normalize all the tallied data
	if (!floatEquals(all_tallies->phi_left, 0.0, FLOAT_TOL)) {
	  all_tallies->E_left /= all_tallies->phi_left;
	} else {
	  all_tallies->E_left = (1.0 / 3.0);
	  
	}
	if (!floatEquals(all_tallies->phi_right, 0.0, FLOAT_TOL)) {
	  all_tallies->E_right /= all_tallies->phi_right;
	} else {
	  all_tallies->E_right = (1.0 / 3.0);
	}
	all_tallies->phi_left /= (afloat)NP;
	all_tallies->J_left /= (afloat)NP;
	all_tallies->phi_right /= (afloat)NP;
	all_tallies->J_right /= (afloat)NP;

	for (i = 0; i < numCells; i++) {
	  if (!floatEquals(all_tallies->phi_n[i], 0.0, FLOAT_TOL)) {
	    all_tallies->E_n[i] /= all_tallies->phi_n[i];
	  } else {
	    all_tallies->E_n[i] = (1.0 / 3.0);
	  }
	  all_tallies->phi_n[i] /= (afloat)(dx*NP);
	}
	  
	/***************************************************
	 * Calculates the averages
	 **************************************************/
	if( iter <= runningWidth )
	  {
	    for (i = 0; i < numCells; i++) {
	      phi_n_tot[i] = all_tallies->phi_n[i];
	      phiS2_n_tot[i] = all_tallies->phi_n2[i];
	      phi_lo_tot[i] = phi_lo[i];
	      E_n_tot[i] = all_tallies->E_n[i];
	    }
	    phi_left_tot = all_tallies->phi_left;
	    phi_right_tot = all_tallies->phi_right;
	    E_left_tot = all_tallies->E_left;
	    E_right_tot = all_tallies->E_right;
	    J_left_tot = all_tallies->J_left;
	    J_right_tot  = all_tallies->J_right;

	    // for each cell, calculate the average for phi_n, phi_lo and E_n
	    for (i = 0; i < numCells; i++) {
	      phi_n_avg[i] = phi_n_tot[i] ;
	      phi_lo_avg[i] = phi_lo_tot[i];
	      E_n_avg[i] = E_n_tot[i] ;
	    }

	    // calculate the average for left and right faces of phi, E and J
	    phi_left_avg = phi_left_tot;
	    phi_right_avg = phi_right_tot;
	    E_left_avg = E_left_tot ;
	    E_right_avg = E_right_tot ;
	    J_left_avg = J_left_tot ;
	    J_right_avg = J_right_tot;
	  }
	else{


	  iter_avg += 1;

	  //accumulate phase -- add new data to the running averages
	  for (i = 0; i < numCells; i++) {
	    phi_n_tot[i] += all_tallies->phi_n[i];
	    phiS2_n_tot[i] += all_tallies->phi_n2[i];
	    phi_lo_tot[i] += phi_lo[i];
	    E_n_tot[i] += all_tallies->E_n[i];
	  }
	  phi_left_tot += all_tallies->phi_left;
	  phi_right_tot += all_tallies->phi_right;
	  E_left_tot += all_tallies->E_left;
	  E_right_tot += all_tallies->E_right;
	  J_left_tot += all_tallies->J_left;
	  J_right_tot += all_tallies->J_right;

	  // for each cell, calculate the average for phi_n, phi_lo and E_n
	  for (i = 0; i < numCells; i++) {
	    phi_n_avg[i] = phi_n_tot[i] / iter_avg;
	    phi_lo_avg[i] = phi_lo_tot[i] / iter_avg;
	    E_n_avg[i] = E_n_tot[i] / iter_avg;
	  }

	  // calculate the average for left and right faces of phi, E and J
	  phi_left_avg = phi_left_tot / iter_avg;
	  phi_right_avg = phi_right_tot / iter_avg;
	  E_left_avg = E_left_tot / iter_avg;
	  E_right_avg = E_right_tot / iter_avg;
	  J_left_avg = J_left_tot / iter_avg;
	  J_right_avg = J_right_tot / iter_avg;

	  //check for convergence
#if !defined(L2_1) && !defined(L2_2) && !defined(L2_3)
	  // for each cell, do adds for phi_ho_s1_n and phi_ho_s2_n
	  samp_cnt = NP_tot * ((long) iter_avg);
	  mean_std_calc(phi_n_tot, phiS2_n_tot, samp_cnt, opt.NP, opt.numCells, opt.dx, phi_nStats);

	  printf("The maximum standard deviation of flux at node is, max (sig_phi) = %f\n", maxFAS(&phi_nStats[0].stdDev, numCells, 2));

	  if (maxFAS(&phi_nStats[0].stdDev, numCells, 2) <= tol_std) {
	    flag_sim = 1;
	  }
#else
	  l2 = l2_norm_cmp(phi_n_avg, anal_soln, numCells, dx);
	  flag_sim = l2 <= tol_std;
	  if(!opt.silent && rank == 0){
	    gettimeofday(&end_time, NULL);
	    printf("L2: %f, Sofar: %ldu_sec\n", l2, (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));
	  }
#endif
	}

	//Solve the lower order system using discretizion methods
	lo_solver(data, phi_left_avg, phi_right_avg, J_left_avg, J_right_avg, E_left_avg, E_right_avg, E_n_avg, 0, phi_lo, A_lo,b_lo);
	  
      }

#ifdef DO_MPI
      if( thread_id == 0 )
	MPI_Bcast(&flag_sim,1,MPI_INT,0,MPI_COMM_WORLD);
#endif

#ifdef DO_OMP 
#pragma omp barrier
#endif 
      //end time per iteration
      gettimeofday(&endPerIter, NULL);
      //printf("ID = %d, thread_id = %d, Time per Iteration: %ldu_sec\n\n",rank, thread_id, (endPerIter.tv_sec - startPerIter.tv_sec)*1000000 + (endPerIter.tv_usec - startPerIter.tv_usec));
    }
#ifdef DO_PAPI
    PAPI_read(EventSet,stop);
    printf("%lld %lld %lld %lld\n",stop[0] - start[0],stop[1] - start[1],stop[2] - start[2],stop[3] - start[3]);
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
#endif

    
    free(thread_tallies[thread_id].phi_n);
    free(thread_tallies[thread_id].phi_n2);
    free(thread_tallies[thread_id].E_n);
#ifdef DO_OMP 
  }
#endif 
  /************************************************
   * Free memory
   *************************************************/
  //PAUL END TIMING
  gettimeofday(&end_time, NULL);
  
  
  free(thread_tallies);
  free(thread_rand);
  if( rank == 0 ){

    if( !opt.silent ){
      printf("NODEAVG\n");
      for (i = 0; i < numCells; i++) {
	printf("%d %f %f %f\n", i, phi_n_avg[i], E_n_avg[i], phi_lo_avg[i]);
      }
    }
    printf("Elapsed Time: %ldu_sec\n", (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));
    
    free(A_lo.a);
    free(A_lo.b);
    free(A_lo.c);
    free(b_lo);
    free(phi_lo);
#ifdef DO_MPI
    free(all_tallies->phi_n);
    free(all_tallies->phi_n2);
    free(all_tallies->E_n);
#endif
    free(Qbar);
    free(q0);
    free(phi_n_tot);
    free(phi_lo_tot);
    free(E_n_tot);
    free(phi_n_avg);
    free(E_n_avg);
    free(phi_lo_avg);
    
#if defined(L2_1) || defined(L2_2) || defined(L2_3)
    free(anal_soln);
#else
    free(phi_nStats);
#endif
  }
#ifdef DO_MPI
  free(all_tallies);
#endif
  
  //PAUL OUTPUT SECTION
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
  int NP = data.NP;
  pfloat sig_t = data.sig_t;
  pfloat sig_s = data.sig_s;

  printf("***********************************************************\n");
  printf("******THE SIMULATION PARAMETERS ARE PRINTED OUT BELOW******\n");
  printf("The system length is, lx = %f\n", lx);
  printf("The cell width is, dx = %f\n", dx);
  printf("The number of cell is, nx = %d\n", nx);
  printf("The reference number of particle is, NP = %d\n", NP);
  printf("The total cross section is, sig_t = %f\n", sig_t);
  printf("The scattering cross section is, sig_s = %f\n", sig_s);
  printf("Floating point data representation is, %lu byte\n", sizeof (pfloat));
  printf("***********************************************************\n");
}

/******************************************************************************************************
 * lo_solver
 *
 * Function call that does the low order solve for a system of equations
 *
 * @param data				data structure that holds initialization and simulation parameters
 * @param phi_ho_left
 * @param phi_ho_right
 * @param J_ho_left
 * @param J_ho_right
 * @param E_ho_left
 * @param E_ho_right
 * @param E_ho_n
 * @param ic				flag for initial condition or not
 *
 * @return array of floats/doubles
 *
 * TODO naming conventions should be more abstract since it is used with a variety of input variables
 *****************************************************************************************************/
void lo_solver(struct data data, pfloat phi_ho_left, pfloat phi_ho_right, pfloat J_ho_left, pfloat J_ho_right, pfloat E_ho_left, pfloat E_ho_right, pfloat * E_ho_n, int ic, pfloat* phi_lo,triMat A, pfloat* b) {

  int i;

  //Initialize simulation parameters
  int nx = data.nx;
  pfloat dx = data.dx;
  pfloat sig_t = data.sig_t;
  pfloat sig_s = data.sig_s;
  pfloat alp = 1.0f / (dx * dx * sig_t);
  pfloat beta = sig_t - sig_s;
  pfloat gam = sig_s / (dx * sig_t * sig_t);
  pfloat D4 = data.D * 4.0f;
  pfloat q0_lo = data.q0_lo;
  /* pfloat * b = (pfloat *) malloc(sizeof (pfloat) * nx); */
  for (i = 0; i < nx; i++) {
    b[i] = q0_lo;
  }

  if (ic == 0) { //not initial condition

    //declare the boundary values for the lo system
    pfloat phi_l = E_ho_n[0] / (E_ho_left - J_ho_left / (phi_ho_left * 2.0f * gam));
    pfloat phi_r = E_ho_n[nx - 1] / (E_ho_right + J_ho_right / (phi_ho_right * 2.0f * gam));

    for (i = 0; i < nx; i++) {
      if (i == 0) { //left boundary cell
	A.b[i] = beta + (gam * E_ho_n[0] * phi_ho_left - phi_l * J_ho_left) / (phi_ho_left * dx);
	A.c[i] = -gam * E_ho_n[1] / dx;
      } else if (i == nx - 1) { //right boundary cell
	A.b[i] = beta + (gam * E_ho_n[nx - 1] * phi_ho_right + phi_r * J_ho_right) / (phi_ho_right * dx);
	A.a[i - 1] = -gam * E_ho_n[nx - 2] / dx;
      } else { //internal cell
	A.b[i] = 2.0f * alp * E_ho_n[i] + beta;
	A.c[i] = -alp * E_ho_n[i + 1];
	A.a[i - 1] = -alp * E_ho_n[i - 1];
      }
    }
  } else { //initial condition

    //declare the boundary values for the lo system
    pfloat phi_l = (D4 / (dx + D4));
    pfloat phi_r = (D4 / (dx + D4));

    for (i = 0; i < nx; i++) {
      if (i == 0) { //left boundary cell
	A.b[i] = beta + alp * (2.0f * E_ho_left * phi_l - 3.0f * E_ho_n[0]);
	A.c[i] = alp * E_ho_n[1];
      } else if (i == nx - 1) { //right boundary cell
	A.b[i] = beta + alp * (2.0f * E_ho_right * phi_r - 3.0f * E_ho_n[nx - 1]);
	A.a[i - 1] = alp * E_ho_n[nx - 2];
      } else { //internal cell
	A.b[i] = 2.0f * alp * E_ho_n[i] + beta;
	A.c[i] = -alp * E_ho_n[i + 1];
	A.a[i - 1] = -alp * E_ho_n[i - 1];
      }
    }
  }

  triSolve(nx, A.a, A.b, A.c, b, phi_lo);
}

/******************************************************************************************************
 * triSolve
 *
 * Function the does a tridiagonal matrix solve
 * Entirely from http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
 * Other than changing to pre-c99 for loops, and switching floats for floats
 *
 * @param n		number of equations
 * @param a		sub-diagonal (means it is the diagonal below the main diagonal)
 * @param b		the main diagonal
 * @param c		sup-diagonal (means it is the diagonal above the main diagonal)
 * @param v		right part
 * @param x		the answer
 *
 * @return void
 *****************************************************************************************************/
 void triSolve(int n, pfloat *a, pfloat *b, pfloat *c, pfloat *v, pfloat *x) {
  int i;
  for (i = 1; i < n; i++) {
    pfloat m = a[i - 1] / b[i - 1];
    b[i] = b[i] - m * c[i - 1];
    v[i] = v[i] - m * v[i - 1];
  }

  x[n - 1] = v[n - 1] / b[n - 1];

  for (i = n - 2; i >= 0; i--)
    x[i] = (v[i] - c[i] * x[i + 1]) / b[i];
}

/**********************************************************************************************************
 * collision_and_tally
 *
 * 
 *
 * @param data
 * @param Qbar
 *
 * @return colData
 *
 * TODO modularize and more optimized parallel reduction and/or alternate way to get data
 *********************************************************************************************************/
void collision_and_tally(struct data data, pfloat * Qbar, colData * tallies, int thread_id, int nthreads, dsfmt_t * dsfmt, int rank, int nprocs) {
  //initialize simulation parameters
  int nx = data.nx;
  int NP = data.NP;
  	
  //stream the particles, collide and calculate moment all simultaneously
  //calculate the final location of the particle
  //initialize tallies and corresponding variables
  memset(tallies->phi_n,0,sizeof(afloat)*nx);
  memset(tallies->phi_n2,0,sizeof(afloat)*nx);
  memset(tallies->E_n,0,sizeof(afloat)*nx);
  tallies->E_left = 0.0;
  tallies->J_left = 0.0;
  tallies->phi_left = 0.0;
  tallies->E_right = 0.0;
  tallies->J_right = 0.0;
  tallies->phi_right = 0.0;


  struct timeval startBoth, endBoth; 

  int localnx,localstart;

  //Each process spawns particles in one cell of the domain and streams it. Divide up the cells among all threads on all the MPI processes more or less evenly
#ifdef DO_MPI
  int lower_proc = nx/nprocs;
  int higher_proc = lower_proc + 1;
  int switch_proc =  higher_proc * nprocs - nx;
  if( rank < switch_proc )
    {
      localstart = lower_proc * rank;
      localnx = lower_proc;
    }
  else
    {
      localstart = lower_proc * switch_proc + ( rank - switch_proc ) * higher_proc;
      localnx = higher_proc;
    }
#else
  localnx = nx;
  localstart = 0;
#endif

  int start_val,end_val;

#ifdef DO_OMP 
  int local_lower = localnx / nthreads;
  int local_upper = local_lower + 1;
  int switch_thread = local_upper * nthreads - localnx;
  if( thread_id < switch_thread )
    {
      start_val = localstart + thread_id * local_lower;
      end_val = start_val + local_lower;
    }
  else
    {
      start_val = localstart + switch_thread * local_lower + ( thread_id - switch_thread ) * local_upper ;
      end_val = start_val + local_upper;
    }
#else
  start_val = localstart;
  end_val = localstart + localnx;
#endif

  
  if (NP != 0) {
    //start time for montecarlo
    //gettimeofday(&startBoth, NULL);
    
    gen_particles_and_tally(data, Qbar, tallies, start_val, end_val, dsfmt);
    
    //end time for montecarlo
    //gettimeofday(&endBoth, NULL);
    //printf("Time for Monte Carlo per Iteration: %ld u_sec\n", (endBoth.tv_sec - startBoth.tv_sec)*1000000 + (endBoth.tv_usec - startBoth.tv_usec));
  }
}

//Function to spawn particles and stream them
//INPUT: 1) data : the physical parameters of the system
//       2) The source term per cell
//       3) Pointer to structure that holds the tallies
//       4) The number of the cell the thread starts spawning particles in
//       5) The number of the cell the threads stops spawning particles in
void gen_particles_and_tally(struct data data, pfloat *Qbar, colData *tallies, int start_iter_cell, int end_iter_cell, dsfmt_t * dsfmt){
  
  const pfloat lx = data.lx, eps = data.eps;                                                          //System length and cut off value for mu
  const int numCells = data.nx;
  const pfloat sig_t_recip = data.sig_t_recip;                                                       
  const pfloat cellsize = data.dx, cellsize_recip = data.dx_recip, cellsizeS2 = data.dx2;             //size of cell, its reciprocal and its square
  const long long NPc = data.NPc;                                                                     //Number of particles per cell
  
  //Temporaries
  pfloat fx, mu, efmu, efmu_recip, start, end;
  pfloat weight;
  unsigned int num_right;
  int i, j, k;
  pfloat efmu_recipS2;
  pfloat weightS2;
  pfloat absdist;
  int startcell, endcell;
  pfloat begin_rightface, end_leftface;
  pfloat weight_efmu, weight_efmu_recip, weightS2_efmu_recipS2, weight_cellsize;
  pfloat weight_cellsize_efmu_recip, weightS2_cellsizeS2_efmu_recipS2, weight_cellsize_efmu;

#ifndef NDEBUG
  pfloat fx_avg = 0.0;
  long long NP = data.NP;
#endif

  /* create and tally all particles */
  for (i = start_iter_cell; i < end_iter_cell; i++) {
    weight = Qbar[i];											// weight of each particle
    weightS2 = weight*weight;						                        	// for STDEV

    // Half the particles are going to the right
    num_right = NPc>>1;

    // Stream right-moving particles
    for (j = 0; j < num_right; j++) {

      // find angle, distance travelled, and absolute start and end positions
      mu = unirandMT_0_1(dsfmt); 
      fx = (-log(unirandMT_0_1(dsfmt)) * sig_t_recip); 
      start = (i  +  unirandMT_0_1(dsfmt)) * cellsize;
      end = start + mu * fx;	
#ifndef NDEBUG
      fx_avg += fx;
#endif
      
      efmu = (mu > eps) ? mu : eps/2;	                                                  		/* if mu is too small, replace it with eps/2 */
      efmu_recip = 1.0 / efmu;                                                  			/* reciprocal */ 
      efmu_recipS2 = efmu_recip*efmu_recip;                                              		/* for STDEV */ 
      /* pre-compute repeatedly used measurements*/  
      weight_efmu_recip = weight*efmu_recip;	
      weight_efmu = weight*efmu;		   
      weight_cellsize = weight*cellsize;
      weightS2_efmu_recipS2 = weightS2*efmu_recipS2;	

      /* corner-case analysis, right-moving particles */      
      // left-most cell is the starting cell
      startcell = i; 
      // right-most cell is the ending cell. 
      //If particles streams out
      if (end >= lx) {	
	end = lx;
	endcell = numCells-1;
	
	// for tallying, record this corner case
	tallies->phi_right += weight_efmu_recip; //weight * efmu_recip;//
	tallies->E_right += weight_efmu; //weight * efmu;//
	tallies->J_right += weight;
      }
      else {
	endcell = (int)(end * cellsize_recip);
      }
      
      // tally up the information from the corner cells		
      if (startcell == endcell){					
	/* tally once, with difference of particles */			
	absdist = (end-start);						
	tallies->phi_n[startcell] += absdist * weight_efmu_recip;	
	tallies->phi_n2[startcell] += absdist * absdist * weightS2_efmu_recipS2; 
	tallies->E_n[startcell] += absdist * weight_efmu;		
      }									
      else{								
	/* starting cell */						
	begin_rightface = (startcell + 1) * cellsize;			
	absdist = (begin_rightface - start); /* otherwise -0.0f can mess things up */ 
	assert(absdist >= -0.0f);					
	tallies->phi_n[startcell] += absdist * weight_efmu_recip;	
	tallies->phi_n2[startcell] += absdist * absdist * weightS2_efmu_recipS2; 
	tallies->E_n[startcell] += absdist * weight_efmu;		
	/* ending cell */						
	end_leftface = endcell * cellsize;				
	absdist = (end - end_leftface);	/* otherwise -0.0f can mess things up */ 
	assert(absdist >= -0.0f);					
	tallies->phi_n[endcell] += absdist * weight_efmu_recip;		
	tallies->phi_n2[endcell] += absdist * absdist * weightS2_efmu_recipS2; 
	tallies->E_n[endcell] += absdist * weight_efmu;			
      }									
      
      // precompute values for inner loop
      weight_cellsize_efmu_recip = (afloat)weight_efmu_recip*cellsize;	
      weightS2_cellsizeS2_efmu_recipS2 = (afloat)weightS2_efmu_recipS2*cellsizeS2; 
      weight_cellsize_efmu = (afloat)weight_efmu*cellsize;		

      for (k = startcell+1; k <= endcell-1; k++) {			
	tallies->phi_n[k] += weight_cellsize_efmu_recip;		
	tallies->phi_n2[k] += weightS2_cellsizeS2_efmu_recipS2;		
	tallies->E_n[k] += weight_cellsize_efmu;			
      } 
      
      // sanity checks
      assert(startcell <= endcell);					
      assert(startcell >= 0 && endcell <= numCells - 1);		
      assert(start <= end);						
      assert(start >= 0.0);						
      assert(end <= lx);						
    }
    
    // left-moving particles
    for (j = 0; j < NPc-num_right; j++) {

      // find angle, distance travelled, and absolute start and end positions
      mu = unirandMT_0_1(dsfmt);			
      fx = (-log(unirandMT_0_1(dsfmt)) * sig_t_recip); 
      end = (i  +  unirandMT_0_1(dsfmt)) * cellsize;
      start = end - mu * fx;	
#ifndef NDEBUG
      fx_avg += fx;
#endif
      
      // precompute per-loop constants, to be used below
      efmu = (mu > eps) ? mu : eps/2;			/* with epsilon boundaries */ 
      efmu_recip = 1.0 / efmu;				/* reciprocal */ 
      efmu_recipS2 = efmu_recip*efmu_recip;		/* for STDEV */ 
      /* pre-compute repeatedly used measurements*/			
      weight_efmu_recip = weight*efmu_recip;				
      weight_efmu = weight*efmu;					
      weight_cellsize = weight*cellsize;				
      weightS2_efmu_recipS2 = weightS2*efmu_recipS2;			
      
      /* corner-case analysis, left-moving particles */

      // right-most cell is the starting cell
      endcell = i;
      // left-most cell is the starting cell
      //If the particle streams out
      if (start <= 0.0) {	// corner case
	start = 0.0;
	startcell = 0;	
	
	// for tallying, record this corner case
	tallies->phi_left += weight * efmu_recip;//weight_efmu_recip; 
	tallies->E_left += weight * efmu; //weight_efmu; 
	tallies->J_left += -weight;	// note: sign is negative!
      }
      else {
	startcell = (int)(start * cellsize_recip);
      }					
      
      
      // tally up the information from the corner cells
      /* tally corner cells */ 
      if (startcell == endcell){ 
	/* tally once, with difference of particles */ 
	absdist = (end-start); 
	tallies->phi_n[startcell] += absdist * weight_efmu_recip; 
	tallies->phi_n2[startcell] += absdist * absdist * weightS2_efmu_recipS2; 
	tallies->E_n[startcell] += absdist * weight_efmu; 
      } 
      else{ 
	/* starting cell */ 
	begin_rightface = (startcell + 1) * cellsize; 
	absdist = (begin_rightface - start); /* otherwise -0.0f can mess things up */ 
	assert(absdist >= -0.0f); 
	tallies->phi_n[startcell] += absdist * weight_efmu_recip; 
	tallies->phi_n2[startcell] += absdist * absdist * weightS2_efmu_recipS2; 
	tallies->E_n[startcell] += absdist * weight_efmu; 
	/* ending cell */ 
	end_leftface = endcell * cellsize; 
	absdist = (end - end_leftface);	/* otherwise -0.0f can mess things up */ 
	assert(absdist >= -0.0f); 
	tallies->phi_n[endcell] += absdist * weight_efmu_recip; 
	tallies->phi_n2[endcell] += absdist * absdist * weightS2_efmu_recipS2; 
	tallies->E_n[endcell] += absdist * weight_efmu; 
      } 

      // precompute values for inner loop
      weight_cellsize_efmu_recip = (afloat)weight_efmu_recip*cellsize;	
      weightS2_cellsizeS2_efmu_recipS2 = (afloat)weightS2_efmu_recipS2*cellsizeS2; 
      weight_cellsize_efmu = (afloat)weight_efmu*cellsize; 

      for (k = startcell+1; k <= endcell-1; k++) {			
	tallies->phi_n[k] += weight_cellsize_efmu_recip;		
	tallies->phi_n2[k] += weightS2_cellsizeS2_efmu_recipS2;		
	tallies->E_n[k] += weight_cellsize_efmu;			
      }

      //Sanity checks
      assert(startcell <= endcell);					
      assert(startcell >= 0 && endcell <= numCells - 1);		
      assert(start <= end);						
      assert(start >= 0.0);						
      assert(end <= lx);						
    }
  }
#ifndef NDEBUG
  printf("Avg fx = %lf\n",fx_avg / ( numCells * NP ));
#endif
}

// calculate the mean and stdev of phi
// assumes that phi^2 was pre-calculated
// value is phi, value2 is phi^2
// NP is used for scaling, along with scale
void mean_std_calc(afloat * value, afloat * value2, unsigned long int samp_cnt, int NP, int arrLen, pfloat scale, meanDev* retvals){
  int i;
  for (i = 0; i < arrLen; i++) {
    retvals[i].mean = (value[i] * NP * scale) / samp_cnt;
    retvals[i].stdDev = sqrt(fabs((value2[i] / samp_cnt) - pow(retvals[i].mean, 2.0f)) / (samp_cnt - 1));
  }
}


