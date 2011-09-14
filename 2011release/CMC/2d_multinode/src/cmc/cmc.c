/* LANL IS&T Summer Co-Design School. Summer 2011
 * Mahesh Ravishankar : 2D Monte Carlo simulation
 *
 * Created on Aug 21, 2011
 */

#include "emmintrin.h"
#include "cmc.h"
#include "cmc_utils.h"
#include "../ecodes.h"
#ifdef DO_PAPI
#include "papi.h"
#endif
#ifdef DO_MPI
#include "mpi.h"
#endif

#define PI 3.1415926535897932384626433832795028841971693993

colData allocate_tallies(int);
void deallocate_tallies(colData);

/*
 * Main Function
 */
int run(int rank) {

  long long NP;	       	                		// Number of particles
  pfloat sysLengx;		                        // Length of system along x axis
  int numCellsx;		                        // Number of cells in system along x axis
  pfloat sysLengy;		                        // Length of system along y axis
  int numCellsy;		                        // Number of cells in system along y axis
  int numCells;                                         // Total number of cells = numCellsx * numCellsy
  pfloat tol;	                                        // Relative difference tolerance for convergence
  pfloat tol_std;		                        // Tolerance for standard deviation calculations
  /* TEMPORARY */

  //stores user input
  NP = opt.NP;
  sysLengx = opt.sysLenx;
  numCellsx = opt.numCellsx;
  sysLengy = opt.sysLeny;
  numCellsy = opt.numCellsy;
  numCells = numCellsx * numCellsy;
  tol = opt.tol;
  tol_std = opt.tol_std;

  int nprocs;
#ifdef DO_MPI
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
#else
  nprocs = 1;
#endif

  //Initialization of vital simulation parameters
  pfloat dx = sysLengx / numCellsx;                    	//cell width
  pfloat dy = sysLengy / numCellsy;                    	//cell width
  pfloat eps = 0.01;                                    //the threshold angle for tallying
  pfloat sig_t = opt.sig_t;                            	//total collision cross-section
  pfloat sig_s = opt.sig_s;

  //The source term uniform for now	
  pfloat q0_avg = 1.0 ;

  //Form data structure for simulation parameters
  struct data data;
  data.NP = NP;						//Number of Particles
  data.lx = sysLengx;					//Length of System along x
  data.nx = numCellsx;					//Number of Cells along x
  data.dx = dx;						//Cell Width along x
  data.dx_recip = 1.0 / dx;                             //reciprocal of cellwidth along x
  data.ly = sysLengy;					//Length of System along y
  data.ny = numCellsy;					//Number of Cells along y
  data.dy = dy;						//Cell Width along y
  data.dy_recip = 1.0/ dy;                              //reciprocal of cell width along y
  data.nc = data.nx * data.ny;                          //Total number of cells 
  data.sig_t = sig_t;					//Total Collision Cross-Section
  data.sig_t_recip = 1.0/sig_t;			        // Pre-computed reciprocal
  data.sig_s = sig_s;					//Scattering Collision Cross-Section
  data.q0_avg = q0_avg;                                 //The source term for HO equation.
  data.eps = eps;					//Threshold Angle for Tallying
  data.reflectx0 = opt.reflectx0;                       //Is x=0 a reflecting surface ( 0 if not, 1 if it is) 
  data.reflectxN = opt.reflectxN;                       //Is x=lx a reflecting surface ( 0 if not, 1 if it is)  
  data.reflecty0 = opt.reflecty0;                       //Is y=0 a reflecting surface ( 0 if not, 1 if it is)    
  data.reflectyN = opt.reflectyN;                       //Is y=ly a reflecting surface ( 0 if not, 1 if it is) 

  
  long long NPc = (long long )((pfloat)NP / (numCells));       // number of particles in cell
  long long NP_tot = NP; 			        //total number of particles for the CMC
  data.NP_tot = NP_tot;				        // Total Number of Particles
  data.NPc = NPc;					// Number of Particles per Cell

  //Initialize the average values
  int iter = 0;                                         //tally for iteration for convergence
  int iter_avg = 0;                                     // iteration for averaging
  int i,j;

  //time keeping structures
  struct timeval start_time, end_time, startPerIter, endPerIter;

  afloat * phi_n_avg = NULL;                            //The average value of phi across iterations
  afloat * phi_n_tot = NULL;                            //The sum of phi across iterations
  afloat * phiS2_n_tot = NULL;
#if defined(L2_1)
  pfloat l2;                                            //l2_norm for comparing with analytical soln.
  afloat * anal_soln = NULL;                            //Holds the analytical solution for comparison
#else
  meanDev * phi_nStats = NULL;                          //Holds the mean and std deviations 
  long long samp_cnt = 0;      	                        //total history tally
#endif
  colData all_tallies;                                  //Structure to return the tallies of phi_n and phi_n2 for every iteration
  
  if( rank == 0 )
    {
      //Memory allocation for all the arrays
      phi_n_avg = (afloat*) calloc(numCells, sizeof(afloat));
      phi_n_tot = (afloat*) calloc(numCells, sizeof(afloat));
      phiS2_n_tot = (afloat *) calloc(numCells, sizeof(afloat)); 		
#if defined(L2_1) 
      anal_soln = (afloat *) calloc(numCells, sizeof (afloat)); 			
      init_L2_norm_1(anal_soln);
#else
      phi_nStats = (meanDev*) malloc(sizeof(meanDev)*numCells);
#endif
    }

  //Plot and print initial condition as well as simulation parameters
  if (rank == 0 && !opt.silent) sim_param_print(data);
  
  int flag_sim = 0;                                     //flag set to 1 when convergence has reached

  dsfmt_t * dsfmt = (dsfmt_t*)malloc(sizeof(dsfmt_t));  // Structure for random number generator
  // MT RNG. Initialize random numbe generator
  dsfmt_init_gen_rand(dsfmt, (int)time(NULL));
    
  //Hold the tallies per thread. For sequential , it is alliased to all_tallies
  colData tallies;
  //Allocate phi_n and phi_n2 to be on different cache lines to avaoid false sharing
  tallies = allocate_tallies(numCells);

#ifdef DO_MPI
  //For MPI version, allocate the global tally on process 0
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
  //For sequential , alias the global tally to tally on rank 0 ( the only rank that exists )
  all_tallies = tallies;
#endif

  double running_col = 0.0;                             //Sum of collisions across all particles and processes   
  double avg_col = 0;                                   //the average_number of collisions per particle across all processes

  //Calculation starts here
  gettimeofday(&start_time, NULL);
	
  // while not converged and total number of iterations < opt.numiters
  while (flag_sim == 0 && iter < opt.numIters) {

    /* //start time for iteration */
    /* gettimeofday(&startPerIter, NULL); */

    iter += 1;
    if(!opt.silent && rank==0) printf("This is the %dth iteration\n", iter);

#ifdef DO_PAPI    
    PAPI_library_init(PAPI_VER_CURRENT);
    int EventSet = PAPI_NULL;
    long long start[4],stop[4];
    PAPI_create_eventset(&EventSet);
    PAPI_add_event(EventSet,PAPI_TOT_CYC);
    PAPI_add_event(EventSet,PAPI_TOT_INS);
    PAPI_add_event(EventSet,PAPI_L2_DCM); 
    //    PAPI_add_event(EventSet,PAPI_L1_DCM); 
    PAPI_start(EventSet);
    PAPI_read(EventSet,start);
#endif
    //calls collision and tally to stream and tally statistics for all particles
    *(tallies.tot_col) = collision_and_tally(data, &tallies,rank,nprocs,dsfmt);
#ifdef DO_PAPI
    PAPI_read(EventSet,stop);
    printf("%lld %lld %lld %lld\n",stop[0] - start[0],stop[1] - start[1],stop[2] - start[2],stop[3] - start[3]);
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(EventSet);
#endif

#ifdef DO_MPI
    //Reduce phi_n and phin_n2 across all MPI processes
    MPI_Reduce(tallies.full_buffer,all_tallies.full_buffer,tallies.buffer_size,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
#endif

    //Averaging is done only on process 0    
    if( rank == 0 )
      {
	running_col += *(all_tallies.tot_col);
	avg_col = running_col / iter / NP;

	//Normalize the phi_n and phi_n2 values
	for( i = 0 ; i < numCells ; i++ )
	  {
	    all_tallies.phi_n[i] /= (afloat)data.dx* data.dy*NP;
	    all_tallies.phi_n[i] *= data.lx * data.ly;
	  }
	
	/***************************************************
	  Calculates the averages 
	  **************************************************\/ */
	iter_avg += 1;
	  
	//accumulate phase -- add new data to the running averages
	for (i = 0; i < numCells; i++) {
	  phi_n_tot[i] += all_tallies.phi_n[i];
	  phiS2_n_tot[i] += all_tallies.phi_n2[i];
	}

	// for each cell, calculate the average for phi_n
	for (i = 0; i < numCells; i++) {
	  phi_n_avg[i] = phi_n_tot[i] / iter_avg;
	}
	  
	//prints out the necessary data
	if( rank == 0){	    
	  printf("Running Average number of collisions = %lf\n",avg_col);
	}

	//check for convergence
#if !defined(L2_1) 
	//Calculate mean and stddev
	samp_cnt = NP_tot * ((long long) iter_avg);
	mean_std_calc(phi_n_tot, phiS2_n_tot, samp_cnt, opt.NP, numCells, opt.dx, phi_nStats);
	if (maxFAS(&phi_nStats[0].stdDev, numCells, 2) <= tol_std) {
	  flag_sim = 1;
	}
	  
	if( !opt.silent ){
	  printf("The maximum standard deviation of flux at node is, max (sig_phi) = %f\n", maxFAS(&phi_nStats[0].stdDev, numCells, 2));
	}
#else
	//Use L2 Norm for comparison with analytical solution
	l2 = l2_norm_cmp(phi_n_avg, anal_soln, numCells, dx);
	flag_sim = l2 <= tol_std;
	if(rank == 0){
	  gettimeofday(&end_time, NULL);
	  printf("L2: %f, Sofar: %ldu_sec\n", l2, (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));
	}
#endif	  
      }	
#ifdef DO_MPI
    //Tell all otehr process if convergence is reached
    MPI_Bcast(&flag_sim,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
    
    /* //end time per iteration */
    /* gettimeofday(&endPerIter, NULL); */
    /* printf("ID = %d, Time per Iteration: %ldu_sec, flags sim = %d\n\n", rank,(endPerIter.tv_sec - startPerIter.tv_sec)*1000000 + (endPerIter.tv_usec - startPerIter.tv_usec),flag_sim); */
  }
  
  //PAUL END TIMING
  gettimeofday(&end_time, NULL);
  if( rank == 0 )
    printf("Elapsed Time: %ldu_sec\n", (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));

  //Print out the final value of phi_n_avg
  if(rank == 0 && !opt.silent){
    FILE* outfile = fopen("phi_values.dat","w");
    for (i = 0; i < numCellsy; i++) {
      for( j = 0 ; j < numCellsx ; j++ )
	fprintf(outfile," %lf %lf %lf \n", j * data.dx + data.dx / 2.0, i * data.dy + data.dy / 2.0, phi_n_avg[i*numCellsx + j]);
    }
    fclose(outfile);
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
      free(phi_n_avg);
      free(phiS2_n_tot);
#if defined(L2_1)
      free(anal_soln);
#else
      free(phi_nStats);
#endif
    }

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
  pfloat ly = data.ly;
  pfloat dy = data.dy;
  int nx = data.nx;
  int ny = data.ny;
  int nc = data.nc;
  long long NP = data.NP;
  pfloat sig_t = data.sig_t;
  pfloat sig_s = data.sig_s;

  printf("***********************************************************\n");
  printf("******THE SIMULATION PARAMETERS ARE PRINTED OUT BELOW******\n");
  printf("The system length is, lx = %lf, ly = %lf\n", lx,ly);
  printf("The cell width is, dx = %lf, dy = %lf\n", dx, dy);
  printf("The number of cell is, nx = %d , ny = %d, nc = %d\n", nx,ny,nc);
  printf("The reference number of particle is, NP = %lld\n", NP);
  printf("The total cross section is, sig_t = %f\n", sig_t);
  printf("The scattering cross section is, sig_s = %f\n", sig_s);
  printf("Floating point data representation is, %lu byte\n", sizeof (pfloat));
  printf("Number of particles = %lld, NPc = %lld\n",data.NP,data.NPc);
  printf("***********************************************************\n");
}


/**********************************************************************************************************
 * collision_and_tally
 * Input :
 * data : describing the physical domain
 * tallies : to store the values of phi_n and phi_n2 
 * rank : Rank of the current process
 * nprocs : total number of processes used
 * 
 * Output
 * colData * tallies : Updated with tallies for this iteration
 *
 * TODO modularize and more optimized parallel reduction and/or alternate way to get data
 *********************************************************************************************************/
double collision_and_tally(struct data data, colData * tallies, int rank, int nprocs, dsfmt_t * dsfmt) {
  //initialize simulation parameters
  int nc = data.nc;
  long long NP = data.NP;

  //initialize tallies and corresponding variables
  memset(tallies->full_buffer,0,tallies->buffer_size);

  struct timeval startBoth, endBoth; 

  //total number of collisions for this iterations
  double tot_num_col = 0;
  
  if (NP != 0) { 
    //start time for montecarlo
    /* gettimeofday(&startBoth, NULL); */

    int lo_cell = nc /nprocs;
    int hi_cell = lo_cell + 1;
    int switch_rank  = hi_cell * nprocs - nc ;
    int startcell,endcell;
    if( rank < switch_rank )
      {
    	startcell = rank * lo_cell ;
    	endcell = startcell + lo_cell ;
      }
    else
      {
    	startcell = switch_rank * lo_cell + ( rank - switch_rank ) * hi_cell;
    	endcell = startcell + hi_cell;
      }
    
    //Stream the particles and tally
    tot_num_col = gen_particles_and_tally(data, tallies,startcell,endcell,dsfmt);

    //end time for montecarlo
    /* gettimeofday(&endBoth, NULL); */
    /* printf("Time for Monte Carlo per Iteration: %ld u_sec\n", (endBoth.tv_sec - startBoth.tv_sec)*1000000 + (endBoth.tv_usec - startBoth.tv_usec)); */
  }

  return tot_num_col;
}
	

/* merged particle generation and tally */
double gen_particles_and_tally(struct data data, colData *tallies, int start_iter_cell, int end_iter_cell, dsfmt_t* dsfmt){
  
  //system dimensions
  const pfloat lx = data.lx;
  const pfloat ly = data.ly;
  const pfloat eps = data.eps;                                                          //Cutoff for sintheta . if sintheta < eps , sintheta set to eps / 2
  const pfloat sig_t_recip = data.sig_t_recip;                                          //inverse of total cross section
  const pfloat cellsizex = data.dx;                                                     //Cellsize along x
  const pfloat cellsizey = data.dy;                                                     //Cellsize along y
  const int numcellsx = data.nx , numcellsy = data.ny;                                  //Number of cells along x and y
  const long long NPc = data.NPc;                                                       //Number of particles to simulate per cell
  const pfloat weight = data.q0_avg;							//Weight of each particle
  const pfloat weightS2 = weight * weight;                                              //Square of the weight
  const pfloat cellsizex_recip = 1.0 / cellsizex;
  const pfloat cellsizey_recip = 1.0 / cellsizey;
  

  //Temporray variables
  int ix,iy,i;
  long long j;
  int collided;
  pfloat ft, mu, sintheta, phi, cosphi, sinphi, tanphi, tanphi_recip, efsintheta, efsintheta_recip, startx, do_collision, starty;
  pfloat cosphi_recip,sinphi_recip;
  pfloat dist, distS2;
  pfloat efsintheta_recipS2;
  int  targetcell, targetcellx, targetcelly;
  int stepX,stepY,signX,signY;
  pfloat weight_efsintheta_recip, weightS2_efsintheta_recipS2;
  pfloat tdeltaX,tdeltaY,tmaxX,tmaxY,tmax;
  //int bdycelly,bdycellx,bdyreflectx,bdyreflecty,swbdycellx,swbdyreflectx,swbdycelly,swbdyreflecty;
  int endcellx, endcelly,endcell;
  int bdycellx,bdycelly;
  pfloat distx0,distxN,disty0,distyN,distxbdy,distybdy;//,isx0,isy0;
  /* int reflectx,reflecty,reflectxbdy,reflectybdy; */
  /* pfloat currft; */

#ifndef NDEBUG
  pfloat  sintheta_avg = 0.0;
  pfloat  fz_avg = 0.0, fx_avg = 0.0, fy_avg = 0.0;
  pfloat phi_avg = 0.0;
  int counter = 0;
  printf("start_iter_cell = %d, last_iter_cell = %d, NPc = %d\n",start_iter_cell,end_iter_cell,NPc);
#endif
  double col_tot = 0.0;                                                                 //Total number of collisions for this iterations
  
  /* create and tally all particles */

  for( i = start_iter_cell ; i < end_iter_cell ; i++ ){
    ix = i % numcellsx;                                                                 //x co-ordinate of starting cell
    iy = i / numcellsx;                                                                 //y co-ordinate of starting cell 
    for (j = 0; j < NPc; j++) {
      //The initial position of the particle
      startx = (ix + unirandMT_0_1(dsfmt)) * cellsizex;                                //x position of particle within the cell
      starty = (iy + unirandMT_0_1(dsfmt)) * cellsizey;                                //y position of particle within the cell
      collided = 0;
      
      //The x and y indices of the first cell to which contribution is added
      targetcellx = ix;
      targetcelly = iy;

      double num_col = 0.0;
      do {
	mu = unirandMT_0_1(dsfmt);                                                      //value of mu = cos(theta), theta = polar angle
	sintheta = sqrt( 1.0 - mu * mu );                                                //value of sintheta
	phi = unirandMT_0_1(dsfmt) * 2.0 * PI;                                          //phi = azimuthal angle
	ft = (-log(unirandMT_0_1(dsfmt)) * sig_t_recip)* sintheta;                      //Distance travelled in the 2D plane = -ln(R) * sintheta / sig_t
	//Variable to check if it went out of system ( 1 if it didnt, 0 if it did )
	collided = 1;
	
	cosphi = cos(phi);                                                            
	cosphi_recip = 1.0/ cosphi;
	sinphi = sin(phi);
	sinphi_recip = 1.0 / sinphi;
	tanphi = sinphi / cosphi;                                                        // tan(phi)
	tanphi_recip = cosphi / sinphi;                                                  // cot(phi)
	  
#ifndef NDEBUG
	fz_avg += ft * sqrt( 1.0 - sintheta * sintheta) / sintheta;
	//The final position of the particle
#endif
	pfloat endx = startx + ft * cosphi;
	pfloat endy = starty + ft * sinphi;
	
	distx0 = -startx * cosphi_recip;                                                 // distance to travel before hitting boundary x = 0
	distxN = ( lx - startx ) * cosphi_recip;                                         // distance to travel before hitting boundary x = lx
	disty0 = -starty * sinphi_recip;                                                 // distance to travel before hitting boundary y = 0
	distyN = ( ly - starty ) * sinphi_recip;                                         // distance to travel before hitting boundary y = ly

	//Particle is moving left
	if( distx0 > 0.0 )
	  {
	    distxbdy = distx0;
	    bdycellx = 0;
	    signX = -1;
	    stepX = 0;
	  }
	//Particle is moving rigth
	else
	  {
	    distxbdy = distxN;
	    bdycellx = numcellsx - 1;
	    signX = 1;
	    stepX = 1;
	  }

	//Particle is moving down
	if( disty0 > 0.0 )
	  {
	    distybdy = disty0;
	    bdycelly = 0;
	    signY = -1;
	    stepY = 0;
	  }
	//Particle is moving up
	else
	  {
	    distybdy = distyN;
	    bdycelly = numcellsy - 1;
	    signY = 1;
	    stepY = 1;
	  }	
	  
	endcellx = (int)(endx * cellsizex_recip);
	endcelly = (int)(endy * cellsizey_recip);
	//Will hit an x bdy first or a y bdy
	if( distxbdy > distybdy )
	  {
	    //Will it hit the boundary at all
	    if( ft > distybdy )
	      {
		ft = distybdy - 1e-9 ;
		endcelly = bdycelly;
		endcellx = (int)( (startx + ft * cosphi) * cellsizex_recip);
		collided = 0;
	      }
	  }
	else
	  //Will it hit the boundary at all
	  if( ft > distxbdy )
	    {
	      ft = distxbdy - 1e-9 ;
	      endcellx = bdycellx;
	      endcelly = (int)( ( starty + ft * sinphi) * cellsizey_recip);
	      collided = 0;
	    }
	endcell = endcelly * numcellsx + endcellx;
	
#ifndef NDEBUG	  
	sintheta_avg += sintheta;
	phi_avg += phi;
	counter++;
#endif

	// precompute per-loop constants, to be used below
	efsintheta = (sintheta > eps) ? sintheta : eps/2;			       /* with epsilon boundaries */
	efsintheta_recip = 1.0 / efsintheta;					       /* reciprocal */
	efsintheta_recipS2 = efsintheta_recip*efsintheta_recip;		               /* for STDEV */
	/* pre-compute repeatedly used measurements*/    
	weight_efsintheta_recip = weight*efsintheta_recip;
	weightS2_efsintheta_recipS2 = weightS2*efsintheta_recipS2; 

	tmaxX =  ( ( targetcellx + stepX ) * cellsizex - startx ) * cosphi_recip;      //Distance travelled before the particle hits a x=c boundary
	tmaxY =  ( ( targetcelly + stepY ) * cellsizey - starty ) * sinphi_recip;      //Distance travelled before the particle hits a y=c boundary
	tdeltaX = signX * cellsizex * cosphi_recip;                                    //Incremenet to be added to tmaxX if a particle hits the x=c bdy
	tdeltaY = signY * cellsizey * sinphi_recip;                                    //Incremenet to be added to tmaxX if a particle hits the y=c bdy
	tmax = 0.0;                                                                    //Total distance travelled by the particle

	while( ( targetcell = targetcelly * numcellsx + targetcellx ) != endcell )     //Check if the particle has reached the last cell for this flight
	  {
	    //Does particle hit x=c bdy first or y = c bdy first
	    if( tmaxY < tmaxX ) 
	      {
		dist = tmaxY - tmax;
		startx += dist * cosphi;                                                //Update the x posn of the particle
		starty += dist * sinphi;                                                //Update the y-posn of the particle
		tmax += dist;                                                           //Update the total distance travelled by the particle
		tmaxY += tdeltaY;                  
		targetcelly += signY;
	      }
	    else
	      {
		dist = tmaxX - tmax;
		startx += dist * cosphi;                                                 //Update the x posn of the particle
		starty += dist * sinphi;                                                 //Update the y-posn of the particle
		tmax += dist;                                                            //Update the total distance travelled by the particle
		tmaxX = tmaxX + tdeltaX;
		targetcellx += signX;
	      }
	    distS2 = dist * dist;
#ifndef NDEBUG 
	    fx_avg += signX * dist * cosphi;
	    fy_avg += signY * dist * sinphi;
#endif
	    tallies->phi_n[targetcell] += weight_efsintheta_recip * dist;            //Add contribution to the current cell dist * weight / sintheta
	    tallies->phi_n2[targetcell] += weightS2_efsintheta_recipS2 * distS2;     //For standard deviation calculation
	  }
	//Tallying through the last cell travlled
	dist = ft - tmax;                                                       //Distance travelled in the final cell
	distS2 = dist * dist;
	startx += dist * cosphi;                                                //Update the x posn of the particle
	starty += dist * sinphi;                                                //Update the y-posn of the particle

#ifndef NDEBUG 
	fx_avg += signX * dist * cosphi;
	fy_avg += signY * dist * sinphi;
#endif
	tallies->phi_n[targetcell] += weight_efsintheta_recip * dist;            //Add contribution to the current cell dist * weight / sintheta
	tallies->phi_n2[targetcell] += weightS2_efsintheta_recipS2 * distS2;     //For standard deviation calculation
	
	num_col += collided;                                                         //Total number of collisions experienced by the particle
	//Continue streaming particle if it reflected of a surface or if it stays within the system and is scattered
      } while(collided && ( ( do_collision = unirandMT_0_1(dsfmt) ) < (data.sig_s * data.sig_t_recip) ) );

      //Add the number of collisions of this particle, to total collisions across all particles
      col_tot += ( do_collision < data.sig_s * data.sig_t_recip ? num_col : num_col - 1.0 ) ;
    }
  }
  //printf("Time for SubTally accusinthetalation per Iteration: %lld u_sec\n", totInner);
#ifndef NDEBUG
  printf("Counter = %d, average sintheta = %lf, average fx = %lf, average fy = %lf, average fz = %lf, average phi = %lf\n",counter,sintheta_avg/counter,fx_avg/counter,fy_avg/counter,fz_avg/counter,phi_avg / counter);   
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
