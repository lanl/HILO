/* 
 * Quasi Diffusion Accelerated Monte Carlo
 *
 * This OpenMP implementation utilizes GPU for particle generation, there is a straightforward GPU implementation
 * and a asynchronous GPU implementation where we attempt to overlap communication and computation
 *
 * Authors:   Han Dong
 *            Paul Sathre
 *            Mahesh Ravishankar
 *            Mike Sullivan
 *            Will Taitano
 *            Jeff Willert
 *
 * Last Update: 8/17/2011 4:22 PM
 */

#include "emmintrin.h"
#include "qdamc.h"
#include "qdamc_utils.h"
#include "../ecodes.h"
#include "omp.h"
#include <CL/cl.h>
#include "ranluxcl.h"

#define CACHE_LINE_SIZE 64
#define VECTOR_WIDTH 2
#ifndef SSE_SWITCH
#define SSE_SWITCH 1000
#endif

/*********************************************************
 * OpenCL error codes
 *
 *********************************************************/
char *print_cl_errstring(cl_int err) 
{
    switch (err)
    {
        case CL_SUCCESS: return strdup("Success!");
        case CL_DEVICE_NOT_FOUND: return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE: return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE: return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES: return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY: return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE: return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP: return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH: return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE: return strdup("Program build failure");
        case CL_MAP_FAILURE: return strdup("Map failure");
        case CL_INVALID_VALUE: return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE: return strdup("Invalid device type");
        case CL_INVALID_PLATFORM: return strdup("Invalid platform");
        case CL_INVALID_DEVICE: return strdup("Invalid device");
        case CL_INVALID_CONTEXT: return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES: return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE: return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR: return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT: return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE: return strdup("Invalid image size");
        case CL_INVALID_SAMPLER: return strdup("Invalid sampler");
        case CL_INVALID_BINARY: return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS: return strdup("Invalid build options");
        case CL_INVALID_PROGRAM: return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE: return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME: return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION: return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL: return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX: return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE: return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE: return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS: return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION: return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE: return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE: return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET: return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST: return strdup("Invalid event wait list");
        case CL_INVALID_EVENT: return strdup("Invalid event");
        case CL_INVALID_OPERATION: return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT: return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE: return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL: return strdup("Invalid mip-map level");
        default: return strdup("Unknown");
    }
}

/*
 * Main Function
 */
int run(int rank) 
{
	/* data variables */
  	int NPc, NP_tot, lbc_flag, i, JP_l, iter, iter_avg, max_threads, a, b;
	struct simData data;
 	struct gpuData gdata;  
  	int * filter_flag = NULL;
  	int * startcells, * endcells;
  
  	//time keeping structures
  	struct timeval start_time, end_time, startPerIter, endPerIter, mcStart, mcEnd, reduceStart, reduceEnd, colTallyStart, colTallyEnd, avgStart, avgEnd, otherStart, otherEnd;
  	struct timeval waitStart, waitEnd, asyncStart, asyncEnd;
  
  	//GPU variables and data structures
  	struct oclData oclEnv;
  	size_t pStart, constOffset, RANDGEN, worksize;
  	cl_int cellOffset, offset, piter;
  	cl_int nskip, lux = 4;
  	cl_event firstRead[6];
  	cl_event kernelEvent;
  
  	//GPU data structures
  	cl_float * recvMu;
  	cl_float * recvStart;
  	cl_float * recvEnd;
  	cl_float * mu0;
  	cl_float * x0;
  	cl_float * xf;
  
	int flag_sim = 0;
  	unsigned long samp_cnt = 0;           

  	//stores user input
  	NP = opt.NP;
  	sysLeng = opt.sysLen;
  	numCells = opt.numCells;
  	tol = opt.tol;
  	tol_std = opt.tol_std;
  	runningWidth = opt.runWidth;
	
	//scales number of particles to be divisible by 64 and numCells
  	piter = 64*numCells;	
  	if((NP % piter) != 0)
  	{
  		NP += piter - (NP % piter);
  	} 

  	//Initialization of vital simulation parameters
  	pfloat dx = sysLeng / numCells;                             	//cell width
  	pfloat eps = 0.1f;                                          	//the threshold angle for tallying
  	pfloat sig_t = opt.sig_t;
  	pfloat sig_s = opt.sig_s;
  	pfloat D = 1.0 / (3 * sig_t );                     				//diffusion coefficient
  	JP_l = 1;      

  	//The source term uniform for now	
  	pfloat * q0 = NULL; 
  	if( rank == 0 )
	{
    	q0 = (pfloat *) malloc(sizeof (pfloat) * numCells);	
    	for (i = 0; i < numCells; i++)
		{
      		q0[i] = 1.0f;
    	}
  	}
  	pfloat q0_lo = 1.0f;

  	//Form data structure for simulation parameters
  	data.NP = NP;													//Number of Particles
  	data.lx = sysLeng;												//Length of System
  	data.nx = numCells;												//Number of Cells
  	data.dx = dx;													//Cell Width
  	data.dx2 = dx*dx;												// Cell Width SQUARED
  	data.dx_recip = 1.0/dx;											// Pre-computed reciprocal
  	data.sig_t = sig_t;												//Total Collision Cross-Section
  	data.sig_t_recip = 1.0/sig_t;									// Pre-computed reciprocal
  	data.sig_s = sig_s;												//Scattering Collision Cross-Section
  	data.q0 = q0;
  	data.Jp_l = JP_l;												//Partial Current on Left Face
  	data.eps = eps;													//Threshold Angle for Tallying
  	data.q0_lo = q0_lo;												//Source Term
  	data.D = D;														//Diffusion Coefficient
  	NPc = rint((pfloat)NP / numCells);  							// number of particles in cell
  	NP_tot = NP; 													//total number of particles for the QDAMC
  	data.NP_tot = NP_tot;											// Total Number of Particles
  	data.NPc = NPc;													// Number of Particles per Cell
  	lbc_flag = 2;
  	data.lbc_flag = lbc_flag;
  	data.NPstep = NPc;
  	data.steps = numCells;
  
  	//data structure for gpu
  	gdata.dx = data.dx;
  	gdata.sig_t = data.sig_t;
  
  	piter = 64;
	//initialize kernel
  	oclEnv = oclInit(&data, &gdata);
	//worksize for GPU kernel
  	worksize = data.NP / piter;

	//calculates how to distribute cells to each OMP thread
  	max_threads = omp_get_max_threads();
  	startcells = (int *) calloc (max_threads, sizeof(int));
  	endcells = (int *) calloc (max_threads, sizeof(int));
  	for(i=0;i<max_threads;i++)
  	{
  		calcStartEnd(&a, &b, i, data.nx, max_threads);
  		startcells[i] = a;
  		endcells[i] = b;
  	}
  
#ifdef DO_ASYNC
  	//allocate GPU recv buffers
  	recvMu = calloc (data.NP, sizeof(float));
  	recvStart = calloc (data.NP, sizeof(float));
  	recvEnd = calloc (data.NP, sizeof(float));
  
	//double buffering
  	mu0 = calloc (data.NP*2, sizeof(float));
  	x0 = calloc (data.NP*2, sizeof(float));
  	xf = calloc (data.NP*2, sizeof(float));
#else
  	//allocate GPU recv buffers
  	mu0 = calloc (data.NP, sizeof(float));
  	x0 = calloc (data.NP, sizeof(float));
  	xf = calloc (data.NP, sizeof(float));
#endif

  	//Initialize the average values
  	iter = 0;                   								//tally for iteration for convergence
  	iter_avg = 1;               								// iteration for averaging
  	afloat phi_left_tot = 0.0;									//tally for average phi
  	afloat phi_right_tot = 0.0;									//tally for average phi
  	afloat J_left_tot = 0.0;  									//tally for average J
  	afloat J_right_tot = 0.0; 									//tally for average J
  	afloat E_left_tot = 0.0;  									//tally for average E
  	afloat E_right_tot = 0.0; 									//tally for average E

  	// Tally for averaging the parameters above
  	pfloat phi_left_avg = 0.0;
  	pfloat phi_right_avg = 0.0;
  	pfloat J_left_avg = 0.0;
  	pfloat J_right_avg = 0.0;
  	pfloat E_left_avg = 0.0;
  	pfloat E_right_avg = 0.0;

  	afloat * phi_n_tot = NULL;
  	afloat * phi_lo_tot = NULL;
  	afloat * E_n_tot = NULL;
  	pfloat * phi_n_avg = NULL;
  	pfloat * E_n_avg = NULL;
  	pfloat * phi_lo_avg = NULL;
  	pfloat * phi_lo_avg_old = NULL;
  	pfloat * E_ho_n = NULL;
  	afloat * phi_ho_s1_n = NULL;
  	afloat * phi_ho_s2_n = NULL;
  	void * f_avg_send = NULL;
  	pfloat * phi_lo = NULL;
  	triMat A_lo;
  	pfloat * b_lo = NULL;
  	meanDev * phi_nStats = NULL;
  	pfloat * Qbar = NULL;
  	pfloat * stdev_vec = NULL;
  	afloat * anal_soln = NULL;
    
  	if(rank == 0)
	{
    	// each cell stores information about phi and E in single precision
    	phi_n_tot = (afloat *) calloc(numCells, sizeof (afloat));
    	phi_lo_tot = (afloat *) calloc(numCells, sizeof (afloat)); 		//tally for average lower order phi
    	E_n_tot = (afloat *) calloc(numCells, sizeof (afloat));
    	
		// each cell stores the averages for E_n, phi_n and phi_lo	
    	phi_n_avg = (pfloat *) calloc(numCells, sizeof (pfloat));
    	E_n_avg = (pfloat *) calloc(numCells, sizeof (pfloat));
    	phi_lo_avg = (pfloat *) calloc(numCells, sizeof (pfloat));
    	phi_lo_avg_old = (pfloat *) calloc(numCells, sizeof (pfloat));
    
		// 1000 element standard deviation vector
    	stdev_vec = (pfloat *) calloc(1000, sizeof (pfloat));

    	// Initialize lower order scalar flux using diffusion approximation
    	E_ho_n = (pfloat *) malloc(sizeof (pfloat) * numCells);
    	for (i = 0; i < numCells; i++)
		{
      		E_ho_n[i] = 1.0f / 3.0f;
    	}
    	phi_ho_s1_n = (afloat *) calloc(numCells, sizeof (afloat)); 		// Mean tally for flux at node
    	phi_ho_s2_n = (afloat *) calloc(numCells, sizeof (afloat)); 		// Mean of square tally for flux at node
    
    	filter_flag = (int*) malloc(data.nx * sizeof(int));
    	int data_size = ( sizeof(afloat) > sizeof(pfloat) ? sizeof(afloat) : sizeof(pfloat) );
    	f_avg_send = malloc(data_size * (data.nx+1));
    	phi_lo = (pfloat*) malloc(data.nx*sizeof(pfloat));
    	A_lo.a = (pfloat*) malloc(sizeof(pfloat)*(data.nx-1));
    	A_lo.b = (pfloat*) malloc(sizeof(pfloat)*data.nx);
    	A_lo.c = (pfloat*) malloc(sizeof(pfloat)*(data.nx-1));
    	b_lo = (pfloat*) malloc(sizeof(pfloat)*data.nx);

    	/*From mean_std_calc*/
    	phi_nStats = (meanDev*) malloc(sizeof(meanDev)*opt.numCells);
    
#ifdef L2_1
    	anal_soln = (afloat *) calloc(numCells, sizeof (afloat)); 			// Reference solution
		init_L2_norm_1(anal_soln);
#else
	#ifdef L2_2
    	anal_soln = (afloat *) calloc(numCells, sizeof (afloat)); 			// Reference solution
		init_L2_norm_2(anal_soln);
	#endif
#endif
  }

  /************************************************
   * Calculation of Initial Condition
   * TODO account for isotropic and beam source
   *************************************************/
	Qbar = malloc(sizeof (pfloat) * numCells);
  	colData * thread_tallies;
  	thread_tallies = (colData*) malloc(max_threads*sizeof(colData));
 
  	//Allocating data structures to be used by each thread
  	for( i = 0 ; i < max_threads ; i++ )
	{
    	//Allocate all data that is written by the thread to cache line size
    	posix_memalign( (void**)&(thread_tallies[i].phi_n), CACHE_LINE_SIZE, numCells*sizeof(afloat));
    	posix_memalign( (void**)&(thread_tallies[i].phi_n2), CACHE_LINE_SIZE, numCells*sizeof(afloat));
    	posix_memalign( (void**)&(thread_tallies[i].E_n), CACHE_LINE_SIZE, numCells*sizeof(afloat));
  	}
  
  	colData * all_tallies;
  	all_tallies = &(thread_tallies[0]);
  
  	if(omp_get_thread_num() == 0)
  	{
  	   /*************************************************************************
	   	* Initializes randlux variables and generates a random number in kernel
	   	* TODO better naming convention
	   	*************************************************************************/
	   	//calls initialization funtion
		cl_float4 * ranfloats = ranluxcl_initialization(lux, 0, worksize, worksize, &nskip, &RANDGEN);
		
	   	//enqueues RANDGEN in buffer to be used in simulation for random number generation
	  	oclEnv.ranBuf = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, RANDGEN, NULL, NULL);   
	   	clEnqueueWriteBuffer(oclEnv.queue, oclEnv.ranBuf, CL_TRUE, 0, RANDGEN, ranfloats, 0, NULL, NULL);
	   	clSetKernelArg(oclEnv.warmup, 0, sizeof (cl_mem), (void*) &oclEnv.ranBuf);
	   	clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.warmup, 1, NULL, &worksize, NULL, 0, NULL, NULL);
	   	clFinish(oclEnv.queue);
		    
	   	//put simData into oclEnv.clData onto the device
	   	clEnqueueWriteBuffer(oclEnv.queue, oclEnv.clData, CL_FALSE, 0, sizeof (struct gpuData), &gdata, 0, NULL, NULL);
			
	   	//set arguments for particle generation
	   	clSetKernelArg(oclEnv.pGenerate2, 0, sizeof (cl_mem), (void*) &oclEnv.clData);
	   	clSetKernelArg(oclEnv.pGenerate2, 1, sizeof (cl_mem), (void*) &oclEnv.clMu0);
	   	clSetKernelArg(oclEnv.pGenerate2, 2, sizeof (cl_mem), (void*) &oclEnv.clX0);
	   	clSetKernelArg(oclEnv.pGenerate2, 3, sizeof (cl_mem), (void*) &oclEnv.clXf);
	   	clSetKernelArg(oclEnv.pGenerate2, 4, sizeof (cl_mem), (void*) &oclEnv.ranBuf);	
   	}

  	//Calculation starts here
  	gettimeofday(&start_time, NULL);

  	if( rank == 0)
	{
    	//Plot and print initial condition as well as simulation parameters
    	if (!opt.silent) sim_param_print(data);

    	//calls low order solver for initial condition
    	lo_solver(data, 0.0, 0.0, 0.0, 0.0, 1.0f / 3.0f, 1.0f / 3.0f, E_ho_n, 1,phi_lo,A_lo,b_lo);
    	free(E_ho_n);
  	}
  
#pragma omp parallel default(shared) shared(oclEnv,recvMu,recvStart,recvEnd,firstRead,mu0,x0,xf) private(iter,i)
{
	iter = 0; //tally for iteration for convergence
	int nthreads = omp_get_num_threads();
    constOffset = sizeof(cl_float) * data.NP;
    int nstages = 0;
    i = 1;
    
	while( i < nthreads )
    {
		nstages++;
		i *=2;
    }
    int thread_id = omp_get_thread_num();

//asynchronous call, preload kernel to fill in 1st buffer and 2nd buffer
#ifdef DO_ASYNC    
   	if(thread_id == 0)
   	{
   		pStart = 0;
		worksize = data.NP / piter;
		cellOffset = data.NPstep;
		offset = 0;

		//first buffer
		clSetKernelArg(oclEnv.pGenerate2, 5, sizeof (cl_int), &cellOffset);
		clSetKernelArg(oclEnv.pGenerate2, 6, sizeof (cl_int), &offset);
		clSetKernelArg(oclEnv.pGenerate2, 7, sizeof (cl_int), &piter);
			
		//launch kernel with worksize NPstep
		clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.pGenerate2, 1, NULL, &worksize, NULL, 0, NULL, NULL);
		clFinish(oclEnv.queue);
		
		//read data back to buffers
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clMu0, CL_FALSE, 0, constOffset, &mu0[pStart], 0, NULL, &firstRead[0]);
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clX0, CL_FALSE, 0, constOffset, &x0[pStart], 0, NULL, &firstRead[1]);
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clXf, CL_FALSE, 0, constOffset, &xf[pStart], 0, NULL, &firstRead[2]);	
				
		pStart = data.NP;
		offset = data.NP;
				
		//second buffer
		clSetKernelArg(oclEnv.pGenerate2, 5, sizeof (cl_int), &cellOffset);
		clSetKernelArg(oclEnv.pGenerate2, 6, sizeof (cl_int), &offset);
		clSetKernelArg(oclEnv.pGenerate2, 7, sizeof (cl_int), &piter);
		
		//launch kernel with worksize NPstep
		clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.pGenerate2, 1, NULL, &worksize, NULL, 0, NULL, NULL);
		clFinish(oclEnv.queue);
				
		//read data back to buffers
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clMu0, CL_FALSE, constOffset, constOffset, &mu0[pStart], 0, NULL, &firstRead[3]);
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clX0, CL_FALSE, constOffset, constOffset, &x0[pStart], 0, NULL, &firstRead[4]);
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clXf, CL_FALSE, constOffset, constOffset, &xf[pStart], 0, NULL, &firstRead[5]);
    }
#endif
    
    /*********************************************************
     * WHILE LOOP
     ********************************************************/
    // while not converged and total number of iterations < 1000
    // TODO as many iterations as needed until convergence or user specific iter value
    while (flag_sim == 0 && iter < opt.numIters)
	{
    	iter += 1;
       
      	//start time for iteration
      	if(thread_id == 0)
      	{
      		gettimeofday(&startPerIter, NULL);
       
#ifdef DO_ASYNC  
      		gettimeofday(&waitStart, NULL);
      	
			//waits to make sure that the asynchronous read is done for 1st or 2nd buffer
      		clWaitForEvents(3, &firstRead[((iter+1)&1) * 3]);
      		clReleaseEvent(firstRead[((iter+1)&1) * 3]);
	  		clReleaseEvent(firstRead[(((iter+1)&1) * 3)+1]);
	  		clReleaseEvent(firstRead[(((iter+1)&1) * 3)+2]);	
	  
			gettimeofday(&waitEnd, NULL);
			printf("Time for WAIT %ldu_sec\n", (waitEnd.tv_sec - waitStart.tv_sec)*1000000 + (waitEnd.tv_usec - waitStart.tv_usec));
#endif 
      	}

      	if(rank == 0 && thread_id == 0 ) 
      	{
      		printf("This is the %dth iteration\n",iter);
      	}
      
      	if( rank == 0 && thread_id == 0 )
      	{
      		gettimeofday(&otherStart, NULL);
			if( iter == 1 )
			{
	  			for (i = 0; i < numCells; i++) 
	  			{
	   				Qbar[i] = fabs(q0[i] + sig_s * phi_lo[i]) * data.lx;	
	  			}
	  		}
			else
			{
	  			for (i = 0; i < numCells; i++)
	  			{
	    			Qbar[i] = fabs(q0[i] + sig_s * phi_lo_avg[i]) * data.lx;
	    		}	
	  		}	
	  
	  		gettimeofday(&otherEnd, NULL);
			printf("Time for OTHER %ldu_sec\n", (otherEnd.tv_sec - otherStart.tv_sec)*1000000 + (otherEnd.tv_usec - otherStart.tv_usec));
      	}

/*************************************
 * ASYNC
 ************************************/
#ifdef DO_ASYNC
		if(thread_id == 0)
		{
			gettimeofday(&colTallyStart, NULL);
		}

#pragma omp barrier	
		//calls collision and tally OPENCL kernel code
   		collision_and_tally(data, Qbar,thread_tallies+ thread_id,thread_id,nthreads,rank,mu0, x0, xf,iter, startcells[thread_id], endcells[thread_id]);
#pragma omp barrier
	      
		if(thread_id == 0)
		{
			gettimeofday(&colTallyEnd, NULL);
			printf("Time for COLLISION AND TALLY %ldu_sec\n", (colTallyEnd.tv_sec - colTallyStart.tv_sec)*1000000 + (colTallyEnd.tv_usec - colTallyStart.tv_usec));
		}

		if(thread_id == 0)
		{
			gettimeofday(&asyncStart, NULL);
		
			pStart = data.NP * ((iter+1)&1);
			worksize = data.NP / piter;
			cellOffset = data.NPstep;
			offset = data.NP * ((iter+1)&1);
			
			clSetKernelArg(oclEnv.pGenerate2, 5, sizeof (cl_int), &cellOffset);
			clSetKernelArg(oclEnv.pGenerate2, 6, sizeof (cl_int), &offset);
			clSetKernelArg(oclEnv.pGenerate2, 7, sizeof (cl_int), &piter);
					
			//launch kernel with worksize NPstep
			clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.pGenerate2, 1, NULL, &worksize, NULL, 0, NULL, &kernelEvent);
			clFinish(oclEnv.queue);
		
			gettimeofday(&asyncEnd, NULL);
			printf("Time for ASYNC %ldu_sec\n", (asyncEnd.tv_sec - asyncStart.tv_sec)*1000000 + (asyncEnd.tv_usec - asyncStart.tv_usec));
		}
/*****************************************
 * GPU
 *****************************************/
#else	
	if(thread_id == 0)
	{
	  	gettimeofday(&mcStart, NULL);
	  
		pStart = 0;
		worksize = data.NP / piter;
		cellOffset = data.NPstep;
		offset = 0;
	
		clSetKernelArg(oclEnv.pGenerate2, 5, sizeof (cl_int), &cellOffset);
		clSetKernelArg(oclEnv.pGenerate2, 6, sizeof (cl_int), &offset);
		clSetKernelArg(oclEnv.pGenerate2, 7, sizeof (cl_int), &piter);
				
		//launch kernel with worksize NPstep
		clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.pGenerate2, 1, NULL, &worksize, NULL, 0, NULL, NULL);
	
		//read data back to buffers
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clMu0, CL_FALSE, 0, constOffset, mu0, 0, NULL, NULL);
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clX0, CL_FALSE, 0, constOffset, x0, 0, NULL, NULL);
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clXf, CL_TRUE, 0, constOffset, xf, 0, NULL, NULL);		
		clFinish(oclEnv.queue);
		
	  	gettimeofday(&mcEnd, NULL);
		printf("Time for MONTE-CARLO %ldu_sec\n", (mcEnd.tv_sec - mcStart.tv_sec)*1000000 + (mcEnd.tv_usec - mcStart.tv_usec));
	}
	
	if(thread_id == 0)
	{
		gettimeofday(&colTallyStart, NULL);
	}
	
#pragma omp barrier
	//calls collision and tally OPENCL kernel code
    collision_and_tally(data, Qbar,thread_tallies+ thread_id,thread_id,nthreads,rank,mu0, x0, xf, iter, startcells[thread_id], endcells[thread_id]);
#pragma omp barrier

	if(thread_id == 0)
	{
		gettimeofday(&colTallyEnd, NULL);
		printf("Time for COLLISION AND TALLY %ldu_sec\n", (colTallyEnd.tv_sec - colTallyStart.tv_sec)*1000000 + (colTallyEnd.tv_usec - colTallyStart.tv_usec));
	}
#endif

/**************************
 * ASYNC
 *************************/	
#ifdef DO_ASYNC
	if(thread_id == 0)
	{
		//read data back to buffers
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clMu0, CL_FALSE, constOffset * ((iter+1)&1), constOffset, &mu0[pStart],0, NULL, &firstRead[(((iter+1)&1) * 3)]);
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clX0, CL_FALSE, constOffset * ((iter+1)&1), constOffset, &x0[pStart], 0, NULL, &firstRead[(((iter+1)&1) * 3)+1]);
		clEnqueueReadBuffer(oclEnv.queue, oclEnv.clXf, CL_FALSE, constOffset * ((iter+1)&1), constOffset, &xf[pStart], 0, NULL, &firstRead[(((iter+1)&1) * 3)+2]);
	}
#endif

/************************************
 * REDUCE
 ************************************/
      if(thread_id == 0)
	  {
		gettimeofday(&reduceStart, NULL);
	  }
      //COMBINE TALLIES HERE - TREE REDUCTION	  
      int factor = 1;
      int k;
      for( i = 0 ; i < nstages ; i++ )
	  {
#pragma omp barrier
	  if( thread_id % (factor*2) == 0 && ( thread_id + factor < nthreads ) )
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
	  factor *= 2;
	}

	if(thread_id == 0)
	{
		gettimeofday(&reduceEnd, NULL);
		printf("Time for tree reduction %ldu_sec\n", (reduceEnd.tv_sec - reduceStart.tv_sec)*1000000 + (reduceEnd.tv_usec - reduceStart.tv_usec));
	}
	
    if(thread_id == 0)
	{
		gettimeofday(&avgStart, NULL);
	}
	  	  
    if( rank == 0 && thread_id == 0 )
    {
	/*Hoisted computation from collision_and_tally */
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
	      phi_ho_s1_n[i] = all_tallies->phi_n[i];
	      phi_ho_s2_n[i] = all_tallies->phi_n2[i];
	    }
	    
	    for (i = 0; i < numCells; i++) {
	      phi_n_tot[i] = all_tallies->phi_n[i];
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
	  samp_cnt = NP_tot * ((long) iter_avg);

	  // for each cell, do single precision adds for phi_ho_s1_n and phi_ho_s2_n
	  for (i = 0; i < numCells; i++) {
	    phi_ho_s1_n[i] += all_tallies->phi_n[i];
	    phi_ho_s2_n[i] += all_tallies->phi_n2[i];
	  }
	  mean_std_calc(phi_ho_s1_n, phi_ho_s2_n, samp_cnt, opt.NP, opt.numCells, opt.dx, phi_nStats);

	  //accumulate phase -- add new data to the running averages
	  for (i = 0; i < numCells; i++) {
	    phi_n_tot[i] += all_tallies->phi_n[i];
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
	
	  //prints out the necessary data
	  if(!opt.silent){
	    printf("NODEAVG\n");
	    for (i = 0; i < numCells; i++) {
	      printf("%d %f %f %f %f\n", i, phi_n_avg[i], E_n_avg[i], phi_lo_avg[i], phi_nStats[i].stdDev);
	    } 
	    printf("The maximum standard deviation of flux at node is, max (sig_phi) = %f\n", maxFAS(&phi_nStats[0].stdDev, numCells, 2));
	  }

      //check for convergence
#ifdef L2_1
	printf("%d\n", rank);
	flag_sim = l2_norm_cmp(phi_n_avg, anal_soln, numCells, dx) <= tol_std;
#else
	#ifdef L2_2
			printf("DOING L2_2\n");
			flag_sim = l2_norm_cmp(phi_n_avg, anal_soln, 100, dx) <= tol_std;
	#else
			if (maxFAS(&phi_nStats[0].stdDev, numCells, 2) <= tol_std) {
			  flag_sim = 1;
			}
	#endif
#endif
	}

	lo_solver(data, phi_left_avg, phi_right_avg, J_left_avg, J_right_avg, E_left_avg, E_right_avg, E_n_avg, 0, phi_lo, A_lo,b_lo);
	  
      }
	if(thread_id == 0)
	{
		gettimeofday(&avgEnd, NULL);
		printf("Time for AVERAGE %ldu_sec\n", (avgEnd.tv_sec - avgStart.tv_sec)*1000000 + (avgEnd.tv_usec - avgStart.tv_usec));
	}
#pragma omp barrier 
      //end time per iteration
      if(thread_id == 0)
      {
      	gettimeofday(&endPerIter, NULL);
      	printf("Time per Iteration: %ldu_sec\n\n", (endPerIter.tv_sec - startPerIter.tv_sec)*1000000 + (endPerIter.tv_usec - startPerIter.tv_usec));
      }
    }
  
    free(thread_tallies[thread_id].phi_n);
    free(thread_tallies[thread_id].phi_n2);
    free(thread_tallies[thread_id].E_n);
  }
  /************************************************
   * Free memory
   *************************************************/
  gettimeofday(&end_time, NULL);
  if( rank == 0 ) printf("Total Elapsed Time: %ldu_sec\n", (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));
  
  free(thread_tallies);
  
  if( rank == 0 ){

    printf("\n\nNODEAVG\n");
    for (i = 0; i < numCells; i++) {
      printf("%d %f %f %f %f\n", i, phi_n_avg[i], E_n_avg[i], phi_lo_avg[i], phi_nStats[i].stdDev);
    }
    
    free(filter_flag);
    free(f_avg_send);
    free(A_lo.a);
    free(A_lo.b);
    free(A_lo.c);
    free(b_lo);
    free(phi_lo);
    free(phi_nStats);
    free(Qbar);
    free(q0);
    free(phi_n_tot);
    free(phi_lo_tot);
    free(E_n_tot);
    free(phi_n_avg);
    free(E_n_avg);
    free(phi_lo_avg);
    free(phi_lo_avg_old);
    free(stdev_vec);
    free(phi_ho_s1_n);
    free(phi_ho_s2_n);
#ifdef L2_1
	free(anal_soln);
#else
	#ifdef L2_2
		free(anal_soln);
	#endif
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
 * TODO better naming convention for data
 **********************************************************************************************/
void sim_param_print(struct simData data) {
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
 * calcStartEnd
 *
 * Function call that determines the start and end cell for each mpi process
 *
 * @param a				
 * @param b
 * @param rank
 * @param nx
 * @param nprocs
 *
 * @return start and end cell
 *
 *****************************************************************************************************/
void calcStartEnd(int *a, int *b, int rank, int nx, int nprocs)
{
  int lower_proc, higher_proc, switch_proc, localstart, localnx;
  	
  lower_proc = nx/nprocs;
  higher_proc = lower_proc + 1;
  switch_proc =  higher_proc * nprocs - nx;
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

  (*a) = localstart;
  (*b) = localstart + localnx;
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
void lo_solver(struct simData data, pfloat phi_ho_left, pfloat phi_ho_right, pfloat J_ho_left, pfloat J_ho_right, pfloat E_ho_left, pfloat E_ho_right, pfloat * E_ho_n, int ic, pfloat* phi_lo,triMat A, pfloat* b) {

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
void collision_and_tally(struct simData data, pfloat * Qbar, colData * tallies, int thread_id, int nthreads, int rank, float * recvMu, float * recvStart, float * recvEnd, int iter, int startcell, int endcell) {
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
  
  if (NP != 0) { //weight per particl3e
    gen_particles_and_tally_ompgpu(data, Qbar, tallies, startcell, endcell, recvMu, recvStart, recvEnd, iter);
  }
}

void gen_particles_and_tally_ompgpu(struct simData data, pfloat * Qbar, colData * tallies, int start_iter_cell, int end_iter_cell, 
float * recvMu, float * recvStart, float * recvEnd, int iter)
{
// particle generation
  pfloat eps = data.eps;
  pfloat fmu, mu, efmu, efmu_recip, start, end, temp;
  pfloat weight;
  int i, j, k;

  // tallies
  pfloat efmu_recipS2;
  pfloat weightS2;
  pfloat absdist;
  int startcell, endcell;
  pfloat begin_rightface, end_leftface;
  // pre-compute useful cell values
  // for inner loop
  pfloat weight_cellsize_efmu_recip, weightS2_cellsizeS2_efmu_recipS2, weight_cellsize_efmu;
 // printf("reading data from %d\n", ((iter+1)&1));
  		for (i = start_iter_cell; i < end_iter_cell; i++) 
		{
			for (j = 0; j < data.NPstep; j++) 
			{
				// all particles are weighted from their starting (not left-most) cell
				weight = Qbar[i];											// note: using a different format from old Qbar
			  	weightS2 = weight * weight;							// for STDEV
#ifdef DO_ASYNC
				//mu = recvMu[i * data.NPstep + j];
				//start = recvStart[i * data.NPstep + j];
				//end = recvEnd[i * data.NPstep + j];
				mu = recvMu[(((iter+1)&1) * data.NP) + (i * data.NPstep + j)];
				start = recvStart[(((iter+1)&1) * data.NP) + (i * data.NPstep + j)];
				end = recvEnd[(((iter+1)&1) * data.NP) + (i * data.NPstep + j)];
#else
				mu = recvMu[i * data.NPstep + j];
				start = recvStart[i * data.NPstep + j];
				end = recvEnd[i * data.NPstep + j];
#endif
				// put mu in a nicer format (reciprocal and respecting epsilon boundaries)
				fmu = fabs(mu); 																// absolute value
				efmu = (fmu > eps) ? fmu : eps / 2;								// with epsilon boundaries
				efmu_recip = 1.0 / efmu;												// reciprocal
				efmu_recipS2 = efmu_recip * efmu_recip;						// for STDEV

				/* make data go from left to right (for start and end)
       			 * and corner-case analysis*/
       			 

				// right-moving particles 
				if (mu >= 0)
				{	// NOTE unmodified mu, to get sign
					// left-most cell is the starting cell
					startcell = i; 
					// right-most cell is the ending cell
					if (end >= data.lx) 
					{	// corner case
						end = data.lx;
						endcell = numCells-1;

						// for tallying, record this corner case
						tallies->phi_right += weight * efmu_recip;//weight_efmu_recip; 
						tallies->E_right += weight * efmu;//weight_efmu; 
						tallies->J_right += weight;
					}
					else 
					{
						endcell = (int)(end * data.dx_recip);
					}
				}
				/* left-moving particles */
				else 
				{	// NOTE unmodified mu, to get sign
					// right-most cell is the starting cell
					endcell = i;
					// swap end and start
					temp = end; end = start; start = temp;
					// now, left-most cell is the starting cell
					if (start <= 0.0) 
					{	// corner case
						start = 0.0;
						startcell = 0;	
	
						// for tallying, record this corner case
						tallies->phi_left += weight * efmu_recip;//weight_efmu_recip; 
						tallies->E_left += weight * efmu; //weight_efmu; 
						tallies->J_left += -weight;	// note: sign is negative!
					}
					else 
					{
						startcell = (int)(start * data.dx_recip);
					}					
				}

				/* tally corner cells */
				/* todo: could factor out absdists and absdist^2s to increase latency tolerance? */
				if (startcell == endcell)
				{
					// tally once, with difference of particles
					absdist = (end-start);
					tallies->phi_n[startcell] += absdist * weight * efmu_recip; //weight_efmu_recip; 
				  	tallies->phi_n2[startcell] += absdist * absdist * weightS2 * efmu_recipS2; //weightS2_efmu_recipS2; 
				  	tallies->E_n[startcell] += absdist * weight * efmu; //weight_efmu; 
				}
				else
				{
					// starting cell
					begin_rightface = (startcell + 1) * data.dx;
					absdist = begin_rightface - start; // otherwise -0.0f can mess things up
					//assert(fabs(absdist) >= -0.00);
					tallies->phi_n[startcell] += absdist * weight * efmu_recip; //weight_efmu_recip; 
					tallies->phi_n2[startcell] += absdist * absdist * weightS2 * efmu_recipS2; //weightS2_efmu_recipS2; 
					tallies->E_n[startcell] += absdist * weight * efmu; //weight_efmu; 
					
					// ending cell
					end_leftface = endcell * data.dx;
					absdist = fabs(end - end_leftface);	// otherwise -0.0f can mess things up
					//assert(absdist >= -0.0); 
					tallies->phi_n[endcell] += absdist * weight * efmu_recip; //weight_efmu_recip; 
				  	tallies->phi_n2[endcell] += absdist * absdist * weightS2 * efmu_recipS2; //weightS2_efmu_recipS2; 
				  	tallies->E_n[endcell] += absdist * weight * efmu; //weight_efmu; 
				}

				weight_cellsize_efmu_recip = (pfloat)weight*efmu_recip*data.dx;
		  		weightS2_cellsizeS2_efmu_recipS2 = (pfloat)weightS2*efmu_recipS2*data.dx*data.dx;
		  		weight_cellsize_efmu = (pfloat)weight*efmu*data.dx;

 		 		/* tally "internal" cells (should be vectorizable!) */
	    		for (k = startcell+1; k <= endcell-1; k++) 
				{
					tallies->phi_n[k] += weight_cellsize_efmu_recip;  //weight * cellsize * efmu_recip; 
					tallies->phi_n2[k] += weightS2_cellsizeS2_efmu_recipS2;  //weightS2 * cellsizeS2 * efmu_recipS2; 
					tallies->E_n[k] += weight_cellsize_efmu; //weight * cellsize * efmu; 
		    	}

				// sanity checks
				//assert(abs(startcell) <= abs(endcell));
				//assert(startcell >= 0 && endcell <= numCells - 1);
				//assert(abs(start) <= abs(end));
				//assert(start >= 0.0);
				//assert(abs(end) <= abs(data.lx));
			}
	  	}
		
}

// filters the Edington tenser and the phi_lo
void f_filter(void * fv, void* f_avg_recv,int nx, int face_flag, int E_flag, int * f_filter_flag) {
  //face_flag = flag that checks if the quantity is defined as a face or cell
  //E_flag = flag that checks if the quantitty being filtered is Eddington Tensor or not
  //center
  //Initialize simulation parameters
  int i;
  if (E_flag == 1) { //Eddington tensor is being filtered
    if (face_flag == 1) { //quantity is defined as face value
      afloat * f = (afloat*)fv;
      afloat * f_avg = (afloat*)f_avg_recv; 
      for (i = 0; i < nx + 1; i++) {
	f_avg[i] = (f[i] >= 0.33f) ? f[i] : filter(f[i - (i == 0 ? 0 : 1)], f[i], f[i + (i == nx ? 0 : 1)]);
      }
      memcpy(f, f_avg, sizeof (afloat) * (nx + 1));
    } else if (face_flag == 0) { //quantity is defined as node value
      afloat * f = (afloat*)fv;
      afloat * f_avg = (afloat*)f_avg_recv;
      for (i = 0; i < nx; i++) {
	f_avg[i] = (f[i] >= 0.33f) ? f[i] : filter(f[i - (i == 0 ? 0 : 1)], f[i], f[i + (i == nx - 1 ? 0 : 1)]);

	f_filter_flag[i] = (f[i] >= 0.33f) ? 0 : 1;

      }
      memcpy(f, f_avg, sizeof (afloat) * nx);
    }
  } else { //phi LO is getting filtered
    pfloat * f = (pfloat*)fv;
    pfloat * f_avg = (pfloat*) f_avg_recv;
    
    for (i = 0; i < nx; i++) {
      f_avg[i] = (f_filter_flag[i]) ? filter(f[i - (i == 0 ? 0 : 1)], f[i], f[i + (i == nx - 1 ? 0 : 1)]) : f[i];
    }
    memcpy(f, f_avg, sizeof (pfloat) * nx);
  }
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

//Returns the first id in the device array of the desired device, or -1 if no such device is present
int getDevID(char * desired, cl_device_id * devices, int numDevices) 
{
    char buff[128];
    int i;
    for (i = 0; i < numDevices; i++) 
    {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, (void *) buff, NULL);
        //printf("%s\n", buff);
        if (strcmp(desired, buff) == 0) return i;
    }
    return -1;
}

/************************************************************************
 * oclInit
 *
 * Initializes environment varables for OpenCL kernel execution
 * by quering devices
 *
 * @param sysData
 * @return oclData
 **********************************************************************/
struct oclData oclInit(struct simData * sysData, struct gpuData * gData) 
{
    struct oclData oclEnv;

    //This section accumulates all devices from all platforms
    int i;
    cl_uint num_platforms = 0, num_devices = 0, temp_uint, temp_uint2;
    cl_int errcode;
    if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) printf("Failed to query platform count!\n");
    printf("\n**********************************************************************\n");
    printf("Number of Platforms: %d\n", num_platforms);

    oclEnv.platforms = (cl_platform_id *) malloc(sizeof (cl_platform_id) * num_platforms);

    if (clGetPlatformIDs(num_platforms, &oclEnv.platforms[0], NULL) != CL_SUCCESS) printf("Failed to get platform IDs\n");

    for (i = 0; i < num_platforms; i++) {
        temp_uint = 0;
        if (clGetDeviceIDs(oclEnv.platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &temp_uint) != CL_SUCCESS) printf("Failed to query device count on platform %d!\n", i);
        num_devices += temp_uint;
    }
    printf("Number of Devices: %d\n", num_devices);

    oclEnv.devices = (cl_device_id *) malloc(sizeof (cl_device_id) * num_devices);
    temp_uint = 0;
    for (i = 0; i < num_platforms; i++) {
        if (clGetDeviceIDs(oclEnv.platforms[i], CL_DEVICE_TYPE_ALL, num_devices, &oclEnv.devices[temp_uint], &temp_uint2) != CL_SUCCESS) printf("Failed to query device IDs on platform %d!\n", i);
        temp_uint += temp_uint2;
        temp_uint2 = 0;
    }

    //Simply print the names of all devices
    char buff[128];
    for (i = 0; i < num_devices; i++) {
        clGetDeviceInfo(oclEnv.devices[i], CL_DEVICE_NAME, 128, (void *) &buff[0], NULL);
        printf("Device %d: %s\n", i, buff);
    }

    //This is how you pick a specific device using an environment variable
    oclEnv.devID = -1;

    if (getenv("TARGET_DEVICE") != NULL) {
        oclEnv.devID = getDevID(getenv("TARGET_DEVICE"), &oclEnv.devices[0], num_devices);
        if (oclEnv.devID < 0) printf("Device \"%s\" not found.\nDefaulting to first device found.\n", getenv("TARGET_DEVICE"));
    } else {
        printf("Environment variable TARGET_DEVICE not set.\nDefaulting to first device found.\n");
    }

    oclEnv.devID = oclEnv.devID < 0 ? 0 : oclEnv.devID; //Ternary check to make sure gpuID is valid, if it's less than zero, default to zero, otherwise keep

    clGetDeviceInfo(oclEnv.devices[oclEnv.devID], CL_DEVICE_NAME, 128, (void *) &buff[0], NULL);
    printf("Selected Device %d: %s\n", oclEnv.devID, buff);

    //Use &devices[gpuID] as your device, like so, and proceed as normal
    //GPUContext = clCreateContext(NULL, 1, &devices[gpuID], NULL, NULL, &errcode);
	
    //create OpenCL context
    oclEnv.context = clCreateContext(NULL, 1, &oclEnv.devices[oclEnv.devID], NULL, NULL, &errcode); //clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create cl context!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }

    //create command-queue for OpenCL
    oclEnv.queue = clCreateCommandQueue(oclEnv.context, oclEnv.devices[oclEnv.devID], 0, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create cl command queue!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    
    //read OpenCL kernel source
    FILE* kernelFile = fopen("MCoCL.cl", "r");
    struct stat st;
    fstat(fileno(kernelFile), &st);
    char * kernelSource = (char*) calloc(st.st_size + 1, sizeof (char));
    fread(kernelSource, sizeof (char), st.st_size, kernelFile);
    fclose(kernelFile);
	
    oclEnv.program = clCreateProgramWithSource(oclEnv.context, 1, (const char **) &kernelSource, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create cl program!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    errcode = clBuildProgram(oclEnv.program, 0, NULL, "", NULL, NULL);
    if (errcode != CL_SUCCESS) {
        printf("failed to build cl program!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    size_t ret_val_size;

    errcode = clGetProgramBuildInfo(oclEnv.program, oclEnv.devices[oclEnv.devID], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    char * string = (char*) malloc(sizeof (char) * (ret_val_size + 1));
    errcode = clGetProgramBuildInfo(oclEnv.program, oclEnv.devices[oclEnv.devID], CL_PROGRAM_BUILD_LOG, ret_val_size, string, NULL);
    printf("BUILD LOG\n*******************************************************\n%s\n***************************************\n", string);

    oclEnv.warmup = clCreateKernel(oclEnv.program, "Kernel_RANLUXCL_Warmup", &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create RANLUX kernel!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    
    oclEnv.pGenerate2 = clCreateKernel(oclEnv.program, "particleGeneration2", &errcode);
	if (errcode != CL_SUCCESS) {
        printf("failed to create RANLUX kernel!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    
    cl_ulong memsize;
    errcode = clGetDeviceInfo(oclEnv.devices[oclEnv.devID], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (cl_ulong), &memsize, NULL);
    printf("Device has %llu bytes of global memory\n", memsize);
    printf("Using %d particles per sub-step\n", sysData->NPstep);
    printf("Adjusted to %d particles per iteration (in %d sub-steps)\n", sysData->NP, sysData->steps);
    printf("*******************************************************\n\n");

#ifdef DO_ASYNC
    oclEnv.clData = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (struct gpuData), NULL, &errcode);
    oclEnv.clMu0 = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, sizeof (cl_float) * sysData->NP*2, NULL, &errcode);
    oclEnv.clX0 = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, sizeof (cl_float) * sysData->NP*2, NULL, &errcode);
    oclEnv.clXf = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, sizeof (cl_float) * sysData->NP*2, NULL, &errcode);
#else
	oclEnv.clData = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (struct gpuData), NULL, &errcode);
    oclEnv.clMu0 = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, sizeof (cl_float) * sysData->NP, NULL, &errcode);
    oclEnv.clX0 = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, sizeof (cl_float) * sysData->NP, NULL, &errcode);
    oclEnv.clXf = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, sizeof (cl_float) * sysData->NP, NULL, &errcode);
#endif

    return oclEnv;
}
