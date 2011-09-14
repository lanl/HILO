#include "stdio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "assert.h"
#include <sys/time.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include "ranluxcl.h"
#include "cmc_utils.h"
#include "ss.h"


int run()
{
  double NP = opt.NP;                                          //Total number of particles
  pfloat sysLeng = opt.sysLen;                                 //System length
  int numCells = opt.numCells;                                 //Number of cells in the system
  pfloat tol_std = opt.tol_std;                                //Tolerance for the std dev
  int i;
  
  pfloat dx = sysLeng / numCells;                              //cell width
  pfloat eps = 0.1;                                            //the threshold angle for tallying
  pfloat sig_t = opt.sig_t;                                    //Total cross-section
  pfloat sig_s = opt.sig_s;                                    //Scattering cross-section
    
  //The source term uniform for now	
  pfloat q0_avg = 1.0;
  
  //Stroing values in "data" struct
  data.NP = NP;
  data.nx = numCells;
  data.lx = sysLeng;
  data.dx = sysLeng / numCells;
  data.sig_t = sig_t;
  data.sig_t_recip = 1.0 / sig_t;
  data.dx_recip = 1.0 / dx;
  data.sig_s = sig_s;
  data.eps = eps;
  data.q0_avg = q0_avg;
  data.NPc = NP/numCells;
  
  //Initialize the OopenCL context, queue and the buffers
  struct oclData oclEnv = oclInit(&data);
  
  //Plot and print initial condition as well as simulation parameters
  sim_param_print(data,oclEnv);

  pfloat * phi_n = (pfloat*)malloc(sizeof(pfloat)*numCells);                               //Vector to hold value of phi_n
  pfloat * phi_n2 = (pfloat*)malloc(sizeof(pfloat)*numCells);                              //Value of phi_n2 for std dev calc
  pfloat * phi_n_tot = (pfloat*)calloc(numCells,sizeof(pfloat));                           //Sum of phi across all iterations
  pfloat * phiS2_n_tot = (pfloat*)calloc(numCells,sizeof(pfloat));                         //Sum of phi^2 across all iterations
  pfloat * phi_n_avg = (pfloat*)malloc(sizeof(pfloat)*numCells);                           //The average phi across all iterations
  pfloat * phi_n_work = (pfloat*)malloc(2*sizeof(pfloat)*numCells*oclEnv.nWorkGroups);     //Work array used to transfer data from GPU
  cl_int * extra_NPc = (cl_int*) malloc(sizeof(cl_int)*numCells );                         //Extra particles simulated by each workgroup ( more than NP )

#ifdef L2_1
  afloat * anal_soln = (afloat *) calloc(numCells, sizeof (afloat));                       // Reference solution
  pfloat l2;
  init_L2_norm_1(anal_soln);
#elif defined(L2_2)
  afloat * anal_soln = (afloat *) calloc(numCells, sizeof (afloat)); 			
  pfloat l2;
  init_L2_norm_2(anal_soln);
#else
  //Use standard deviation if analytical solution is not available
  long long samp_cnt = 0;
  struct meanDev * phi_nStats = (struct meanDev*) malloc(sizeof(struct meanDev)*opt.numCells); //Mean and standard deviation
#endif

 
  //time keeping structures
  struct timeval start_time, end_time, startPerIter, endPerIter;

  int iter = 0;                          //Total number of iterations
  int iter_avg = 0;                      //Total number of iterations to be used for averaging

  int flag_sim = 0;                      //Flag to say that system has converged
  
  //Start timing the simulations
  gettimeofday(&start_time, NULL);    

  // while not converged and total number of iterations < opt.numiters
  while (flag_sim == 0 && iter < opt.numIters) {

    //start time for iteration
    gettimeofday(&startPerIter, NULL);
    
    iter += 1;
    if(!opt.silent) printf("This is the %dth iteration\n", iter);

    //calls collision and tally OPENCL kernel code
    int ret_particles = collision_and_tally(data, oclEnv, phi_n, phi_n2, phi_n_work, extra_NPc);

    //Normalize the values of phi_n
    for( i = 0 ; i < numCells ; i++ )
      {
	phi_n[i] /= (afloat)(data.dx*(data.NP+ret_particles));
	phi_n[i] *= data.lx;
      }
    
    iter_avg += 1;
 
    //accumulate phase -- add new data to the running averages
    for (i = 0; i < numCells; i++) {
      phi_n_tot[i] += phi_n[i];
      phiS2_n_tot[i] += phi_n2[i];
    }

    // for each cell, calculate the average for phi_n, phi_lo and E_n
    for (i = 0; i < numCells; i++) {
      phi_n_avg[i] = phi_n_tot[i] / iter_avg;
    }
	
#if defined(L2_2) || defined(L2_1)
    //Use L2 norm to decide convergence if analytical soln is available
    l2 = l2_norm_cmp(phi_n_avg, anal_soln, numCells, dx);
    flag_sim = l2 <= tol_std;
    printf("L2 Norm with analytical solution = %lf\n",l2);
#else
    //Use mean and std dev to decide convergence
    samp_cnt = (long long)data.NP * ((long long) iter_avg);
    mean_std_calc(phi_n_tot,phiS2_n_tot,samp_cnt, (long long)data.NP, numCells, dx, phi_nStats) ;
    if( maxFAS(&phi_nStats[0].stdDev,numCells,2) <= tol_std )
      flag_sim = 1;
    printf("The maximum standard deviation of flux at node is, max (sig_phi) = %f\n", maxFAS(&phi_nStats[0].stdDev, numCells, 2)); 
#endif
    
    //End of one iteration
    gettimeofday(&endPerIter, NULL);
    printf("Time per Iteration: %ldu_sec\n\n", (endPerIter.tv_sec - startPerIter.tv_sec)*1000000 + (endPerIter.tv_usec - startPerIter.tv_usec));
  }
  
  //End of the simulation
  gettimeofday(&end_time, NULL);
  printf("Elapsed Time: %ldu_sec\n", (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));

  printf("NODEAVG\n");
  for (i = 0; i < numCells; i++) {
    printf("%d %lf\n", i, phi_n_avg[i]);
  }

  //Free all the memory
  free(phi_n);
  free(phi_n2);
  free(phi_n_tot);
  free(phiS2_n_tot);
  free(phi_n_avg);
#if defined(L2_1) || defined(L2_2)
  free(anal_soln);
#else
  free(phi_nStats);
#endif
  free(phi_n_work);

  return (EXIT_SUCCESS);
}

void sim_param_print(struct data data, struct oclData oclEnv) {
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
  printf("The total cross section is, sig_t = %lf, sig_t_recip = %lf\n", sig_t,data.sig_t_recip);
  printf("The scattering cross section is, sig_s = %f\n", sig_s);
  printf("Floating point data representation is, %lu byte\n", sizeof (pfloat));
  printf("Number of Workgroups = %d \n",(int)oclEnv.nWorkGroups);
  printf("Number of WorkItems = %d \n",(int)oclEnv.nWorkItems);
  printf("***********************************************************\n");
}


//Function that calls the OpenCL for streaming of particles. Does the reduction of tally from each workgroup
int collision_and_tally(struct data data, struct oclData oclEnv,  pfloat* phi_n, pfloat * phi_n2, pfloat* phi_n_work, cl_int * extra_NPc)
{
  int nx = data.nx;
  
  //Initialize the tallies to zero for this iteration
  memset(phi_n,0,sizeof(afloat)*nx);
  memset(phi_n2,0,sizeof(afloat)*nx);
  
  int j,k;
  size_t total_work_items = oclEnv.nWorkGroups * oclEnv.nWorkItems;

  //Enqueue the kernel to stream the particles
  clEnqueueNDRangeKernel(oclEnv.queue , oclEnv.particleKern, 1, NULL, &total_work_items, &oclEnv.nWorkItems, 0 , NULL, NULL );
  //Read the tallies for every workgroup from the GPU
  // [0-nWorkGroups*nx) : phi_n [nWorkGroups*nx2*-nWorkGroups*nx) : phi_n2
  clEnqueueReadBuffer(oclEnv.queue , oclEnv.all_tallies, CL_TRUE, 0, 2 * sizeof(pfloat) * oclEnv.nWorkGroups * nx , phi_n_work, 0, NULL,NULL );
  //Read the extra particles simulated by each workgroup
  clEnqueueReadBuffer(oclEnv.queue , oclEnv.extra_NPc, CL_TRUE , 0, sizeof(cl_int) * oclEnv.nWorkGroups , extra_NPc , 0 , NULL, NULL );
  
  //The offset where phi_n2 for all the worgroups are stored
  int n2_offset = oclEnv.nWorkGroups * nx;
  //Total number of extra particles simulated
  int ret_particles = 0;

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);
  //Reduce tallies across all workgroups
  for( j = 0 ; j < oclEnv.nWorkGroups ; j++ ) 
    {
      for( k = 0 ; k < nx ; k++ )
	{
	  phi_n[k] += phi_n_work[j*nx + k];
	  phi_n2[k] += phi_n_work[n2_offset + j*nx + k];
	}
      ret_particles += extra_NPc[j];
    }
  gettimeofday(&end_time, NULL);
  printf("Reduction Time: %ldu_sec\n", (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));
  
  printf("Extra particles = %d\n",ret_particles);
  return ret_particles;
}


//Function to initialize the OCL contex, queues and buffers
struct oclData oclInit(struct data * data) {

    struct oclData oclEnv;
    
    //This section accumulates all devices from all platforms
    int i;
    cl_uint num_platforms = 0, num_devices = 0, temp_uint, temp_uint2;
    cl_int errcode;
    if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) printf("Failed to query platform count!\n");
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

    //This is how you pick a specific device using an environment variable "TARGET_DEVICE"
    oclEnv.devID = -1;
    if (getenv("TARGET_DEVICE") != NULL) {
        oclEnv.devID = getDevID(getenv("TARGET_DEVICE"), &oclEnv.devices[0], num_devices);
        if (oclEnv.devID < 0) printf("Device \"%s\" not found.\nDefaulting to first device found.\n", getenv("TARGET_DEVICE"));
    } else {
        printf("Environment variable TARGET_DEVICE not set.\nDefaulting to first device found.\n");
    }

    oclEnv.devID = oclEnv.devID < 0 ? 0 : oclEnv.devID; //Ternary check to make sure gpuID is valid, if it's less than zero, default to zero, otherwise keep

    //TODO : Make sure you select the GPU
    clGetDeviceInfo(oclEnv.devices[oclEnv.devID], CL_DEVICE_NAME, 128, (void *) &buff[0], NULL);
    printf("Selected Device %d: %s\n", oclEnv.devID, buff);


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
    printf("Reading OpenCL source file...");
    FILE* kernelFile = fopen("CMCoCL.cl", "r");
    struct stat st;
    fstat(fileno(kernelFile), &st);
    char * kernelSource = (char*) calloc(st.st_size + 1, sizeof (char));
    fread(kernelSource, sizeof (char), st.st_size, kernelFile);
    fclose(kernelFile);
    printf("done\n");

    //Build the OpenCL kernel program
    printf("Creating Program from Source...");
    oclEnv.program = clCreateProgramWithSource(oclEnv.context, 1, (const char **) &kernelSource, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
      printf("failed to create cl program!\n\t%s\n", print_cl_errstring(errcode));
      errcode = CL_SUCCESS;
    }
    printf("done\n");
    
    char compiler_flag[19];
    sprintf(compiler_flag,"-DNUM_CELLS=%d -cl-fast-relaxed-math",data->nx);
    printf("Building OpenCL kernel...");
    errcode = clBuildProgram(oclEnv.program, 0, NULL, compiler_flag, NULL, NULL);
    if (errcode != CL_SUCCESS) {
        printf("failed to build cl program!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    printf("done\n");
    
    size_t ret_val_size;
    errcode = clGetProgramBuildInfo(oclEnv.program, oclEnv.devices[oclEnv.devID], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    char * string = (char*) malloc(sizeof (char) * (ret_val_size + 1));
    errcode = clGetProgramBuildInfo(oclEnv.program, oclEnv.devices[oclEnv.devID], CL_PROGRAM_BUILD_LOG, ret_val_size, string, NULL);
    printf("OpenCL Kernel Build Log\n*******************************************************\n%s\n*******************************************************\n", string);

    //Create OpenCL kernel objects
    oclEnv.particleKern = clCreateKernel(oclEnv.program, "gen_particles_tally", &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create xmc kernel!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.warmup = clCreateKernel(oclEnv.program, "Kernel_RANLUXCL_Warmup", &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create RANLUX warmup kernel!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }

    //scale particles-per-sub iteration to fit in the specified fraction of device memory
    cl_ulong memsize;
    cl_uint n_sms;
    errcode = clGetDeviceInfo(oclEnv.devices[oclEnv.devID], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (cl_ulong), &memsize, NULL);
    errcode = clGetDeviceInfo(oclEnv.devices[oclEnv.devID], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &n_sms, NULL);
    printf("Device has %lu bytes of global memory, number of SMs = %u \n", memsize,n_sms);
    
    oclEnv.nstages = 1;
    //Number of workgroups = number of cells
    oclEnv.nWorkGroups = data->nx;
    //Number of threads = number of cells
    oclEnv.nWorkItems = data->nx;

    printf("Number of stages = %d, number of workgroups = %d, size of cl_double = %d\n", oclEnv.nstages,(int)oclEnv.nWorkGroups,(int)sizeof(cl_double));
    
    //allocate device memory objects
    //Memory to hold the physical dimensions of the problem
    oclEnv.cldata = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (struct data), NULL, &errcode); 
    //Memory to hold tallies for each workgroup, size : 2 * numcells * numworkgroups * sizeof(double)
    oclEnv.all_tallies = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, 2 * sizeof (cl_double) * data->nx * oclEnv.nWorkGroups, NULL, &errcode);
    //Memory to hold the extra particles simulated by each workgroup, size : numworkgroups * sizeof(int)
    oclEnv.extra_NPc = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, sizeof(cl_int) * data->nx * oclEnv.nWorkGroups , NULL, &errcode);

    printf("Mem sizes , data : %lld, all_tallies : %lld\n",(long long)(sizeof(struct data)),2*(long long)(sizeof(cl_double) * data->nx * oclEnv.nWorkGroups ));

    //Initialize the ranluxcl buffer
    cl_int nskip, lux = 4;
    size_t RANDsomething;
    size_t Worksize = oclEnv.nWorkGroups * oclEnv.nWorkItems;
    cl_float4 * ranfloats = ranluxcl_initialization(lux, 0, Worksize, Worksize, &nskip, &RANDsomething);
    oclEnv.ranBuf = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, RANDsomething, NULL, NULL);
    printf("Randgen uses : %d\n",(int)RANDsomething);
    clEnqueueWriteBuffer(oclEnv.queue, oclEnv.ranBuf, CL_TRUE, 0, RANDsomething, ranfloats, 0, NULL, NULL);
    clSetKernelArg(oclEnv.warmup, 0, sizeof (cl_mem), (void*) &oclEnv.ranBuf);
    printf("RANDGEN STATUS %s\n", print_cl_errstring(clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.warmup, 1, NULL, &Worksize, &oclEnv.nWorkItems, 0, NULL, NULL)));
    printf("RANDGEN STATUS %s\n", print_cl_errstring(clFinish(oclEnv.queue)));
    
    //Setting arguements to the particleKern and writing the data.
    clSetKernelArg(oclEnv.particleKern , 0 , sizeof(cl_mem), (void*) &oclEnv.cldata );
    clSetKernelArg(oclEnv.particleKern , 1 , sizeof(cl_mem), (void*) &oclEnv.all_tallies );
    clSetKernelArg(oclEnv.particleKern , 2 , sizeof(cl_mem), (void*) &oclEnv.ranBuf );
    clSetKernelArg(oclEnv.particleKern , 3 , sizeof(cl_mem), (void*) &oclEnv.extra_NPc );
    clEnqueueWriteBuffer(oclEnv.queue, oclEnv.cldata, CL_TRUE, 0 , sizeof(struct data) , data, 0 ,NULL,NULL );
    
    return oclEnv;
}


void mean_std_calc(afloat * value, afloat * value2, long long samp_cnt, long long NP, int arrLen, pfloat scale, struct meanDev* retvals){
  int i;
  for (i = 0; i < arrLen; i++) {
    retvals[i].mean = (value[i] * NP * scale) / samp_cnt;
    retvals[i].stdDev = sqrt(fabs((value2[i] / samp_cnt) - pow(retvals[i].mean, 2.0f)) / (samp_cnt - 1));
  }
}
