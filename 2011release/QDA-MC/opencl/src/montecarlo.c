/*
 * Monte Carlo simulation, first draft from Matlab source
 * Author: Paul Sathre (sathre@lanl.gov)
 *
 * we use NAN to represent the sqrt(-1) flag present in the matlab QDAMC code
 *
 * Created on May 26, 2011, 10:51 AM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>
#include "ranluxcl.h"
#include "qdamc_cli.h"

#ifdef DOUBLE
#include
#else
#include "ss.h"
#endif

#define FLOAT_TOL 0.000001
#define MEMSCALENUM 13
#define MEMSCALEDENOM 16
#define RANSTATEPERPART 2800

#define filter(l,m,r) (.25f*l + .5f*m +.25*r)

char *print_cl_errstring(cl_int err) {
    switch (err) {
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

//Returns the first id in the device array of the desired device, or -1 if no such device is present

int getDevID(char * desired, cl_device_id * devices, int numDevices) {
    char buff[128];
    int i;
    for (i = 0; i < numDevices; i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, (void *) buff, NULL);
        //printf("%s\n", buff);
        if (strcmp(desired, buff) == 0) return i;
    }
    return -1;

}

struct data {
    cl_int NP;
    cl_int NPc;
    cl_int NPstep;
    cl_int nx;
    cl_int lbc_flag;
    cl_int steps;
    pfloat lx;
    pfloat dx;
    pfloat n0;
    pfloat sig_t;
    pfloat sig_s;
    pfloat prob_s;
    pfloat Jp_l;
    pfloat eps;
    pfloat q0_lo;
    pfloat D;
    cl_ulong NP_tot;
} data __attribute__((aligned(8)));

struct triMat {
    pfloat * a; //sub-diagonal
    pfloat * b; //diagonal
    pfloat * c; //super-diagonal
};

struct colData {
    ss phi_left;
    ss J_left;
    ss E_left;
    ss phi_right;
    ss J_right;
    ss E_right;
    ss * phi_n;
    ss * phi_n2;
    ss * E_n;
};

struct meanDev {
    pfloat mean;
    pfloat stdDev;
};

struct oclData {
    cl_context context;
    cl_command_queue queue;
    cl_platform_id * platforms;
    cl_device_id * devices;
    cl_program program;
    cl_kernel xmcKern;
    cl_kernel warmup;
    cl_kernel moment;
    cl_kernel reset;
    cl_mem ranBuf;
    cl_mem clData;
    cl_mem clMu0;
    cl_mem clX0;
    cl_mem clXf;
    cl_mem clCell0;
    cl_mem clCellf;
    cl_mem clWx;
    cl_mem clQbar;
    cl_mem clHist;
    cl_int devID;
};

struct timedata {
    long long init;
    long long write;
    long long xmc;
    long long moment;
    long long gpureduce;
    long long pack;
    long long read;
    long long cpureduce;
    long long scale;
};

struct oclData oclInit(struct data *);
void oclCleanup(struct oclData *);

void sim_param_print(struct data);
void lo_solver(struct data, pfloat, pfloat, pfloat, pfloat, pfloat, pfloat, pfloat *, int, pfloat*,struct triMat,pfloat*);
void collision_and_tally(struct data, struct oclData, struct timedata *, pfloat *,struct colData*);
void triSolve(int, pfloat *, pfloat *, pfloat *, pfloat *, pfloat *);

#define floatEquals(a,b,tol) (fabs(a - b) < tol)

#if defined(L2_1) || defined(L2_2)
pfloat l2_norm_cmp(pfloat*,pfloat*,int,pfloat);
#if defined(L2_1)
void init_L2_norm_1(pfloat*);
#elif defined (L2_2)
void init_L2_norm_2(pfloat*);
#endif
#else
void mean_std_calc(ss *, ss *, unsigned long, int, int, pfloat, struct meanDev *);
#endif



pfloat maxFAS(pfloat * array, int leng, int stride) {
    pfloat max = array[0];
    int i;
    for (i = 0; i < leng; i += stride) {
        if (array[i] > max) max = array[i];
        //       printf("MAX= %f\n", max);
    }
    return max;
}

pfloat maxFA(pfloat * array, int leng) {
    return maxFAS(array, leng, 1);
}

pfloat rel_diff_calc(int nx, pfloat * phi, pfloat * phi_old) {
    pfloat * rel_diff = (pfloat *) calloc(nx, sizeof (pfloat));
    int i;
    for (i = 0; i < nx; i++) {
        rel_diff[i] = fabs((phi[i] - phi_old[i]) / phi[i]);
    }
    pfloat max = maxFA(rel_diff, nx);
    free(rel_diff);
    printf("The relative difference between the scalar flux is, %f\n", max);
    return max;
}

/*
 *
 */
int run(){
    int NP;
    pfloat sysLeng;
    int numCells;
    pfloat tolerance;
    pfloat tol_std;
    pfloat eps; //the threshold angle for tallying
    pfloat sig_t; //total collision cross-section
    pfloat sig_s; //scattering collisions cross-section
    int runningWidth = opt.runWidth;

    NP = opt.NP; 
    sysLeng = opt.sysLen;
    numCells = opt.numCells;
    tolerance = opt.tol;
    tol_std = opt.tol_std;
    eps = opt.eps;
    sig_t = opt.sig_t;
    sig_s = opt.sig_s;
    //Initialize random number generator, seed based on the current time
    srand(time(NULL));

    //Initialization of vital simulation parameters
    pfloat n0 = 0.01f; //base density parameter
    pfloat dx = sysLeng / numCells; //cell width
    pfloat D = 1 / (3 * sig_t); //diffusion coefficient
    pfloat prob_s = sig_s / sig_t; //probability of scattering
    int JP_l = 1; //the partial current on the left face
    pfloat * q0 = (pfloat *) calloc(numCells, sizeof (pfloat)); //The source term, uniform for now
    int i;
    for (i = 0; i < numCells; i++) {
        q0[i] = 1.0f;
    }
    pfloat q0_lo = 1.0f;

    struct timeval start_time, end_time;

    //Form data structure for simulation parameters
    data.NP = NP;
    data.lx = sysLeng;
    data.nx = numCells;
    data.dx = dx;
    data.n0 = n0;
    data.sig_t = sig_t;
    data.sig_s = sig_s;
    data.prob_s = prob_s;
    data.Jp_l = JP_l;
    data.eps = eps;
    data.q0_lo = q0_lo;
    data.D = D;

    struct timedata times;
    times.init = 0;
    times.write = 0;
    times.xmc = 0;
    times.moment = 0;
    times.gpureduce = 0;
    times.pack = 0;
    times.read = 0;
    times.cpureduce = 0;
    times.scale = 0;

    struct oclData oclEnv = oclInit(&data);
    int NPc = rint(data.NP * 1.0f / numCells); // number of particles in cell
    int NP_tot = NP; //total number of particles for the QDAMC
    data.NP_tot = NP_tot;
    data.NPc = NPc;
    
    printf("TEST %lld %d %d %d\n", (data.NPc>>1)*(1), 90128901 &(~1023), (~(((data.NPc>>1)*(1)) & 1023) + 1), (((data.NPc>>1)*(data.nx +1) + 1023) >> 10) -1);
    //Calculation of initial condition
    data.lbc_flag = 2;

    //Plot and print initial condition as well as simulation parameters
    sim_param_print(data);

    //Initialize the average values
    int iter = 0; //tally for iteration for convergence
    int iter_avg; // iteration for averaging
    if( runningWidth == 0 )
      iter_avg = 0;
    else
      iter_avg = 1;

    ss phi_left_tot; //tally for average phi
    phi_left_tot.num = 0.0f; //tally for average phi
    phi_left_tot.err = 0.0f; //tally for average phi
    ss phi_right_tot; //tally for average phi
    phi_right_tot.num = 0.0f; //tally for average phi
    phi_right_tot.err = 0.0f; //tally for average phi
    ss J_left_tot; //tally for average J
    J_left_tot.num = 0.0f; //tally for average J
    J_left_tot.err = 0.0f; //tally for average J
    ss J_right_tot; //tally for average J
    J_right_tot.num = 0.0f; //tally for average J
    J_right_tot.err = 0.0f; //tally for average J
    ss E_left_tot; //tally for average E
    E_left_tot.num = 0.0f; //tally for average E
    E_left_tot.err = 0.0f; //tally for average E
    ss E_right_tot; //tally for average E
    E_right_tot.num = 0.0f; //tally for average E
    E_right_tot.err = 0.0f; //tally for average E
    pfloat phi_left_avg = 0.0f;
    pfloat phi_right_avg = 0.0f;
    pfloat phi_left_avg_old = 0.0f;
    pfloat phi_right_avg_old = 0.0f;
    pfloat J_left_avg = 0.0f;
    pfloat J_right_avg = 0.0f;
    pfloat E_left_avg = 0.0f;
    pfloat E_right_avg = 0.0f;
    ss * phi_n_tot = (ss *) calloc(numCells, sizeof (ss));
    ss * phiS2_n_tot = (ss*) calloc(numCells, sizeof(ss));
    ss * phi_lo_tot = (ss *) calloc(numCells, sizeof (ss)); //tally for average lower order phi
    ss * E_n_tot = (ss *) calloc(numCells, sizeof (ss));
    pfloat * phi_n_avg = (pfloat *) calloc(numCells, sizeof (pfloat));
    pfloat * E_n_avg = (pfloat *) calloc(numCells, sizeof (pfloat));
    pfloat * phi_lo_avg = (pfloat *) calloc(numCells, sizeof (pfloat));
    pfloat * E_ho_n = (pfloat *) malloc(sizeof (pfloat) * numCells);
    pfloat * Qbar = (pfloat*)calloc(numCells, sizeof (pfloat));
    struct colData tallies;
    tallies.phi_n = (ss*)malloc(sizeof(ss)*numCells);
    tallies.phi_n2 = (ss*)malloc(sizeof(ss)*numCells);
    tallies.E_n = (ss*)malloc(sizeof(ss)*numCells);
    struct triMat A_lo;
    A_lo.a = (pfloat*) malloc(sizeof(pfloat)*(numCells-1));
    A_lo.b = (pfloat*) malloc(sizeof(pfloat)*(numCells));
    A_lo.c = (pfloat*) malloc(sizeof(pfloat)*(numCells-1));
    pfloat * b_lo = (pfloat*)malloc(sizeof(pfloat)*numCells);
    pfloat * phi_lo = (pfloat*)malloc(sizeof(pfloat)*numCells);

#if defined(L2_1) || defined(L2_2) || defined(L2_3)
    pfloat * anal_soln = (pfloat*)malloc(sizeof(pfloat)*numCells);                       //Array to hold the analytical solution 
    pfloat l2 = 0.0;
#if defined(L2_1)
    init_L2_norm_1(anal_soln);
#elif defined(L2_2)
    init_L2_norm_2(anal_soln);
#endif
#else
    struct meanDev * phi_nStats = (struct meanDev *) malloc(sizeof (struct meanDev) * numCells);
    unsigned long samp_cnt = 0; //total history tally
#endif

    for (i = 0; i < numCells; i++) {
        E_ho_n[i] = 1.0f / 3.0f;
    }
    lo_solver(data, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f / 3.0f, 1.0f / 3.0f, E_ho_n, 1,phi_lo,A_lo,b_lo);
    
    free(E_ho_n);
    

    //Calculation starts here
    //PAUL START TIMING
    gettimeofday(&start_time, NULL);
    
    int flag_sim = 0;
    // while not converged and total number of iterations < opt.numiters
    while (flag_sim == 0 && iter < opt.numIters) {
      
      iter += 1;
      if(!opt.silent ) printf("This is the %dth iteration\n",iter);

      if( iter == 1 )
	for (i = 0; i < numCells; i++) {
	  Qbar[i] = fabs(q0[i] + sig_s * phi_lo[i]);
	}
      else
	for (i = 0; i < numCells; i++) {
	  Qbar[i] = fabs(q0[i] + sig_s * phi_lo_avg[i]) ;
	}
            
      collision_and_tally(data, oclEnv, &times, Qbar, &tallies);
	    
      if( iter <= runningWidth )
	{
	  //accumulate phase
	  for (i = 0; i < numCells; i++) {
	    phi_n_tot[i] = tallies.phi_n[i];
	    phiS2_n_tot[i] = tallies.phi_n2[i];
	    phi_lo_tot[i].num = phi_lo[i];
	    E_n_tot[i] = tallies.E_n[i];
	  }
	  phi_left_tot = tallies.phi_left;
	  phi_right_tot = tallies.phi_right;
	  E_left_tot = tallies.E_left;
	  E_right_tot = tallies.E_right;
	  J_left_tot = tallies.J_left;
	  J_right_tot  = tallies.J_right;
	  
	  // for each cell, calculate the average for phi_n, phi_lo and E_n
	  for (i = 0; i < numCells; i++) {
	    phi_n_avg[i] = phi_n_tot[i].num ;
	    phi_lo_avg[i] = phi_lo_tot[i].num;
	    E_n_avg[i] = E_n_tot[i].num ;
	  }

	  // calculate the average for left and right faces of phi, E and J
	  phi_left_avg = phi_left_tot.num;
	  phi_right_avg = phi_right_tot.num;
	  E_left_avg = E_left_tot .num;
	  E_right_avg = E_right_tot .num;
	  J_left_avg = J_left_tot .num;
	  J_right_avg = J_right_tot.num;

	}
      else
	{
	  iter_avg++;

	  //accumulate phase
	  for (i = 0; i < numCells; i++) {
            phi_n_tot[i] = ss_ss_add(tallies.phi_n[i], phi_n_tot[i]);
	    phiS2_n_tot[i] = ss_ss_add(tallies.phi_n2[i], phiS2_n_tot[i]);
            phi_lo_tot[i] = num_ss_add(phi_lo[i], phi_lo_tot[i]);
            E_n_tot[i] = ss_ss_add(tallies.E_n[i], E_n_tot[i]);
	  }
	  phi_left_tot = ss_ss_add(tallies.phi_left, phi_left_tot);
	  phi_right_tot = ss_ss_add(tallies.phi_right, phi_right_tot);
	  E_left_tot = ss_ss_add(tallies.E_left, E_left_tot);
	  E_right_tot = ss_ss_add(tallies.E_right, E_right_tot);
	  J_left_tot = ss_ss_add(tallies.J_left, J_left_tot);
	  J_right_tot = ss_ss_add(tallies.J_right, J_right_tot);
	  //reduce phase
	  for (i = 0; i < numCells; i++) {
            phi_n_avg[i] = phi_n_tot[i].num / iter_avg;
            phi_lo_avg[i] = phi_lo_tot[i].num / iter_avg;
            E_n_avg[i] = E_n_tot[i].num / iter_avg;
	  }
	  phi_left_avg = phi_left_tot.num / iter_avg;
	  phi_right_avg = phi_right_tot.num / iter_avg;
	  E_left_avg = E_left_tot.num / iter_avg;
	  E_right_avg = E_right_tot.num / iter_avg;
	  J_left_avg = J_left_tot.num / iter_avg;
	  J_right_avg = J_right_tot.num / iter_avg;

	  //check for convergence
#if !defined(L2_1) && !defined(L2_2) && !defined(L2_3)
	  samp_cnt = NP_tot * ((long) iter_avg);
	  mean_std_calc(phi_n_tot, phiS2_n_tot, samp_cnt, opt.NP, opt.numCells, opt.dx, phi_nStats);

	  printf("The maximum standard deviation of flux at node is, max (sig_phi) = %f\n", maxFAS(&phi_nStats[0].stdDev, numCells, 2));

	  if (maxFAS(&phi_nStats[0].stdDev, numCells, 2) <= tol_std) {
	    flag_sim = 1;
	  }
#else
	  l2 = l2_norm_cmp(phi_n_avg, anal_soln, numCells, dx);
	  flag_sim = l2 <= tol_std;
	  if(!opt.silent ){
	    gettimeofday(&end_time, NULL);
	    printf("L2: %f, Sofar: %ldu_sec\n", l2, (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));
	  }
#endif


	}

      lo_solver(data, phi_left_avg, phi_right_avg, J_left_avg, J_right_avg, E_left_avg, E_right_avg, E_n_avg, 0, phi_lo, A_lo,b_lo);
    }
    
        //PAUL END TIMING
    gettimeofday(&end_time, NULL);
    
    if( !opt.silent ){
      printf("NODEAVG\n");
      for (i = 0; i < numCells; i++) {
	printf("%d %f %f %f\n", i, phi_n_avg[i], E_n_avg[i], phi_lo_avg[i]);
      }
    }

    free(Qbar);
    free(q0);
    free(phi_n_tot);
    free(phi_lo_tot);
    free(E_n_tot);
    free(phi_n_avg);
    free(E_n_avg);
    free(phi_lo_avg);
    free(tallies.phi_n);
    free(tallies.phi_n2);
    free(tallies.E_n);
    
#if defined(L2_1) || defined(L2_2) || defined(L2_3)
    free(anal_soln);
#else
    free(phi_nStats);
#endif
    
    free(phi_lo);
    free(A_lo.a);
    free(A_lo.b);
    free(A_lo.c);
    free(b_lo);
    oclCleanup(&oclEnv);
    printf("Cumulative Iteration Time Stats\n\tInitialize %lld  Write %lld  MonteCarlo %lld\n\tGPU Histogram %lld  GPU Reduce %lld  GPU Data Packing %lld\n\tRead %lld  CPU Reduce %lld  CPU Scaling %lld\n", times.init, times.write, times.xmc, times.moment, times.gpureduce, times.pack, times.read, times.cpureduce, times.scale);
    printf("Total Elapsed Time: %lldu_sec\n", (end_time.tv_sec - start_time.tv_sec)*1000000 + (end_time.tv_usec - start_time.tv_usec));

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
void lo_solver(struct data data, pfloat phi_ho_left, pfloat phi_ho_right, pfloat J_ho_left, pfloat J_ho_right, pfloat E_ho_left, pfloat E_ho_right, pfloat * E_ho_n, int ic, pfloat* phi_lo,struct triMat A, pfloat* b) {

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


/*
 Manages all device side Monte-Carlo and histogramming steps and sub-iterations*/
void collision_and_tally(struct data data, struct oclData oclEnv, struct timedata * times, pfloat * Qbar , struct colData * tallies) {
    //initialize simulation parameters
    int nx = data.nx;
    pfloat dx = data.dx;
    int NP = data.NP;
    int i, j, subitr;
    struct timeval start, end;
    long long int inittime = 0, writetime = 0, xmctime = 0, histtime = 0, gpuredtime = 0, gpupacktime = 0, readtime = 0, cpuredtime = 0, cpuscaletime = 0;

    gettimeofday(&start, NULL);

    memset(tallies->phi_n,0,sizeof(ss)*nx);
    memset(tallies->phi_n2,0,sizeof(ss)*nx);
    memset(tallies->E_n,0,sizeof(ss)*nx);
    tallies->E_left.num = 0.0f;
    tallies->E_left.err = 0.0f;
    tallies->J_left.num = 0.0f;
    tallies->J_left.err = 0.0f;
    tallies->phi_left.num = 0.0f;
    tallies->phi_left.err = 0.0f;
    tallies->E_right.num = 0.0f;
    tallies->E_right.err = 0.0f;
    tallies->J_right.num = 0.0f;
    tallies->J_right.err = 0.0f;
    tallies->phi_right.num = 0.0f;
    tallies->phi_right.err = 0.0f;


   // data.steps>>6;


    clSetKernelArg(oclEnv.reset, 0, sizeof (cl_mem), (void*) &oclEnv.clHist);
    size_t Worksize = (data.nx + 2)*3 * 1;

    clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.reset, 1, NULL, &Worksize, NULL, 0, NULL, NULL);

    clFinish(oclEnv.queue);
    gettimeofday(&end, NULL);
    inittime = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    gettimeofday(&start, NULL);

    cl_int numsubs, offset, flag;
    clEnqueueWriteBuffer(oclEnv.queue, oclEnv.clData, CL_FALSE, 0, sizeof (struct data), &data, 0, NULL, NULL);
    clEnqueueWriteBuffer(oclEnv.queue, oclEnv.clQbar, CL_TRUE, 0, sizeof (pfloat) * data.nx, Qbar, 0, NULL, NULL);
    clFinish(oclEnv.queue);
    gettimeofday(&end, NULL);
    writetime = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    gettimeofday(&start, NULL);
    clSetKernelArg(oclEnv.xmcKern, 0, sizeof (cl_mem), (void*) &oclEnv.clData);
    clSetKernelArg(oclEnv.xmcKern, 1, sizeof (cl_mem), (void*) &oclEnv.clMu0);
    clSetKernelArg(oclEnv.xmcKern, 2, sizeof (cl_mem), (void*) &oclEnv.clX0);
    clSetKernelArg(oclEnv.xmcKern, 3, sizeof (cl_mem), (void*) &oclEnv.clXf);
    clSetKernelArg(oclEnv.xmcKern, 4, sizeof (cl_mem), (void*) &oclEnv.clCell0);
    clSetKernelArg(oclEnv.xmcKern, 5, sizeof (cl_mem), (void*) &oclEnv.clCellf);
    clSetKernelArg(oclEnv.xmcKern, 6, sizeof (cl_mem), (void*) &oclEnv.clWx);
    clSetKernelArg(oclEnv.xmcKern, 7, sizeof (cl_mem), (void*) &oclEnv.clQbar);
    clSetKernelArg(oclEnv.xmcKern, 8, sizeof (cl_mem), (void*) &oclEnv.ranBuf);
    Worksize = data.NP>>6;
    size_t Localsize = 1024;

    printf(" XMC Enqueue %s\n", print_cl_errstring(clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.xmcKern, 1, NULL, &Worksize, NULL, 0, NULL, NULL)));

    printf(" XMC Finish %s\n", print_cl_errstring(clFinish(oclEnv.queue)));
    gettimeofday(&end, NULL);
    xmctime = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    clFinish(oclEnv.queue);
    gettimeofday(&start, NULL);


    data.steps <<= 6;
    //printf("Steps! %d\n", data.steps);
    clEnqueueWriteBuffer(oclEnv.queue, oclEnv.clData, CL_TRUE, 0, sizeof (struct data), &data, 0, NULL, NULL);

    clSetKernelArg(oclEnv.moment, 0, sizeof (cl_mem), (void*) &oclEnv.clData);
    clSetKernelArg(oclEnv.moment, 1, sizeof (cl_mem), (void*) &oclEnv.clMu0);
    clSetKernelArg(oclEnv.moment, 2, sizeof (cl_mem), (void*) &oclEnv.clX0);
    clSetKernelArg(oclEnv.moment, 3, sizeof (cl_mem), (void*) &oclEnv.clXf);
    clSetKernelArg(oclEnv.moment, 4, sizeof (cl_mem), (void*) &oclEnv.clCell0);
    clSetKernelArg(oclEnv.moment, 5, sizeof (cl_mem), (void*) &oclEnv.clCellf);
    clSetKernelArg(oclEnv.moment, 6, sizeof (cl_mem), (void*) &oclEnv.clWx);
    clSetKernelArg(oclEnv.moment, 7, sizeof (cl_mem), (void*) &oclEnv.clHist);
    Worksize = 1024 * (data.nx + 2);
    Localsize = 1024;

    printf(" Hist Enqueue %s\n", print_cl_errstring(clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.moment, 1, NULL, &Worksize, &Localsize, 0, NULL, NULL)));
    printf(" Hist Finish %s\n", print_cl_errstring(clFinish(oclEnv.queue)));
    gettimeofday(&end, NULL);
    histtime += (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    gettimeofday(&start, NULL);
    numsubs = 1;
    offset = 0;
    ss * subs = calloc(numsubs * (nx + 2)*3, sizeof (ss));

    gettimeofday(&start, NULL);
    //printf("HULLABALLOO\n");
    clEnqueueReadBuffer(oclEnv.queue, oclEnv.clHist, CL_TRUE, 0, numsubs * (nx + 2)*3 * sizeof (ss), subs, 0, NULL, NULL);
    //printf("bonanaza\n");
    clFinish(oclEnv.queue);
    gettimeofday(&end, NULL);
    readtime = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    gettimeofday(&start, NULL);

    for (i = 0; i < numsubs; i++) {
        offset = i;
        tallies->phi_left = ss_ss_add(subs[offset], tallies->phi_left);
        offset += numsubs;
        tallies->E_left = ss_ss_add(subs[offset], tallies->E_left);
        offset += numsubs;
        tallies->J_left = ss_ss_add(subs[offset], tallies->J_left);
        offset += numsubs;
        tallies->phi_right = ss_ss_add(subs[offset], tallies->phi_right);
        offset += numsubs;
        tallies->E_right = ss_ss_add(subs[offset], tallies->E_right);
        offset += numsubs;
        tallies->J_right = ss_ss_add(subs[offset], tallies->J_right);
        offset += numsubs;
        for (j = 0; j < data.nx; j++) {
            tallies->phi_n[j] = ss_ss_add(subs[offset], tallies->phi_n[j]);
            offset += numsubs;
            tallies->phi_n2[j] = ss_ss_add(subs[offset], tallies->phi_n2[j]);
            offset += numsubs;
            tallies->E_n[j] = ss_ss_add(subs[offset], tallies->E_n[j]);
        //printf("%d %f %f %f\n", j, tallies->phi_n[j], tallies->phi_n2[j], tallies->E_n[j]);
            offset += numsubs;
        }

    }

    gettimeofday(&end, NULL);
    cpuredtime = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    gettimeofday(&start, NULL);

    free(subs);


    if (!floatEquals(tallies->phi_left.num, 0.0f, FLOAT_TOL)) {
        tallies->E_left = ss_ss_div(tallies->E_left, tallies->phi_left);
    } else {
        tallies->E_left.num = 1.0f / 3.0f;
        tallies->E_left.err = 0.0f;
    }
    if (!floatEquals(tallies->phi_right.num, 0.0f, FLOAT_TOL)) {
        tallies->E_right = ss_ss_div(tallies->E_right, tallies->phi_right);
    } else {
        tallies->E_right.num = 1.0f / 3.0f;
        tallies->E_right.err = 0.0f;
    }
    tallies->phi_left = ss_num_div(tallies->phi_left, (float) data.NP);
    tallies->J_left = ss_num_div(tallies->J_left, (float) data.NP);
    tallies->phi_right = ss_num_div(tallies->phi_right, (float) data.NP);
    tallies->J_right = ss_num_div(tallies->J_right, (float) data.NP);
    // Scaling the moments for node values
    for (i = 0; i < nx; i++) {
        if (!floatEquals(tallies->phi_n[i].num, 0.0f, FLOAT_TOL)) {
            tallies->E_n[i] = ss_ss_div(tallies->E_n[i], tallies->phi_n[i]);
        } else {
            tallies->E_n[i].num = 1.0f / 3.0f;
            tallies->E_n[i].err = 0.0f;
        }
        tallies->phi_n[i] = ss_num_div(tallies->phi_n[i],data.dx* data.NP);
 //       tallies->phi_n2[i] = ss_num_div(tallies->phi_n2[i], dx*dx );
    }

    gettimeofday(&end, NULL);
    cpuscaletime = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);

    printf("Time Stats\n\tInitialize %lld  Write %lld  MonteCarlo %lld\n\tGPU Histogram %lld  GPU Reduce %lld  GPU Data Packing %lld\n\tRead %lld  CPU Reduce %lld  CPU Scaling %lld\n", inittime, writetime, xmctime, histtime, gpuredtime, gpupacktime, readtime, cpuredtime, cpuscaletime);
    times->init += inittime;
    times->write += writetime;
    times->xmc += xmctime;
    times->moment += histtime;
    times->gpureduce += gpuredtime;
    times->pack += gpupacktime;
    times->read += readtime;
    times->cpureduce += cpuredtime;
    times->scale += cpuscaletime;
}

struct oclData oclInit(struct data * sysData) {

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
    FILE* kernelFile = fopen("src/MCoCL.cl", "r");
    struct stat st;
    fstat(fileno(kernelFile), &st);
    char * kernelSource = (char*) calloc(st.st_size + 1, sizeof (char));
    fread(kernelSource, sizeof (char), st.st_size, kernelFile);
    fclose(kernelFile);

    //Build the OpenCL kernel program
    oclEnv.program = clCreateProgramWithSource(oclEnv.context, 1, (const char **) &kernelSource, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create cl program!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
#ifdef DOUBLE
    errcode = clBuildProgram(oclEnv.program, 0, NULL, "-cl-fast-relaxed-math -D DOUBLE", NULL, NULL);
#else
    errcode = clBuildProgram(oclEnv.program, 0, NULL, "-cl-fast-relaxed-math -D NVIDIA", NULL, NULL);
#endif
    if (errcode != CL_SUCCESS) {
        printf("failed to build cl program!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    size_t ret_val_size;
    errcode = clGetProgramBuildInfo(oclEnv.program, oclEnv.devices[oclEnv.devID], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    char * string = (char*) malloc(sizeof (char) * (ret_val_size + 1));
    errcode = clGetProgramBuildInfo(oclEnv.program, oclEnv.devices[oclEnv.devID], CL_PROGRAM_BUILD_LOG, ret_val_size, string, NULL);
    printf("OpenCL Kernel Build Log\n*******************************************************\n%s\n*******************************************************\n", string);

    //Create OpenCL kernel objects
    oclEnv.xmcKern = clCreateKernel(oclEnv.program, "xmcKern", &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create xmc kernel!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.warmup = clCreateKernel(oclEnv.program, "Kernel_RANLUXCL_Warmup", &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create RANLUX warmup kernel!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.moment = clCreateKernel(oclEnv.program, "cell1024Kern", &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create moment calc kernel!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.reset = clCreateKernel(oclEnv.program, "resetSaveHist", &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create saved histogram reset kernel!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }

    //scale particles-per-sub iteration to fit in the specified fraction of device memory
    cl_ulong memsize;
    errcode = clGetDeviceInfo(oclEnv.devices[oclEnv.devID], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (cl_ulong), &memsize, NULL);
    printf("Device has %llu bytes of global memory\n", memsize);
    sysData->NPstep = 64*(memsize - sizeof (cl_float) * 6 * (2 + sysData->nx)) * MEMSCALENUM / ((64*6 + RANSTATEPERPART) * sizeof (float) *MEMSCALEDENOM);
    int temp = (sysData->NP < sysData->NPstep ? sysData->NP : sysData->NPstep);
    int bitmask = 1024; //2^15 - nVidia GeForce GTX 580 has some sort of fatal flaw at exactly 2^16, so for safety we've limited it to 2^15
    //bitmask = 65536;
    //find the most significant bit of the minimum of NPstep and NP, to restrict particles per sub-iteration to powers of two
    //if we're already less than NPstep, and "rounding down" to the next power of two would make us less than NP
    // then don't round down, so we only have to perform one sub-iteration
    //   while (!((bitmask & temp) || ((bitmask >> 1 < sysData->NP) && (bitmask < sysData->NPstep)))) {
    //       bitmask >>= 1;
    //   }
    sysData->NPstep = bitmask;
    //sysData->NPstep = 32768;
    //    sysData->NPstep = 10000000;
    printf("Using %d particles per sub-step (%llu bytes)\n", sysData->NPstep, (6*64 + RANSTATEPERPART) * sizeof (float) *sysData->NPstep/64);
    sysData->steps = sysData->NP / sysData->NPstep + (sysData->NP % sysData->NPstep > 0 ? 1 : 0);
    sysData->steps += (sysData->steps % sysData->nx > 0 ? (sysData->nx-(sysData->steps % sysData->nx)) : 0);
    sysData->NP = sysData->NPstep * sysData->steps;

    printf("Adjusted to %d particles per iteration (in %d sub-steps)\n", sysData->NP, sysData->steps);

    //allocate device memory objects
    oclEnv.clData = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (struct data), NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create GPU system parameter buffer!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.clMu0 = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (float) * sysData->NP, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create GPU mu buffer!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.clX0 = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (float) * sysData->NP, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create GPU leftmost position buffer!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.clXf = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (float) * sysData->NP, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create GPU rightmost position buffer!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.clCell0 = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (int) * sysData->NP, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create GPU leftmost cell buffer!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.clCellf = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (int) * sysData->NP, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create GPU rightmost cell buffer!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.clWx = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (float) * sysData->NP, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create GPU weight buffer!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.clQbar = clCreateBuffer(oclEnv.context, CL_MEM_READ_ONLY, sizeof (cl_float) * sysData->nx, NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create GPU Qbar buffer!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }
    oclEnv.clHist = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, sizeof (cl_float2) * 3 * (2 + sysData->nx), NULL, &errcode);
    if (errcode != CL_SUCCESS) {
        printf("failed to create GPU tally buffer!\n\t%s\n", print_cl_errstring(errcode));
        errcode = CL_SUCCESS;
    }


    cl_int nskip, lux = 4;
    size_t RANDsomething;
    sysData->steps >>=6;
    size_t Worksize = sysData->NP>>6;
    cl_float4 * ranfloats = ranluxcl_initialization(lux, 0, Worksize, Worksize, &nskip, &RANDsomething);
    oclEnv.ranBuf = clCreateBuffer(oclEnv.context, CL_MEM_READ_WRITE, RANDsomething, NULL, NULL);
    clEnqueueWriteBuffer(oclEnv.queue, oclEnv.ranBuf, CL_TRUE, 0, RANDsomething, ranfloats, 0, NULL, NULL);
    clSetKernelArg(oclEnv.warmup, 0, sizeof (cl_mem), (void*) &oclEnv.ranBuf);
    printf("RANDGEN STATUS %s\n", print_cl_errstring(clEnqueueNDRangeKernel(oclEnv.queue, oclEnv.warmup, 1, NULL, &Worksize, NULL, 0, NULL, NULL)));
    printf("RANDGEN STATUS %s\n", print_cl_errstring(clFinish(oclEnv.queue)));
    free(ranfloats);
    return oclEnv;
}

void oclCleanup(struct oclData * oclEnv) {
    clReleaseMemObject(oclEnv->ranBuf);
    clReleaseMemObject(oclEnv->clHist);
    clReleaseMemObject(oclEnv->clQbar);
    clReleaseMemObject(oclEnv->clWx);
    clReleaseMemObject(oclEnv->clMu0);
    clReleaseMemObject(oclEnv->clX0);
    clReleaseMemObject(oclEnv->clXf);
    clReleaseMemObject(oclEnv->clCell0);
    clReleaseMemObject(oclEnv->clCellf);
    clReleaseMemObject(oclEnv->clData);
    clReleaseKernel(oclEnv->warmup);
    clReleaseKernel(oclEnv->reset);
    clReleaseKernel(oclEnv->moment);
    clReleaseKernel(oclEnv->xmcKern);
    clReleaseProgram(oclEnv->program);
    clReleaseCommandQueue(oclEnv->queue);
    clReleaseContext(oclEnv->context);
}

#if defined(L2_1) || defined(L2_2)
// compare the generated solution with an analytical one (L2 norm)
 pfloat l2_norm_cmp(pfloat * phis, pfloat * anal_soln, int numCells, pfloat dx){
  pfloat sratio,temp;
	int i;
	sratio = 0.0;
	for (i = 0; i < numCells; i++){
	       temp = fabs(anal_soln[i] - phis[i]) / fabs(anal_soln[i]);
	       sratio += temp * temp;
	} 
	//printf("L2: %f\n", dx*sratio);
	return dx * sratio;
}


#if defined(L2_1)
// read in analytical solution, dataset 1 (Lx = 100, nx = 100, sigt = 10, sigs = 0.999)
 void init_L2_norm_1(pfloat * anal_soln){
	pfloat anal_data[] = {91.15
									, 236.09
									, 357.65
									, 459.89
									, 545.85
									, 618.13
									, 678.91
									, 730.01
									, 772.98
									, 809.11
									, 839.49
									, 865.04
									, 886.52
									, 904.58
									, 919.77
									, 932.54
									, 943.27
									, 952.3
									, 959.89
									, 966.28
									, 971.64
									, 976.16
									, 979.95
									, 983.14
									, 985.82
									, 988.08
									, 989.98
									, 991.57
									, 992.91
									, 994.04
									, 994.98
									, 995.78
									, 996.45
									, 997.01
									, 997.48
									, 997.88
									, 998.21
									, 998.49
									, 998.72
									, 998.92
									, 999.08
									, 999.22
									, 999.33
									, 999.42
									, 999.49
									, 999.55
									, 999.59
									, 999.63
									, 999.65
									, 999.66
									, 999.66
									, 999.65
									, 999.63
									, 999.59
									, 999.55
									, 999.49
									, 999.42
									, 999.33
									, 999.22
									, 999.08
									, 998.92
									, 998.72
									, 998.49
									, 998.21
									, 997.88
									, 997.48
									, 997.01
									, 996.45
									, 995.78
									, 994.98
									, 994.04
									, 992.91
									, 991.57
									, 989.98
									, 988.08
									, 985.82
									, 983.14
									, 979.95
									, 976.16
									, 971.64
									, 966.28
									, 959.89
									, 952.3
									, 943.27
									, 932.54
									, 919.77
									, 904.58
									, 886.52
									, 865.04
									, 839.49
									, 809.11
									, 772.98
									, 730.01
									, 678.91
									, 618.13
									, 545.85
									, 459.89
									, 357.65
									, 236.09
									, 91.15};
	memcpy (anal_soln,anal_data,100*sizeof(pfloat));
}

#elif defined(L2_2)
// read in analytical solution, dataset 1 (Lx = 100, nx = 100, sigt = 10, sigs = 0.999)
 void init_L2_norm_2(pfloat * anal_soln){
	pfloat anal_data[] = {7.592
												, 9.1071
												, 10.398
												, 11.586
												, 12.702
												, 13.763
												, 14.779
												, 15.754
												, 16.692
												, 17.598
												, 18.472
												, 19.317
												, 20.133
												, 20.923
												, 21.686
												, 22.423
												, 23.136
												, 23.824
												, 24.488
												, 25.129
												, 25.746
												, 26.34
												, 26.912
												, 27.462
												, 27.989
												, 28.495
												, 28.979
												, 29.441
												, 29.882
												, 30.303
												, 30.702
												, 31.08
												, 31.438
												, 31.775
												, 32.092
												, 32.389
												, 32.665
												, 32.921
												, 33.157
												, 33.374
												, 33.57
												, 33.747
												, 33.903
												, 34.04
												, 34.158
												, 34.256
												, 34.334
												, 34.392
												, 34.431
												, 34.451
												, 34.451
												, 34.431
												, 34.392
												, 34.334
												, 34.256
												, 34.158
												, 34.04
												, 33.903
												, 33.747
												, 33.57
												, 33.374
												, 33.157
												, 32.921
												, 32.665
												, 32.389
												, 32.092
												, 31.775
												, 31.438
												, 31.08
												, 30.702
												, 30.303
												, 29.882
												, 29.441
												, 28.979
												, 28.495
												, 27.989
												, 27.462
												, 26.912
												, 26.34
												, 25.746
												, 25.129
												, 24.488
												, 23.824
												, 23.136
												, 22.423
												, 21.686
												, 20.923
												, 20.133
												, 19.317
												, 18.472
												, 17.598
												, 16.692
												, 15.754
												, 14.779
												, 13.763
												, 12.702
												, 11.586
												, 10.398
												, 9.1071
												, 7.592};
	memcpy (anal_soln,anal_data,100*sizeof(pfloat));
}
#endif
#else

void mean_std_calc(ss * value, ss * value2, unsigned long samp_cnt, int NP, int arrLen, pfloat scale, struct meanDev* retvals) {
    int i;
    for (i = 0; i < arrLen; i++) {
        retvals[i].mean = (value[i].num * NP * scale) / samp_cnt;
        retvals[i].stdDev = sqrt(fabs((value2[i].num / samp_cnt) - pow(retvals[i].mean, 2.0f)) / (samp_cnt - 1));
    }
}
#endif
