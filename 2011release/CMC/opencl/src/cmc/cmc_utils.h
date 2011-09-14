#ifndef CMC_UTILS_H
#define CMC_UTILS_H

#include "../cmc_cli.h"

#define FLOAT_TOL 0.000001
#define MEMSCALENUM 13
#define MEMSCALEDENOM 16
#define RANSTATEPERPART 2800
int reducethresh;


#define filter(l,m,r) (.25f*l + .5f*m +.25*r)

/* #ifdef FLOAT64 */
/* typedef cl_double pfloat; */
/* #else */
/* typedef cl_float pfloat; */
/* #endif */


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
  float NP;
  cl_int nx;
  float lx;
  float dx;
  float dx_recip;
  float sig_t;
  float sig_t_recip;
  float sig_s;
  float eps;
  float q0_avg;
  float NPc;
} data __attribute__((aligned(8)));


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
  cl_kernel particleKern;
  cl_kernel warmup;
  cl_mem cldata;
  cl_mem all_tallies;
  cl_mem extra_NPc;
  cl_mem ranBuf;
  size_t nWorkGroups;
  size_t nWorkItems;
  int nstages;
  int devID;
};

 
struct oclData oclInit(struct data *);
void sim_param_print(struct data, struct oclData);
int collision_and_tally(struct data, struct oclData, pfloat*,pfloat*,pfloat*,int*);
void mean_std_calc(afloat*,afloat*,long long , long long, int , pfloat, struct meanDev*);

int compare_floats(const void *a, const void *b) {
    pfloat temp = *((pfloat *) a)-*((pfloat *) b);
    if (temp > 0.0f) return 1;
    else if (temp < 0.0f) return -1;
    else return 0;
}

int rcompare_floats(const void *a, const void *b) {
    return -compare_floats(a, b);
}

int compare_ints(const void *a, const void *b) {
    int temp = *((int *) a)-*((int *) b);
    if (temp > 0) return 1;
    else if (temp < 0) return -1;
    else return 0;
}

int rcompare_ints(const void *a, const void *b) {
    return -compare_ints(a, b);
}

#define floatEquals(a,b,tol) (fabs(a - b) < tol)


/* Generates a uniform random number on the range [low, high)
 */
pfloat unirand(pfloat low, pfloat high) {
    return rand()*1.0f / RAND_MAX * (high - low) + low;
}

int intsum(int * array, int count) {
    int i;
    int sum = 0;
    for (i = 0; i < count; i++) {
        sum += array[i];
    }
    return sum;
}

pfloat floatsum(pfloat * array, int count) {
    int i;
    pfloat sum = 0.0f;
    for (i = 0; i < count; i++) {
        sum += array[i];
    }
    return sum;
}

/* Generates a uniform random number on the range [0,1) */
inline pfloat unirand_0_1() {
    return (pfloat)rand() / RAND_MAX;
}


/* Generates a uniform random number on the range [low, high) */
inline pfloat unirand_1_1() {
    return (pfloat)rand() / RAND_MAX * 2 - 1;
}



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

/* for L2 norm */

// read in analytical solution, dataset 1 (Lx = 100, nx = 100, sigt = 10, sigs = 0.999)
inline void init_L2_norm_1(afloat * anal_soln){
	afloat anal_data[] = {91.15
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
	memcpy (anal_soln,anal_data,100*sizeof(afloat));
}

// read in analytical solution, dataset 1 (Lx = 100, nx = 100, sigt = 10, sigs = 0.999)
inline void init_L2_norm_2(afloat * anal_soln){
	afloat anal_data[] = {7.592
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
	memcpy (anal_soln,anal_data,100*sizeof(afloat));
}

// compare the generated solution with an analytical one (L2 norm)
inline pfloat l2_norm_cmp(afloat * phis, afloat * anal_soln, int numCells, pfloat dx){
  afloat sratio,temp;
	int i;
	sratio = 0.0;
	for (i = 0; i < numCells; i++){
	  temp = fabs(anal_soln[i] - phis[i]) / fabs(anal_soln[i]);
	  sratio += temp*temp;
	} 
	printf("L2: %f\n", dx*sratio);
	return dx * sratio;
}

#endif
