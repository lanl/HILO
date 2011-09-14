/* 
 * QDAMC Utility Functions
 */

#ifndef __QDAMC_UTILS_H__
#define __QDAMC_UTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include "../qdamc_cli.h"
#include "../ecodes.h"

/* utility functions */
inline int compare_floats(const void *a, const void *b) {
    pfloat temp = *((pfloat *) a)-*((pfloat *) b);
    if (temp > 0.0f) return 1;
    else if (temp < 0.0f) return -1;
    else return 0;
}

inline int rcompare_floats(const void *a, const void *b) {
    return -compare_floats(a, b);
}

inline int compare_ints(const void *a, const void *b) {
    int temp = *((int *) a)-*((int *) b);
    if (temp > 0) return 1;
    else if (temp < 0) return -1;
    else return 0;
}

inline int rcompare_ints(const void *a, const void *b) {
    return -compare_ints(a, b);
}

/* Generates a uniform random number on the range [0,1) */
inline pfloat unirand_0_1() {
    return (pfloat)rand() / RAND_MAX;
}

/* Generates a uniform random number on the range [low, high) */
inline pfloat unirand_1_1() {
    return (pfloat)rand() / RAND_MAX * 2 - 1;
}

/* Generates a uniform random number on the range [low, high) */
inline pfloat unirand(pfloat low, pfloat high) {
    return (pfloat)rand() / RAND_MAX * (high - low) + low;
}

inline int intsum(int * array, int count) {
    int i;
    int sum = 0;
    for (i = 0; i < count; i++) {
        sum += array[i];
    }
    return sum;
}

inline pfloat floatsum(pfloat * array, int count) {
    int i;
    pfloat sum = 0.0;
    for (i = 0; i < count; i++) {
        sum += array[i];
    }
    return sum;
}

inline pfloat maxFAS(pfloat * array, int leng, int stride) {
    pfloat max = array[0];
    int i;
    for (i = 0; i < leng; i += stride) {
        if (array[i] > max) max = array[i];
        //       printf("MAX= %f\n", max);
    }
    return max;
}

inline pfloat maxFA(pfloat * array, int leng) {
    return maxFAS(array, leng, 1);
}

inline pfloat rel_diff_calc(int nx, pfloat * phi, pfloat * phi_old) {
    pfloat * rel_diff = (pfloat *) calloc(nx, sizeof (pfloat));
    int i;
    for (i = 0; i < nx; i++) {
        rel_diff[i] = fabs((phi[i] - phi_old[i]) / phi[i]);
    }
    pfloat max = maxFA(rel_diff, nx);
    free(rel_diff);
    //printf("The relative difference between the scalar flux is, %f\n", max);
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
	       sratio += temp * temp;
	} 
	printf("L2: %f\n", dx*sratio);
	return dx * sratio;
}

#endif /* __QDAMC_UTILS_H__ */
