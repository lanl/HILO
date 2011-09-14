/* 
 * Monte Carlo simulation, Header File
 */

#ifndef __CMC_H__
#define __CMC_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <malloc.h>
#include "../dSFMT-2.1/dSFMT.h"
#include "../cmc_cli.h"

/* constants */
#define FLOAT_TOL 0.000001
#define DEFAULT_Q0 1.0                                     // is malloc'ed and applied to every element

/* macros */
#define filter(l,m,r) (.25f*l + .5f*m +.25*r)
#define floatEquals(a,b,tol) (fabs(a - b) < tol)

/* data structures */

// using accumulation fp type
typedef struct {
  afloat * phi_n;
  afloat * phi_n2; 
  afloat * tot_col;
  afloat * full_buffer;
  int buffer_size;
} colData;

typedef struct {
  pfloat mean;
  pfloat stdDev;
} meanDev __attribute__ ((aligned (16)));


struct data {
  long long NP;
  pfloat lx;
  int nx;
  pfloat dx;
  pfloat dx2;				// cellsize^2
  pfloat dx_recip;
  pfloat sig_t;
  pfloat sig_t_recip;
  pfloat sig_s;
  pfloat eps;
  pfloat q0_avg;
  long long NP_tot;
  long long NPc;
} data  __attribute__ ((aligned (16)));

int run(int);
void sim_param_print(struct data);
double collision_and_tally(struct data, colData*,int,int,dsfmt_t*);
pfloat absdistcalc(int, int, int, pfloat, pfloat, pfloat, int);
void mean_std_calc(afloat *, afloat *, long long, long long, int, pfloat, meanDev*);
double gen_particles_and_tally(struct data, colData *,int,int,dsfmt_t*);
#endif /* __CMC_H__ */
