/* 
 * Monte Carlo simulation, Header File
 */

#ifndef __QDAMC_H__
#define __QDAMC_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <malloc.h>
#include "../qdamc_cli.h"
#include "../dSFMT-2.1/dSFMT.h"

/* constants */
#define FLOAT_TOL 0.000001
#define DEFAULT_Q0 1.0                                     // is malloc'ed and applied to every element

/* macros */
#define filter(l,m,r) (.25f*l + .5f*m +.25*r)
#define floatEquals(a,b,tol) (fabs(a - b) < tol)

/* data structures */

typedef struct {
  pfloat * a; //sub-diagonal
  pfloat * b; //diagonal
  pfloat * c; //super-diagonal
} triMat;

// using accumulation fp type
typedef struct {
  afloat phi_left;
  afloat J_left;
  afloat E_left;
  afloat phi_right;
  afloat J_right;
  afloat E_right;
  afloat * phi_n;
  afloat * phi_n2;
  afloat * E_n;
  void * buffer;
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
  pfloat * q0;
  pfloat eps;
  pfloat q0_lo;
  pfloat D;
  long long NP_tot;
  long long NPc;
} data  __attribute__ ((aligned (16)));

int run(int);
void sim_param_print(struct data);
void lo_solver(struct data, pfloat, pfloat, pfloat, pfloat, pfloat, pfloat, pfloat *, int, pfloat*,triMat,pfloat*);
void collision_and_tally(struct data, pfloat *, colData*, int,int,dsfmt_t*,int,int);
void triSolve(int, pfloat *, pfloat *, pfloat *, pfloat *, pfloat *);
pfloat absdistcalc(int, int, int, pfloat, pfloat, pfloat, int);
void mean_std_calc(afloat *, afloat *, unsigned long int, int, int, pfloat, meanDev*);
void gen_particles_and_tally(struct data, pfloat *, colData *,int,int,dsfmt_t*);
#endif /* __QDAMC_H__ */
