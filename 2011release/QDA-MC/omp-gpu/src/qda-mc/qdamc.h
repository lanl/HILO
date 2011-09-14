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
#include <sys/stat.h>
#include <assert.h>
#include <malloc.h>
#include "../qdamc_cli.h"
#include <CL/cl.h>

/* constants */
#define FLOAT_TOL 0.000001
#define DEFAULT_Q0 1.0                                     // is malloc'ed and applied to every element

/* macros */
#define filter(l,m,r) (.25f*l + .5f*m +.25*r)
#define floatEquals(a,b,tol) (fabs(a - b) < tol)

int NP;				// Number of particles
pfloat sysLeng;		// Length of system
int numCells;		// Number of cells in system
pfloat tol;			// Relative difference tolerance for convergence
pfloat tol_std;		// Tolerance for standard deviation calculations
pfloat fig_upd;		// TODO
int runningWidth;	// Number of iterations to keep for averaging

/* data structures */
typedef struct {
    int iter_avg;    								//iteration for averaging
    afloat phi_left_tot;  							//tally for average phi
    afloat phi_right_tot;     					//tally for average phi
    afloat J_left_tot;                  //tally for average J
    afloat J_right_tot;        					//tally for average J
   	afloat E_left_tot;  								//tally for average E
    afloat E_right_tot;  								//tally for average E
    pfloat phi_left_avg; 
    pfloat phi_right_avg; 
    pfloat phi_left_avg_old;
    pfloat phi_right_avg_old; 
    pfloat J_left_avg; 
    pfloat J_right_avg;
    pfloat E_left_avg; 
    pfloat E_right_avg;
} avedat;

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
    pfloat * x0;
    pfloat * mu0;
    pfloat * wx;
    pfloat * xf;
    int * cell0;
    int * cellf;
} colData_additional;

typedef struct {
    int lbc_NP;
    pfloat * lbc_mu;
    int * lbc_wp;
} lbcData;

typedef struct {
    pfloat * xf;
    int * cell0;
    int * cellf;
} xmcData;

typedef struct {
    pfloat mean;
    pfloat stdDev;
} meanDev __attribute__ ((aligned (16)));

struct simData {
    int NP;
    int NPstep;
    int steps;
    pfloat lx;
    int nx;
    pfloat dx;
	pfloat dx2;				// cellsize^2
	pfloat dx_recip;
    pfloat sig_t;
	pfloat sig_t_recip;
    pfloat sig_s;
    pfloat * q0;
    pfloat Jp_l;
    pfloat eps;
    pfloat q0_lo;
    pfloat D;
    unsigned long NP_tot;
    int NPc;
    int lbc_flag;
} __attribute__ ((aligned (16)));

struct gpuData
{
	cl_float sig_t;
	cl_float dx;
};

struct oclData {
    cl_context context;
    cl_command_queue queue;
    cl_platform_id * platforms;
    cl_device_id * devices;
    cl_program program;
    cl_kernel warmup;
    cl_kernel pGenerate2;
    cl_mem ranBuf;
    cl_mem clData;
    cl_mem clMu0;
    cl_mem clX0;
    cl_mem clXf;
    int devID;
};

int run(int);
void calcStartEnd(int*, int*, int, int, int);
struct oclData oclInit(struct simData *, struct gpuData *);
void sim_param_print(struct simData);
void lo_solver(struct simData, pfloat, pfloat, pfloat, pfloat, pfloat, pfloat, pfloat *, int, pfloat*,triMat,pfloat*);
void collision_and_tally(struct simData, pfloat *, colData*, int,int,int,float*,float*,float*,int, int, int);
void triSolve(int, pfloat *, pfloat *, pfloat *, pfloat *, pfloat *);
pfloat absdistcalc(int, int, int, pfloat, pfloat, pfloat, int);
void f_filter(void *, void*, int, int, int, int *);
void mean_std_calc(afloat *, afloat *, unsigned long int, int, int, pfloat, meanDev*);
void gen_particles_and_tally_ompgpu(struct simData,pfloat*,colData*,int,int,float*,float*,float*,int);
int getDevID(char * desired, cl_device_id * devices, int numDevices);
#endif /* __QDAMC_H__ */
