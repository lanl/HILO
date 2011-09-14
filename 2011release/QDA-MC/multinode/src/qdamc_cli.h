/*
 * qdamc_cli.h - Command Line Interface Header File
 */

#ifndef __QDAMC_CLI_H__
#define __QDAMC_CLI_H__

/*
 * program info
 */
#define PROGRAM "1-D Particle Transport Simulation"
#define PACKAGE "qdamc"
#define VERSION "0.03"
#define AUTHOR "LANL Co-Design School"
#define YEAR "2011"

/* Default Values */
#define DEFAULT_PARTICLES 10000000
#define DEFAULT_SYSLEN 10
#define DEFAULT_NUMCELLS 100
#define DEFAULT_TOL 0.001
#define DEFAULT_TOLstd 0.0005
#define DEFAULT_RUNWIDTH 5
#define DEFAULT_NUMITERS 1000

#define DEFAULT_eps 0.1f;                                      
#define DEFAULT_sig_t 1.0f;                                        	
#define DEFAULT_q0_lo 1.0f;

#define DEFAULT_D (opt.sig_s / (3 * opt.sig_t * opt.sig_t));
#define DEFAULT_D4 4.0f * DEFAULT_D;                                        
#define DEFAULT_sig_s 0.99f * opt.sig_t;                               	
#define DEFAULT_dx opt.sysLen / opt.numCells;                         
#define DEFAULT_NPc rint(opt.NP * 1.0f / opt.numCells);    // depends on math.h							

/* Global Typedefs */

/* for determining FP precision */
#define true 1
#define false 0
// general floating-point format
typedef double pfloat;
// floating-point format for accumulation
typedef double afloat;

/* CLI function prototypes */
void getopts(int, char*[]);
void show_version(void);
void show_usage(void);

/*
 * Command Line Argument Parsing
 */
typedef struct
{
  short silent; 
  int timing;       // Report timing information with N iterations
  long long  NP;    // Number of particles
  pfloat sysLen;    // Length of system
  int numCells;	    // Number of cells in system
  pfloat tol;	    // Relative difference tolerance for convergence
  pfloat tol_std;   // Tolerance for standard deviation calculations
  int runWidth;	    // Number of iterations to keep for averaging
  int numIters;     // Number of iterations to complete if stopping condition is never reached

  pfloat dx;        // Cell Width
  pfloat sig_t;     // Total Collision Cross-Section
  pfloat sig_s;     // Scattering Collision Cross-Section
  pfloat eps;       // Threshold Angle for Tallying
  pfloat q0_lo;     // Source Term
  pfloat D4;        // 4*Diffusion Coefficient
  long long NPc;          // Number of Particles per Cell TODO: USED ONCE
} options;

extern options opt;

#endif /* __QDAMC_CLI_H__ */
