/*
 * cmc_cli.h - Command Line Interface Header File
 */

#ifndef __CMC_CLI_H__
#define __CMC_CLI_H__

/*
 * program info
 */
#define PROGRAM "1-D Particle Transport Simulation"
#define PACKAGE "cmc"
#define VERSION "0.03"
#define AUTHOR "LANL Co-Design School"
#define YEAR "2011"

/* Default Values */
#define DEFAULT_PARTICLES 10000000
#define DEFAULT_SYSLENX 10
#define DEFAULT_SYSLENY 10
#define DEFAULT_NUMCELLSX 100
#define DEFAULT_NUMCELLSY 100
#define DEFAULT_TOL 0.001
#define DEFAULT_TOLstd 0.0005
#define DEFAULT_NUMITERS 1000

#define DEFAULT_eps 0.1f                                      
#define DEFAULT_sig_t 1.0f                                        	
#define DEFAULT_sig_s 0.99f * opt.sig_t                               	
#define DEFAULT_dx opt.sysLenx / opt.numCellsx
#define DEFAULT_dy opt.sysLeny / opt.numCellsy
#define DEFAULT_NPc (long long)(opt.NP * 1.0f / (opt.numCellsx * opt.numCellsy))    // depends on math.h							

#define DEFAULT_reflectx0 0
#define DEFAULT_reflectxN 0
#define DEFAULT_reflecty0 1
#define DEFAULT_reflectyN 1


/* Global Typedefs */
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
  long long NP;						// Number of particles
  pfloat sysLenx;		// Length of system
  pfloat sysLeny;
  int numCellsx;			// Number of cells in system
  int numCellsy;
  pfloat tol;				// Relative difference tolerance for convergence
  pfloat tol_std;		// Tolerance for standard deviation calculations
  int numIters;     // Number of iterations to complete if stopping condition is never reached
  int reflectx0;
  int reflecty0;
  int reflectxN;
  int reflectyN;
  pfloat dx;        // Cell Width
  pfloat dy;
  pfloat sig_t;     // Total Collision Cross-Section
  pfloat sig_s;     // Scattering Collision Cross-Section
  pfloat eps;       // Threshold Angle for Tallying
  pfloat q0_avg;     // Source Term
  pfloat scatter_prog; //Scattering probability
  long long NPc;          // Number of Particles per Cell TODO: USED ONCE
} options;

extern options opt;

#endif /* __CMC_CLI_H__ */
