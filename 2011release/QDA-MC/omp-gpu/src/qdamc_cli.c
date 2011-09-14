/*
 * qdamc_cli.c - QDA-MC Main Application
 */

/*
 * TODO: Get command line working.
 */ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include "qdamc_cli.h"
#include "ecodes.h"								// error code definitions
#include "qda-mc/qdamc.h"	    	  // particle transport simulation
#include "getopt/getopt.h"				// for command-line option parsing

/* For Command Line */
options opt;

/* start of program execution */
int main(int argc, char* argv[]){

  int rank;
  rank = 0;

  /* parse command-line options */
  getopts(argc, argv);

  /* if verbose mode, print out program information as a header */
  if(rank == 0 && !opt.silent){
    printf("------------------------------------\n");
    show_version();
    printf("------------------------------------\n");
  }

  int retval = run(rank);
  /* run the simulation until end of program*/

  return retval;
}

/* parse command-line options */
void getopts(int argc, char* argv[]){
  /* local variable declarations for getopt*/
  int optch = 0;						/* for parsing */
  int optdex = 0;						/* to store option indices */
  static char stropts[] = "hVsp:l:c:t:T:r:i:G:";
  static struct option long_options[] =
    {
      {"help",		no_argument,       	 0, 'h'},
      {"version",	no_argument,       	 0, 'v'},
      {"silent",  no_argument,         0, 's'},
      {"particles", required_argument, 0, 'p'},
      {"syslength", required_argument, 0, 'l'},
      {"numcells", required_argument, 0, 'c'},
      {"tolerance", required_argument, 0, 't'},
      {"tolstd", required_argument, 0, 'T'},
      {"runwidth", required_argument, 0, 'r'},
      {"numiters", required_argument, 0, 'i'},
      {"timing", required_argument, 0, 'G'},
      {"sigs", required_argument,0,'q'},
      {"sigt", required_argument,0,'w'},
      {0, 0, 0, 0}
    };

  /* Set default options, if any */
  memset(&opt, 0, sizeof(options));
  opt.silent = false;		/* default choice */
  opt.NP = DEFAULT_PARTICLES;
  opt.sysLen = DEFAULT_SYSLEN;
  opt.numCells = DEFAULT_NUMCELLS;
  opt.tol = DEFAULT_TOL;
  opt.tol_std = DEFAULT_TOLstd;
  opt.runWidth = DEFAULT_RUNWIDTH;
  opt.numIters = DEFAULT_NUMITERS;

  opt.eps = DEFAULT_eps;                                      
  opt.sig_t = DEFAULT_sig_t;                                       	
  opt.Jp_l = DEFAULT_Jp_l;                                         
  opt.q0_lo = DEFAULT_q0_lo;
  opt.lbc_flag = DEFAULT_lbc_flag;

  opt.sig_s = DEFAULT_sig_s;                            	
  opt.D4 = DEFAULT_D4;               
  opt.dx = DEFAULT_dx;                         
  opt.NPc = DEFAULT_NPc;

  /* Parse command line arguments */
  extern char *optarg;
  while ((optch = getopt_long (argc, argv, stropts, long_options, &optdex)) != -1)
    switch (optch){
    case 0:
      break;
    case 'h':
      show_usage();
      exit(EXIT_SUCCESS);
    case 'v':
      show_version();
      exit(EXIT_SUCCESS);
    case 's':
      opt.silent = true;            	// suppress all unnecessary output
      break;
    case 'p': 
      opt.NP = atoi(optarg);					// num particles
      if(opt.NP > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'c': 
      opt.numCells = atoi(optarg);		// num cells
      if(opt.numCells > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'r': 
      opt.runWidth = atoi(optarg);		// run width
      if(opt.runWidth >= 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'l': 
      opt.sysLen = atof(optarg);			// system length
      if(opt.sysLen > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 't': 
      opt.tol = atof(optarg);					// tolerance
      if(opt.tol > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'T': 
      opt.tol_std = atof(optarg);			// tolerance (stdev)
      if(opt.tol_std > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'i': 
      opt.numIters = atoi(optarg);  	// num iterations
      if(opt.numIters > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'G': 
      opt.timing = atoi(optarg);  	  // report timing over N iterations
      if(opt.timing > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'q':
      opt.sig_s = atof(optarg);
      if( opt.sig_s >= 0.0 ) break;
      else exit(BAD_COMMAND_LINE);	
    case 'w':
      opt.sig_t = atof(optarg);
      if( opt.sig_t >= 0.0 ) break;
      else exit(BAD_COMMAND_LINE);	
    case '?':
      if (isprint (optopt))
	fprintf (stderr, "Unknown option `-%c'.\n", optopt);
      else
	fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
      show_usage();
      exit(BAD_COMMAND_LINE);
    default:
      fprintf (stderr, "Internal CLI parsing error. Unrecognized internal representation.\n");
      assert(false);		/* default should never be reached */
    }

  /* Remaining arguments -> error! */
  /*if(optind >= argc){
    fprintf(stderr, "Too few command-line arguments!\n");
    show_usage();
    exit(BAD_COMMAND_LINE);
    }*/
  if(optind < argc){
    fprintf(stderr, "Too many command-line arguments!\n");
    show_usage();
    exit(BAD_COMMAND_LINE);
  }
}

/* show program version and author information */
void show_version(void){
  printf("%s ( %s ) v. %s\n", PROGRAM, PACKAGE, VERSION);
  printf("%s (c) %s\n", AUTHOR, YEAR);
}

/* show command line syntax */
void show_usage(void){
  printf("\nUsage : %s [-hVp:]\n", PACKAGE);
  printf("  --help                display this help and exit\n");
  printf("  --version             output version information and exit\n");
  printf("  --particles           the number of particles to simulate\n");
  printf("  --numcells            the number of discrete simulation cells\n");
  printf("  --runwidth            run width for averaging\n");
  printf("  --syslength           the length of the system\n");
  printf("  --tolerance           end-condition tolerance\n");
  printf("  --tolstd              end-condition tolerance (standard deviation)\n");
  printf("  --numiters            number of iterations to run if stopping condition is unsatisfied\n");
  printf("  --runwidth            number of iterations to skip\n");
  printf("  --sigs            	  scattering cross section\n");
  printf("  --sigt            	  total cross section\n");
  printf("\n");
}
