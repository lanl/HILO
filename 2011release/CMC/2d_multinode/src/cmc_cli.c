/*
 * cmc_cli.c - C-MC Main Application
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
#include "cmc_cli.h"
#include "ecodes.h"								// error code definitions
#include "cmc/cmc.h"	    	  // particle transport simulation
#include "getopt/getopt.h"				// for command-line option parsing
#ifdef DO_MPI
#include "mpi.h"
#endif

/* For Command Line */
options opt;

/* start of program execution */
int main(int argc, char* argv[]){
  
  int rank;

#ifdef DO_MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#else
  rank = 0;
#endif

  /* parse command-line options */
  getopts(argc, argv);

  /* if verbose mode, print out program information as a header */
  if(rank == 0 && !opt.silent){
    printf("------------------------------------\n");
    show_version();
    printf("------------------------------------\n");
  }
  int retval = run(rank);
  
#ifdef DO_MPI
  MPI_Finalize();
#endif

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
      {"syslengthx", required_argument, 0, 'l'},
      {"syslengthy", required_argument, 0 , 'm'},
      {"numcellsx", required_argument, 0, 'c'},
      {"numcellsy", required_argument, 0, 'd'},
      {"tolerance", required_argument, 0, 't'},
      {"tolstd", required_argument, 0, 'T'},
      {"numiters", required_argument, 0, 'i'},
      {"timing", required_argument, 0, 'G'},
      {"sigs", required_argument,0,'q'},
      {"sigt", required_argument,0,'w'},
      {"reflectx0", required_argument,0,'e'},
      {"reflectxN", required_argument,0,'f'},
      {"reflecty0", required_argument,0,'g'},
      {"reflectyN", required_argument,0,'j'},
      {0, 0, 0, 0}
    };

  /* Set default options, if any */
  memset(&opt, 0, sizeof(options));
  opt.silent = false;		/* default choice */
  opt.NP = DEFAULT_PARTICLES;
  opt.sysLenx = DEFAULT_SYSLENX;
  opt.sysLeny = DEFAULT_SYSLENY;
  opt.numCellsx = DEFAULT_NUMCELLSX;
  opt.numCellsy = DEFAULT_NUMCELLSY;
  opt.tol = DEFAULT_TOL;
  opt.tol_std = DEFAULT_TOLstd;
  opt.numIters = DEFAULT_NUMITERS;

  opt.eps = DEFAULT_eps;                                      
  opt.sig_t = DEFAULT_sig_t;                                       	
  opt.sig_s = DEFAULT_sig_s;                            	
  opt.dx = DEFAULT_dx;                         
  opt.NPc = DEFAULT_NPc;

  opt.reflectx0 = DEFAULT_reflectx0;
  opt.reflectxN = DEFAULT_reflectxN;
  opt.reflecty0 = DEFAULT_reflecty0;
  opt.reflectyN = DEFAULT_reflectyN;


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
      opt.numCellsx = atoi(optarg);		// num cells
      if(opt.numCellsx > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'd': 
      opt.numCellsy = atoi(optarg);		// num cells
      if(opt.numCellsy > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'l': 
      opt.sysLenx = atof(optarg);			// system length
      if(opt.sysLenx > 0) break;
      else exit(BAD_COMMAND_LINE);
    case 'm': 
      opt.sysLeny = atof(optarg);			// system length
      if(opt.sysLeny > 0) break;
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
    case 'e':
      opt.reflectx0 = atoi(optarg);
      if( (opt.reflectx0 == 0) || ( opt.reflectx0 == 1 ) ) break;
      else exit(BAD_COMMAND_LINE);
    case 'f':
      opt.reflectxN = atoi(optarg);
      if( (opt.reflectxN == 0) || ( opt.reflectxN == 1 ) ) break;
      else exit(BAD_COMMAND_LINE);
    case 'g':
      opt.reflecty0 = atoi(optarg);
      if( (opt.reflecty0 == 0) || ( opt.reflecty0 == 1 ) ) break;
      else exit(BAD_COMMAND_LINE);
    case 'j':
      opt.reflectyN = atoi(optarg);
      if( (opt.reflectyN == 0) || ( opt.reflectyN == 1 ) ) break;
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
  printf("  --numcellsx           the number of discrete simulation cells in x direction\n");
  printf("  --numcellsy           the number of discrete simulation cells in y direction\n");
  printf("  --syslengthx          the length of the system along x direction\n");
  printf("  --syslengthy          the length of the system along y direction\n");
  printf("  --tolerance           end-condition tolerance\n");
  printf("  --tolstd              end-condition tolerance (standard deviation)\n");
  printf("  --numiters            number of iterations to run if stopping condition is unsatisfied\n");
  printf("\n");
}
