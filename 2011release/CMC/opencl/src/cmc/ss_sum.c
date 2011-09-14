#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "ss.h"

/* parameters */

#define NUMITERS 50000 //60000

#define INITIAL_NUMBER 0.0

#define LOWER_RAND_BOUND 0.0
/* upper bound for first iteration */ 
#define FIRST_UPPER_RAND_BOUND 0.00000001
/* dynamically adjusts the upper bound by dsum >> UPPER_RAND_DIVIDER */ 
#define UPPER_RAND_DIVIDER -9

/* datatypes */

typedef struct{
  float srandnum;
  double  drandnum;
} randnum;

/* helper functions */

inline randnum gen_randnum(double lower, double upper){
  /* generates a multiprecision random number */
  int randint = rand();
  randnum retval;
  retval.srandnum = (float)(((upper-lower)*((double)randint/RAND_MAX))+lower);
  retval.drandnum = ((upper-lower)*((double)randint/RAND_MAX))+lower;

  return retval;
}

inline float single_sum(float entry, float sum){
  /* single precision, regular summation */
  return sum + entry; 
}

inline double double_sum(double entry, double sum){
  /* double precision, regular summation */
  return sum + entry; 
}

inline double percent_diff(double actual, float obtained){
  return ((actual-(double)obtained)/actual)*100.0;
}


/* ARITHMETIC is included in ss.h */

/* APPLICATION */

int main(){

  /* initialize normal sum, kahan sum, double precision for check */
  float fsum = INITIAL_NUMBER;
  double dsum = INITIAL_NUMBER;
  ss ssum;
  ssum.num = INITIAL_NUMBER; ssum.err = 0.0;
  double upper_bound;

  /* initialize RNG, numbers to sum */
  unsigned long long int loop;
  randnum curr_num;
  srand(time(NULL));
  
  /* do all types of sums (overflow unlikely) */
  for(loop = 0; loop < NUMITERS; loop++){
    /* ldexp is like >> for floating-point numbers */
    upper_bound = (loop > 0) ? ldexp(dsum, UPPER_RAND_DIVIDER) : FIRST_UPPER_RAND_BOUND;
    curr_num = gen_randnum(LOWER_RAND_BOUND, upper_bound);

    /* single precision, double precision */
    fsum = single_sum(curr_num.srandnum, fsum);
    dsum = double_sum(curr_num.drandnum, dsum);

    /* s-s sum */
    ssum = num_ss_add(curr_num.srandnum, ssum);   

#if MONITOR_KAHAN > 0
    if(loop % KAHAN_MONITOR_INTERVAL == 0){
      printf("%d\t%.65e\t%.65e\t%.65e\n", loop, ksum.sum, ksum.correction, percent_diff(dsum, ksum.sum));
    }
#endif
  }

  /* print out results */
  printf("Experiment with Proportional Numbers:\n");
  printf("Double precision: %e\n", dsum);
  printf("Single precision: %e (%e %% difference)\n", fsum, percent_diff(dsum, fsum));
  printf("S-S sum: %e (%e %% difference)\n", ssum.num, percent_diff(dsum, ssum.num));

  return 0;
}
