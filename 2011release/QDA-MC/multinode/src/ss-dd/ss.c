/*
 * ss.c - Typedefs / Structs for SS
 *
 */

#include "ss.h"
#include "util_ss.h"
#include <math.h>

/* 
 * TYPE CONVERSIONS
 */

/* single-single */

ss ss_init(){
  /* initial ss */
  ss retval;
  retval.num = 0.0; 
  retval.err = 0.0; 
  return retval;
}

ss num_num_to_ss(float hi, float lo){ 
  /* float/float to ss */
  ss retval;
  retval.num = hi; 
  retval.err = lo;
  return retval;
}

ss num_to_ss(float h){ 
  /* float to ss */
  ss retval;
  retval.num = h; 
  retval.err = 0.0; 
  return retval;
}

ss i_to_ss(int h){
  /* int to ss */
  ss retval;
  retval.num = (float)h;
  retval.err = 0.0;
  return retval;
}

/* 
 * BASIC ARITHMETIC 
 */

/* add */

/* ss(C) = float(a) + float(b) */
ss num_num_add_ss(const float a, const float b){
  return ss_two_sum(a, b);
}


/* ss(C) = ss(a) + float(b) */
ss ss_num_add_ss(const ss a, const float b){
  /* containers and temporary variables */
  ss retval;

  /* perform ss arithmetic */
  retval = ss_two_sum(a.num, b);
  retval.err += a.err;
  retval = ss_quick_two_sum(retval.num, retval.err);
  
  return retval;
}

/* ss(C) = float(a) + ss(b) */
ss num_ss_add_ss(const float a, const ss b){
  /* perform ss arithmetic */
  return ss_num_add_ss(b, a);
}

/* ss(C) = ss(a) + ss(b) */
ss ss_ss_add_ss(const ss a, const ss b){
  /* containers and temporary variables */
  ss temp, retval;
  
  /* perform ss arithmetic */
  retval = ss_two_sum(a.num, b.num);
  temp = ss_two_sum(a.err, b.err);   
  retval.err += temp.num;
  retval = ss_quick_two_sum(retval.num, retval.err);
  retval.err += temp.err;
  retval = ss_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* sub */

/* ss(C) = num(a) - num(b) */
ss num_num_sub_ss(const float a, const float b){
 return ss_two_diff(a, b);
}

/* ss(C) = ss(a) - num(b) */
ss ss_num_sub_ss(const ss a, const float b){
  /* containers and temporary variables */
  ss retval;

  /* perform ss arithmetic */
  retval = ss_two_diff(a.num, b);
  retval.err += a.err;
  retval = ss_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* ss(C) = num(a) - ss(b) */
ss num_ss_sub_ss(const float a, const ss b){
  /* containers and temporary variables */
  ss retval;

  /* perform ss arithmetic */
  retval = ss_two_diff(a, b.num);
  retval.err -= b.err;
  retval = ss_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* ss(C) = ss(a) - ss(b) */
ss ss_ss_sub_ss(const ss a, const ss b){
  /* containers and temporary variables */
  ss retval;
  
  /* perform ss arithmetic (QD_IEEE_ADD variant) */
  retval = ss_two_diff(a.num, b.num);
  retval.err += a.err;
  retval.err -= b.err;
  retval = ss_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* mul */

/* ss(C) = num(a) * num(b) */
ss num_num_mul_ss(const float a, const float b){
  return ss_two_prod(a, b);
}

/* ss(C) = ss(a) * num(b) */
ss ss_num_mul_ss(const ss a, const float b){
  /* containers and temporary variables */
  ss retval;

  /* perform ss arithmetic */
  retval = ss_two_prod(a.num, b); // p2 is retval.err
  retval.err += (a.err * b);
  retval = ss_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* ss(C) = num(a) * ss(b) */
ss num_ss_mul_ss(const float a, const ss b){
  /* perform ss arithmetic */
  return ss_num_mul_ss(b, a);
}

/* ss(C) = ss(a) * ss(b) */
ss ss_ss_mul_ss(const ss a, const ss b){
  /* containers and temporary variables */
  ss retval;
  
  /* perform ss arithmetic */
  retval = ss_two_prod(a.num, b.num);
  retval.err += (a.num * b.err + a.err * b.num);
  retval = ss_quick_two_sum(retval.num, retval.err);
  
  return retval;
}

/* div */

/* ss(C) = num(a) / num(b) */
ss num_num_div_ss(const float a, const float b){
  /* containers and temporary variables */
  ss q, p, retval;

  /* perform ss arithmetic */
  q = num_to_ss(a / b);
  /* Compute  a - q.num * b */
  p = ss_two_prod(q.num, b);
  retval = ss_two_diff(a, p.num);
  retval.err -= p.err;
  /* get next approximation */
  q.err = (retval.num + retval.err) / b;
  retval = ss_quick_two_sum(q.num, q.err);

  return retval;
}

/* ss(C) = ss(a) / num(b) */
ss ss_num_div_ss(const ss a, const float b){
  /* containers and temporary variables */
  ss q, p, retval;
  
  /* perform ss arithmetic */
  q = num_to_ss(a.num / b);   /* approximate quotient. */

  /* Compute  this - q.num * d */
  p = ss_two_prod(q.num, b);
  retval = ss_two_diff(a.num, p.num);
  retval.err += a.err;
  retval.err -= p.err;
  
  /* get next approximation. */
  q.err = (retval.num + retval.err) / b;

  /* renormalize */
  retval = ss_quick_two_sum(q.num, q.err);

  return retval;
}

/* ss(C) = ss(a) / ss(b) */
ss ss_ss_div_ss(const ss a, const ss b){
  /* containers and temporary variables */
  ss q, p, retval;

  /* perform ss arithmetic */
  q = num_to_ss(a.num / b.num);  /* approximate quotient */
  retval = ss_ss_sub_ss(a, num_ss_mul_ss(q.num, b)); 
  q.err = retval.num / b.num;
  retval = ss_ss_sub_ss(retval, num_ss_mul_ss(q.err, b));
  p.num = retval.num / b.num; 
  q = ss_quick_two_sum(q.num, q.err);
  retval = ss_num_add_ss(q, p.num);

  return retval;
}

/* ss(C) = num(a) / ss(b) */
ss num_ss_div_ss(float a, const ss b){
  /* perform ss arithmetic */
  return ss_ss_div_ss(num_to_ss(a), b);
}

/* reciprocal */

/* ss(B) = 1.0 / num(a) */
ss num_recip_ss(float a){
  /* perform ss arithmetic */
  return num_num_div_ss(1.0, a);
}

/* ss(B) = 1.0 / ss(a) */
ss ss_recip_ss(ss a){
  /* perform ss arithmetic */
  return num_ss_div_ss(1.0, a);
}

/* copy */

ss ss_copy_ss(const ss a){
  /* containers and temporary variables */
  ss retval;

  /* copy the input value */
  retval.num = a.num;
  retval.err = a.err;

  return retval;
}

/* misc math.h */

/* is zero */
int is_zero_ss(const ss a){
  return (a.num == 0.0 && a.err == 0);
}

/* is positive */
int is_positive_ss(const ss a){
  return (a.num > 0.0 && a.err > 0) || (a.num > 0 && a.err < a.num);
}

/* is negative */
int is_negative_ss(const ss a){
  return (a.num < 0.0 && a.err < 0) || (a.num < 0 && a.err < a.num);
}

/* floor ss to nearest int */
ss floor_ss(const ss a){
  float hi = floorf(a.num);
  float lo = 0.0;

  if(hi == a.num){
    /* high word is integer already.  Round the low word. */
    lo = floorf(a.err);
    return ss_quick_two_sum(hi, lo);
  }
  else{
    /* can be rounded without uncertainty */
    return num_num_to_ss(hi, lo);
  }
}

/* ceil ss to nearest int */
ss ceil_ss(const ss a){
  float hi = ceilf(a.num);
  float lo = 0.0;

  if(hi == a.num){
    /* high word is integer already.  Round the low word. */
    lo = ceilf(a.err);
    return ss_quick_two_sum(hi, lo);
  }
  else{
    /* can be rounded without uncertainty */
    return num_num_to_ss(hi, lo);
  }
}
