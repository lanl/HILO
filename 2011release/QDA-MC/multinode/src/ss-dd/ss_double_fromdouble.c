/*
 * ss_double_fromdouble.c - Drop in "double" replacement for Single-Single (where singles are actually doubles)
 *
 */

#include "ss_double_fromdouble.h"
#include <math.h>

/* 
 * TYPE CONVERSIONS
 */

/* single-single (double wrapper) */

ss ss_init(){
  /* initial ss */
  ss retval;
  retval.num = 0.0;
  return retval;
}

ss num_num_to_ss(double hi, double lo){ 
  /* double/double to ss */
  ss retval;
  retval.num = hi; 
  return retval;
}

ss num_to_ss(double h){ 
  /* double to ss */
  ss retval;
  retval.num = h; 
  return retval;
}

ss i_to_ss(int h){
  /* int to ss */
  ss retval;
  retval.num = (double)h;
  return retval;
}

/* add */

/* ss(C) = double(a) + double(b) */
ss num_num_add_ss(const double a, const double b){
  return num_to_ss(a + b);
}


/* ss(C) = ss(a) + double(b) */
ss ss_num_add_ss(const ss a, const double b){
  return num_to_ss(a.num + b);
}

/* ss(C) = double(a) + ss(b) */
ss num_ss_add_ss(const double a, const ss b){
  /* perform ss arithmetic */
  return num_to_ss(a + b.num);
}

/* ss(C) = ss(a) + ss(b) */
ss ss_ss_add_ss(const ss a, const ss b){
  return num_to_ss(a.num + b.num);
}

/* sub */

/* ss(C) = double(a) - double(b) */
ss num_num_sub_ss(const double a, const double b){
  return num_to_ss(a - b);
}


/* ss(C) = ss(a) - double(b) */
ss ss_num_sub_ss(const ss a, const double b){
  return num_to_ss(a.num - b);
}

/* ss(C) = double(a) - ss(b) */
ss num_ss_sub_ss(const double a, const ss b){
  /* perform ss arithmetic */
  return num_to_ss(a - b.num);
}

/* ss(C) = ss(a) - ss(b) */
ss ss_ss_sub_ss(const ss a, const ss b){
  return num_to_ss(a.num - b.num);
}

/* mul */

/* ss(C) = double(a) * double(b) */
ss num_num_mul_ss(const double a, const double b){
  return num_to_ss(a * b);
}

/* ss(C) = ss(a) * double(b) */
ss ss_num_mul_ss(const ss a, const double b){
  return num_to_ss(a.num * b);
}

/* ss(C) = double(a) * ss(b) */
ss num_ss_mul_ss(const double a, const ss b){
  /* perform ss arithmetic */
  return num_to_ss(a * b.num);
}

/* ss(C) = ss(a) * ss(b) */
ss ss_ss_mul_ss(const ss a, const ss b){
  return num_to_ss(a.num * b.num);
}


/* div */

/* ss(C) = double(a) / double(b) */
ss num_num_div_ss(const double a, const double b){
  return num_to_ss(a / b);
}

/* ss(C) = ss(a) / double(b) */
ss ss_num_div_ss(const ss a, const double b){
  return num_to_ss(a.num / b);
}

/* ss(C) = double(a) / ss(b) */
ss num_ss_div_ss(const double a, const ss b){
  /* perform ss arithmetic */
  return num_to_ss(a / b.num);
}

/* ss(C) = ss(a) / ss(b) */
ss ss_ss_div_ss(const ss a, const ss b){
  return num_to_ss(a.num / b.num);
}

/* reciprocal */

/* ss(B) = 1.0 / num(a) */
ss num_recip_ss(double a){
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

  return retval;
}


/* misc math.h */

/* is zero */
int is_zero_ss(const ss a){
  return (a.num == 0.0);
}

/* is positive */
int is_positive_ss(const ss a){
  return (a.num > 0.0);
}

/* is negative */
int is_negative_ss(const ss a){
  return (a.num < 0.0);
}

/* floor ss to nearest int */
ss floor_ss(const ss a){
  double hi = floor(a.num);
  return num_to_ss(hi);
}

/* ceil ss to nearest int */
ss ceil_ss(const ss a){
  double hi = ceil(a.num);
  return num_to_ss(hi);
}
