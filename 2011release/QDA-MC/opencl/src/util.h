/*
 * util.h - Math helper fuctions
 *
 * Code taken from QD 2.3.11, Copyright (c) 2000-2001 DoE
 */

#ifndef _UTIL_H
#define _UTIL_H

#include "types.h" /* ss and dd */

/* ss */

/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
inline ss ss_quick_two_sum(float a, float b){
  float s;
  s = a + b;
  return num_num_to_ss(s, b - (s - a));
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
inline ss ss_quick_two_diff(float a, float b){
  float s;
  s = a - b;
  return num_num_to_ss(s, (a - s) - b);
}

/* Computes fl(a+b) and err(a+b).  */
inline ss ss_two_sum(float a, float b){
  ss retval;
  float bb;

  retval.num = a + b;
  bb = retval.num - a;
  retval.err = (a - (retval.num - bb)) + (b - bb);
  
  return retval;
}

/* Computes fl(a-b) and err(a-b).  */
inline ss ss_two_diff(float a, float b){
  ss retval;
  float bb;

  retval.num = a - b;
  bb = retval.num - a;
  retval.err = (a - (retval.num - bb)) - (b + bb);
  
  return retval;
}

#define _SS_SPLITTER 9.0                      // = 2^3 + 1
#define _SS_SPLIT_THRESH 2.12676479325587e+37 // = 2^124

/* Computes high word and lo word of a        */
/* uses ss.num as hi word, ss.err as low word */
inline ss ss_split(double a){
  ss retval;
  double temp;

  if (a > _SS_SPLIT_THRESH || a < -_SS_SPLIT_THRESH){
    a *= 0.0625;                 // 2^-4
    temp = _SS_SPLITTER * a;
    retval.num = temp - (temp - a);
    retval.err = a - retval.num;
    retval.num *= 16.0;          // 2^4
    retval.err *= 16.0;          // 2^4
  } else {
    temp = _SS_SPLITTER * a;
    retval.num = temp - (temp - a);
    retval.err = a - retval.num;
  }

  return retval;
}

/* Computes fl(a*b) and err(a*b). */
inline ss ss_two_prod(float a, float b){
  ss retval, split_a, split_b;

  retval.num = a * b;
  split_a = ss_split(a);
  split_b = ss_split(b);
  retval.err = ((split_a.num * split_b.num - retval.num) + split_a.num * split_b.err + split_a.err * split_b.num) + split_a.err * split_b.err;

  return retval;
}

/* dd */

/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
inline dd dd_quick_two_sum(double a, double b){
  double s;
  s = a + b;
  return num_num_to_dd(s, b - (s - a));
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
inline dd dd_quick_two_diff(double a, double b){
  double s;
  s = a - b;
  return num_num_to_dd(s, (a - s) - b);
}

/* Computes fl(a+b) and err(a+b).  */
inline dd dd_two_sum(double a, double b){
  dd retval;
  double bb;

  retval.num = a + b;
  bb = retval.num - a;
  retval.err = (a - (retval.num - bb)) + (b - bb);
  
  return retval;
}

/* Computes fl(a-b) and err(a-b).  */
inline dd dd_two_diff(double a, double b){
  dd retval;
  double bb;

  retval.num = a - b;
  bb = retval.num - a;
  retval.err = (a - (retval.num - bb)) - (b + bb);
  
  return retval;
}

#define _DD_SPLITTER 134217729.0               // = 2^27 + 1
#define _DD_SPLIT_THRESH 6.69692879491417e+299 // = 2^996

/* Computes high word and lo word of a */
/* uses dd.num as hi word, dd.err as low word */
inline dd dd_split(double a){
  dd retval;
  double temp;

  if (a > _DD_SPLIT_THRESH || a < -_DD_SPLIT_THRESH){
    a *= 3.7252902984619140625e-09;      // 2^-28
    temp = _DD_SPLITTER * a;
    retval.num = temp - (temp - a);
    retval.err = a - retval.num;
    retval.num *= 268435456.0;          // 2^28
    retval.err *= 268435456.0;          // 2^28
  } else {
    temp = _DD_SPLITTER * a;
    retval.num = temp - (temp - a);
    retval.err = a - retval.num;
  }

  return retval;
}

/* Computes fl(a*b) and err(a*b). */
inline dd dd_two_prod(double a, double b){
  dd retval, split_a, split_b;

  retval.num = a * b;
  split_a = dd_split(a);
  split_b = dd_split(b);
  retval.err = ((split_a.num * split_b.num - retval.num) + split_a.num * split_b.err + split_a.err * split_b.num) + split_a.err * split_b.err;

  return retval;
}

#endif /* _UTIL_H */
