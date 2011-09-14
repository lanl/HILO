/*
 * ss.h - Single-Single
 *
 * Code taken from QD 2.3.11, Copyright (c) 2000-2001 DoE
 */

#ifndef _SS_H
#define _SS_H

#include "util.h"
#include "types.h"

/* 
 * BASIC ARITHMETIC 
 */

/* add */

/* ss(C) = float(a) + float(b) */
inline ss num_num_add(const float a, const float b){
  return ss_two_sum(a, b);
}


/* ss(C) = ss(a) + float(b) */
inline ss ss_num_add(const ss a, const float b){
  /* containers and temporary variables */
  ss retval;

  /* perform ss arithmetic */
  retval = ss_two_sum(a.num, b);
  retval.err += a.err;
  retval = ss_quick_two_sum(retval.num, retval.err);
  
  return retval;
}

/* ss(C) = float(a) + ss(b) */
inline ss num_ss_add(const float a, const ss b){
  /* perform ss arithmetic */
  return ss_num_add(b, a);
}

/* ss(C) = ss(a) + ss(b) */
inline ss ss_ss_add(const ss a, const ss b){
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
inline ss num_num_sub(const float a, const float b){
 return ss_two_diff(a, b);
}

/* ss(C) = ss(a) - num(b) */
inline ss ss_num_sub(const ss a, const float b){
  /* containers and temporary variables */
  ss retval;

  /* perform ss arithmetic */
  retval = ss_two_diff(a.num, b);
  retval.err += a.err;
  retval = ss_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* ss(C) = num(a) - ss(b) */
inline ss num_ss_sub(const float a, const ss b){
  /* containers and temporary variables */
  ss retval;

  /* perform ss arithmetic */
  retval = ss_two_diff(a, b.num);
  retval.err -= b.err;
  retval = ss_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* ss(C) = ss(a) - ss(b) */
inline ss ss_ss_sub(const ss a, const ss b){
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
inline ss num_num_mul(const float a, const float b){
  return ss_two_prod(a, b);
}

/* ss(C) = ss(a) * num(b) */
inline ss ss_num_mul(const ss a, const float b){
  /* containers and temporary variables */
  ss retval;

  /* perform ss arithmetic */
  retval = ss_two_prod(a.num, b); // p2 is retval.err
  retval.err += (a.err * b);
  retval = ss_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* ss(C) = num(a) * ss(b) */
inline ss num_ss_mul(const float a, const ss b){
  /* perform ss arithmetic */
  return ss_num_mul(b, a);
}

/* ss(C) = ss(a) * ss(b) */
inline ss ss_ss_mul(const ss a, const ss b){
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
inline ss num_num_div(const float a, const float b){
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
inline ss ss_num_div(const ss a, const float b){
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
inline ss ss_ss_div(const ss a, const ss b){
  /* containers and temporary variables */
  ss q, p, retval;

  /* perform ss arithmetic */
  q = num_to_ss(a.num / b.num);  /* approximate quotient */
  retval = ss_ss_sub(a, num_ss_mul(q.num, b)); 
  q.err = retval.num / b.num;
  retval = ss_ss_sub(retval, num_ss_mul(q.err, b));
  p.num = retval.num / b.num; 
  q = ss_quick_two_sum(q.num, q.err);
  retval = ss_num_add(q, p.num);

  return retval;
}

/* ss(C) = num(a) / ss(b) */
inline ss num_ss_div(float a, const ss b){
  /* perform ss arithmetic */
  return ss_ss_div(num_to_ss(a), b);
}

/* copy */

inline ss ss_copy_ss(const ss a){
  /* containers and temporary variables */
  ss retval;

  /* copy the input value */
  retval.num = a.num;
  retval.err = a.err;

  return retval;
}

inline ss ss_copy_dd(const dd a){
  /* containers and temporary variables */
  ss retval;

  /* copy the input value */
  retval.num = (float)a.num;
  retval.err = (float)a.err;

  return retval;
}

#endif /* _SS_H */
