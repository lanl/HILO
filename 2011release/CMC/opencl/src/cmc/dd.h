/*
 * dd.h - Double-Double
 *
 * Code taken from QD 2.3.11, Copyright (c) 2000-2001 DoE
 */

#ifndef _DD_H
#define _DD_H

#include "util.h"
#include "types.h"

/* 
 * BASIC ARITHMETIC 
 */

/* add */

/* dd(C) = float(a) + float(b) */
inline dd num_num_add(const float a, const float b){
  return dd_two_sum(a, b);
}


/* dd(C) = dd(a) + float(b) */
inline dd dd_num_add(const dd a, const float b){
  /* containers and temporary variables */
  dd retval;

  /* perform dd arithmetic */
  retval = dd_two_sum(a.num, b);
  retval.err += a.err;
  retval = dd_quick_two_sum(retval.num, retval.err);
  
  return retval;
}

/* dd(C) = float(a) + dd(b) */
inline dd num_dd_add(const float a, const dd b){
  /* perform dd arithmetic */
  return dd_num_add(b, a);
}

/* dd(C) = dd(a) + dd(b) */
inline dd dd_dd_add(const dd a, const dd b){
  /* containers and temporary variables */
  dd temp, retval;
  
  /* perform dd arithmetic */
  retval = dd_two_sum(a.num, b.num);
  temp = dd_two_sum(a.err, b.err);   
  retval.err += temp.num;
  retval = dd_quick_two_sum(retval.num, retval.err);
  retval.err += temp.err;
  retval = dd_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* sub */

/* dd(C) = num(a) - num(b) */
inline dd num_num_sub(const float a, const float b){
 return dd_two_diff(a, b);
}

/* dd(C) = dd(a) - num(b) */
inline dd dd_num_sub(const dd a, const float b){
  /* containers and temporary variables */
  dd retval;

  /* perform dd arithmetic */
  retval = dd_two_diff(a.num, b);
  retval.err += a.err;
  retval = dd_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* dd(C) = num(a) - dd(b) */
inline dd num_dd_sub(const float a, const dd b){
  /* containers and temporary variables */
  dd retval;

  /* perform dd arithmetic */
  retval = dd_two_diff(a, b.num);
  retval.err -= b.err;
  retval = dd_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* dd(C) = dd(a) - dd(b) */
inline dd dd_dd_sub(const dd a, const dd b){
  /* containers and temporary variables */
  dd retval;
  
  /* perform dd arithmetic (QD_IEEE_ADD variant) */
  retval = dd_two_diff(a.num, b.num);
  retval.err += a.err;
  retval.err -= b.err;
  retval = dd_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* mul */

/* dd(C) = num(a) * num(b) */
inline dd num_num_mul(const float a, const float b){
  return dd_two_prod(a, b);
}

/* dd(C) = dd(a) * num(b) */
inline dd dd_num_mul(const dd a, const float b){
  /* containers and temporary variables */
  dd retval;

  /* perform dd arithmetic */
  retval = dd_two_prod(a.num, b); // p2 is retval.err
  retval.err += (a.err * b);
  retval = dd_quick_two_sum(retval.num, retval.err);

  return retval;
}

/* dd(C) = num(a) * dd(b) */
inline dd num_dd_mul(const float a, const dd b){
  /* perform dd arithmetic */
  return dd_num_mul(b, a);
}

/* dd(C) = dd(a) * dd(b) */
inline dd dd_dd_mul(const dd a, const dd b){
  /* containers and temporary variables */
  dd retval;
  
  /* perform dd arithmetic */
  retval = dd_two_prod(a.num, b.num);
  retval.err += (a.num * b.err + a.err * b.num);
  retval = dd_quick_two_sum(retval.num, retval.err);
  
  return retval;
}

/* div */

/* dd(C) = num(a) / num(b) */
inline dd num_num_div(const float a, const float b){
  /* containers and temporary variables */
  dd q, p, retval;

  /* perform dd arithmetic */
  q = num_to_dd(a / b);
  /* Compute  a - q.num * b */
  p = dd_two_prod(q.num, b);
  retval = dd_two_diff(a, p.num);
  retval.err -= p.err;
  /* get next approximation */
  q.err = (retval.num + retval.err) / b;
  retval = dd_quick_two_sum(q.num, q.err);

  return retval;
}

/* dd(C) = dd(a) / num(b) */
inline dd dd_num_div(const dd a, const float b){
  /* containers and temporary variables */
  dd q, p, retval;
  
  /* perform dd arithmetic */
  q = num_to_dd(a.num / b);   /* approximate quotient. */

  /* Compute  this - q.num * d */
  p = dd_two_prod(q.num, b);
  retval = dd_two_diff(a.num, p.num);
  retval.err += a.err;
  retval.err -= p.err;
  
  /* get next approximation. */
  q.err = (retval.num + retval.err) / b;

  /* renormalize */
  retval = dd_quick_two_sum(q.num, q.err);

  return retval;
}

/* dd(C) = dd(a) / dd(b) */
inline dd dd_dd_div(const dd a, const dd b){
  /* containers and temporary variables */
  dd q, p, retval;

  /* perform dd arithmetic */
  q = num_to_dd(a.num / b.num);  /* approximate quotient */
  retval = dd_dd_sub(a, num_dd_mul(q.num, b)); 
  q.err = retval.num / b.num;
  retval = dd_dd_sub(retval, num_dd_mul(q.err, b));
  p.num = retval.num / b.num; 
  q = dd_quick_two_sum(q.num, q.err);
  retval = dd_num_add(q, p.num);

  return retval;
}

/* dd(C) = num(a) / dd(b) */
inline dd num_dd_div(float a, const dd b){
  /* perform dd arithmetic */
  return dd_dd_div(num_to_dd(a), b);
}

/* copy */

inline dd dd_copy_dd(const dd a){
  /* containers and temporary variables */
  dd retval;

  /* copy the input value */
  retval.num = a.num;
  retval.err = a.err;

  return retval;
}

inline dd dd_copy_ss(const ss a){
  /* containers and temporary variables */
  dd retval;

  /* copy the input value */
  retval.num = (double)a.num;
  retval.err = (double)a.err;

  return retval;
}

#endif /* _DD_H */
