/*
 * ss_double_fromdouble.h - Drop in "double" replacement for Single-Single (with singles as doubles)
 *
 */

#ifndef _SS_DOUBLE_FROMDOUBLE_H
#define _SS_DOUBLE_FROMDOUBLE_H

typedef struct{
  double num;
} ss;

/* 
 * TYPE CONVERSIONS
 */

/* single-single (double wrapper) */
ss ss_init();
ss num_num_to_ss(double hi, double lo);
ss num_to_ss(double h);
ss i_to_ss(int h);

/* 
 * BASIC ARITHMETIC 
 */

/* add */

/* ss(C) = double(a) + double(b) */
ss num_num_add_ss(const double a, const double b);
/* ss(C) = ss(a) + double(b) */
ss ss_num_add_ss(const ss a, const double b);
/* ss(C) = double(a) + ss(b) */
ss num_ss_add_ss(const double a, const ss b);
/* ss(C) = ss(a) + ss(b) */
ss ss_ss_add_ss(const ss a, const ss b);

/* sub */

/* ss(C) = num(a) - num(b) */
ss num_num_sub_ss(const double a, const double b);
/* ss(C) = ss(a) - num(b) */
ss ss_num_sub_ss(const ss a, const double b);
/* ss(C) = num(a) - ss(b) */
ss num_ss_sub_ss(const double a, const ss b);
/* ss(C) = ss(a) - ss(b) */
ss ss_ss_sub_ss(const ss a, const ss b);

/* mul */

/* ss(C) = num(a) * num(b) */
ss num_num_mul_ss(const double a, const double b);
/* ss(C) = ss(a) * num(b) */
ss ss_num_mul_ss(const ss a, const double b);
/* ss(C) = num(a) * ss(b) */
ss num_ss_mul_ss(const double a, const ss b);
/* ss(C) = ss(a) * ss(b) */
ss ss_ss_mul_ss(const ss a, const ss b);

/* div */

/* ss(C) = num(a) / num(b) */
ss num_num_div_ss(const double a, const double b);
/* ss(C) = ss(a) / num(b) */
ss ss_num_div_ss(const ss a, const double b);
/* ss(C) = ss(a) / ss(b) */
ss ss_ss_div_ss(const ss a, const ss b);
/* ss(C) = num(a) / ss(b) */
ss num_ss_div_ss(double a, const ss b);

/* reciprocal */

/* ss(B) = 1.0 / num(a) */
ss num_recip_ss(double a);
/* ss(B) = 1.0 / ss(a) */
ss ss_recip_ss(ss a);

/* copy */

ss ss_copy_ss(const ss a);

/* misc math.h */

/* is zero */
int is_zero_ss(const ss a);
/* is positive */
int is_positive_ss(const ss a);
/* is negative */
int is_negative_ss(const ss a);
/* floor ss to nearest int */
ss floor_ss(const ss a);
/* ceil ss to nearest int */
ss ceil_ss(const ss a);

#endif /* _SS_H */
