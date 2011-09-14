/*
 * ss.h - Single-Single
 *
 * Code taken from QD 2.3.11, Copyright (c) 2000-2001 DoE
 */

#ifndef _SS_H
#define _SS_H

typedef struct{
  float num;
  float err;
} ss;

/* 
 * TYPE CONVERSIONS
 */

/* single-single */
ss ss_init();
ss num_num_to_ss(float hi, float lo);
ss num_to_ss(float h);
ss i_to_ss(int h);

/* 
 * BASIC ARITHMETIC 
 */

/* add */

/* ss(C) = float(a) + float(b) */
ss num_num_add_ss(const float a, const float b);
/* ss(C) = ss(a) + float(b) */
ss ss_num_add_ss(const ss a, const float b);
/* ss(C) = float(a) + ss(b) */
ss num_ss_add_ss(const float a, const ss b);
/* ss(C) = ss(a) + ss(b) */
ss ss_ss_add_ss(const ss a, const ss b);

/* sub */

/* ss(C) = num(a) - num(b) */
ss num_num_sub_ss(const float a, const float b);
/* ss(C) = ss(a) - num(b) */
ss ss_num_sub_ss(const ss a, const float b);
/* ss(C) = num(a) - ss(b) */
ss num_ss_sub_ss(const float a, const ss b);
/* ss(C) = ss(a) - ss(b) */
ss ss_ss_sub_ss(const ss a, const ss b);

/* mul */

/* ss(C) = num(a) * num(b) */
ss num_num_mul_ss(const float a, const float b);
/* ss(C) = ss(a) * num(b) */
ss ss_num_mul_ss(const ss a, const float b);
/* ss(C) = num(a) * ss(b) */
ss num_ss_mul_ss(const float a, const ss b);
/* ss(C) = ss(a) * ss(b) */
ss ss_ss_mul_ss(const ss a, const ss b);

/* div */

/* ss(C) = num(a) / num(b) */
ss num_num_div_ss(const float a, const float b);
/* ss(C) = ss(a) / num(b) */
ss ss_num_div_ss(const ss a, const float b);
/* ss(C) = ss(a) / ss(b) */
ss ss_ss_div_ss(const ss a, const ss b);
/* ss(C) = num(a) / ss(b) */
ss num_ss_div_ss(float a, const ss b);

/* reciprocal */

/* ss(B) = 1.0 / num(a) */
ss num_recip_ss(float a);
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
