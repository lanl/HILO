/*
 * types.h - Typedefs / Structs
 *
 * Code taken from QD 2.3.11, Copyright (c) 2000-2001 DoE
 */

#ifndef _TYPES_H
#define _TYPES_H

/* single-single */

typedef struct{
  float num;
  float err;
} ss;

inline ss ss_init(){
  /* initial ss */
  ss retval;
  retval.num = 0.0; 
  retval.err = 0.0; 
  return retval;
}

inline ss num_num_to_ss(float hi, float lo){ 
  /* float/float to ss */
  ss retval;
  retval.num = hi; 
  retval.err = lo;
  return retval;
}

inline ss num_to_ss(float h){ 
  /* float to ss */
  ss retval;
  retval.num = h; 
  retval.err = 0.0; 
  return retval;
}

inline ss i_to_ss(int h){
  /* int to ss */
  ss retval;
  retval.num = (float)h;
  retval.err = 0.0;
  return retval;
}

/* double-double */

typedef struct{
  double num;
  double err;
} dd;

inline dd dd_init(){
  /* initial dd */
  dd retval;
  retval.num = 0.0; 
  retval.err = 0.0; 
  return retval;
}

inline dd num_num_to_dd(double hi, double lo){ 
  /* double/double to dd */
  dd retval;
  retval.num = hi; 
  retval.err = lo;
  return retval;
}

inline dd num_to_dd(double h){ 
  /* double to dd */
  dd retval;
  retval.num = h; 
  retval.err = 0.0; 
  return retval;
}

inline dd i_to_dd(int h){
  /* int to dd */
  dd retval;
  retval.num = (double)h;
  retval.err = 0.0;
  return retval;
}

#endif /* _UTIL_H */
