/*
 * ecodes.h - global error codes
 * 	
 */

#ifndef __ECODES_H__
#define __ECODES_H__

#include <stdlib.h>

/* stdlib defines EXIT_SUCCESS and EXIT_FAILURE */
#define BAD_COMMAND_LINE EXIT_FAILURE+1
#define RANDOM_ALLOCATION_ERROR EXIT_FAILURE+2

#endif /* __ECODES_H__ */
