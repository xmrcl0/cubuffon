/** @file     cubuffon.h
 *  @brief    Function prototypes for buffon.
 *  @author   Marcelo Pinto (xmrcl0@gmail.com)
 *  @date     10/24/2017
 *  @date     10/25/2017
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>
#include <time.h>
#include <limits.h>


/** @brief Buffon experiment.
 *
 *  @param[in] l  Needle size
 *  @param[in] n  Number of iterations
 *  @param[in] x  Random vector for needle position
 *  @param[in] y  Random vector for needle angle
 *  @param[in] p  Calculated Pi value
 *  @param[in] s  Experimental error
 *  @return void 
 */
__global__ void buffon_exp (float l, int n, float *x, float *y, float *z, float *p, float *s); 
