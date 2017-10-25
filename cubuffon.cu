/** @file     cubuffon.c
 *  @brief    Buffon functions.
 *  @author   Marcelo Pinto (xmrcl0@gmail.com)
 *  @date     10/24/2017
 *  @update   10/24/2017
 */

#include <cubuffon.h>

__global__
void buffon_exp(float l, int n, float *x, float *y, float *z, float *p, float *s) 
{
  //int index = threadIdx.x;
  //int stride = blockDim.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = index; i < n; i+=stride)
    if ((float) l/2 * cos(x[i] * M_PI_2) >= y[i] * l)  
      z[i] = 1;
    else
      z[i] = 0;

  *p = 0;
  for (int i = 0; i < n; i++)
      *p = *p + z[i];

  *p = (float) n / *p; 
  *s = (float) *p / sqrt((float) n); 
}
