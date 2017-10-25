/** @file     main.c
 *  @brief    Calculates pi using Buffon's technique.
 *  @author   Marcelo Pinto (xmrcl0@gmail.com)
 *  @date     10/24/2017
 *  @update   10/25/2017
 */

#include <utils.h>
#include <cubuffon.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
 printf("Error at %s:%d\n",__FILE__,__LINE__);\
 return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
 printf("Error at %s:%d\n",__FILE__,__LINE__);\
 return EXIT_FAILURE;}} while(0)

void
help (void)
{
  printf ("usage: cubuffon [-h] -n <ndrop> -l <nsize> -x <numBlocks> -b <blockSize>\n");
  printf ("Calculates PI using Buffon's technique using CUDA.\n\n");
  printf ("Options:\n");
  printf ("  -n <ndrop>     Number of needle drops\n");
  printf ("  -l <nsize>     Needle size\n");
  printf ("  -b <blockSize> CUDA blockSize\n");
  printf ("  -x <numBlocks> CUDA numBlocks\n");
  printf ("  -h             Show this help message and exit\n\n");
  printf ("Examples:\n");
  printf ("  cubuffon -n 1000 -l 1 -x 1 -b 256\n");
}


int
parse_cmdline(int argc, char **argv, unsigned long long *n, float *l, int *sflag, int *vflag, int *b, int *x)
{
  int i, c, nflag = 0, lflag = 0;

  opterr = 0;
  while ((c = getopt (argc, argv, "l:n:b:x:h::")) != -1)
  {
    switch (c)
    {
    case 'n':
      nflag = 1;
      if (!is_natural_num (optarg))
      {
	      fprintf (stderr, "%s: error: number of drops must be an integer\n", argv[0]);
	      return 1;
      }
      else
	      *n = strtoull (optarg, NULL, 10);
      break;
    case 'l':
      lflag = 1;
      if (!is_positive_num (optarg))
      {
	      fprintf (stderr, "%s: error: needle size must be positive\n", argv[0]);
	      return 1;
      }
      else
	      *l = strtod (optarg, NULL);
      break;
    case 'b':
      lflag = 1;
      if (!is_positive_num (optarg))
      {
	      fprintf (stderr, "%s: error: blockSize size must be positive\n", argv[0]);
	      return 1;
      }
      else
	      *b = strtod (optarg, NULL);
      break;
    case 'x':
      lflag = 1;
      if (!is_positive_num (optarg))
      {
	      fprintf (stderr, "%s: error: numBlocks must be positive\n", argv[0]);
	      return 1;
      }
      else
	      *x = strtod (optarg, NULL);
      break;
    case 's':
      *sflag = 1;
      break;
    case 'v':
      *vflag = 1;
      break;
    case 'h':
      help ();
	    return 1;
    case '?':
      fprintf (stderr, "%s: error: invalid option\n", argv[0]);
      fprintf (stderr, "usage: cubuffon [-h] -n <ndrop> -l <nsize> -x <numBlocks> -b <blockSize>\n");
      return 1;
    default:
      fprintf (stderr, "usage: cubuffon [-h] -n <ndrop> -l <nsize> -x <numBlocks> -b <blockSize>\n");
      return 1;
    }
  }

  // Check integer overflow 
  for (i = optind; i < argc; i++)
  {
    fprintf (stderr, "%s: error: too many or too few arguments\n", argv[0]);
    return 1;
  }

  // Check integer overflow 
  if (n + 1 < n)
  {
    fprintf (stderr, "%s: error: number of drops must be less than %llu \n", argv[0], ULLONG_MAX);
    return 1;
  }

  // Check if obrigatory argumets were given
  if (nflag == 0 || lflag == 0)
  {
    fprintf (stderr, "%s: error: too few parameters\n", argv[0]);
    fprintf (stderr, "usage: cubuffon [-h] -n <ndrop> -l <nsize> -x <numBlocks> -b <blockSize>\n");
    return 1;
  }
  return 0;
}


int
main (int argc, char **argv)
{
  //size_t i;
  int sflag = 0, vflag = 0, blockSize = 256, numBlocks = 1;
  unsigned long long int n = 0;
  float l;
  float *da, *db, *dc, *dp, *ds, *hp, *hs;
  curandGenerator_t gen;

  // Parse command line arguments
  if (parse_cmdline (argc, argv, &n, &l, &sflag, &vflag, &blockSize, &numBlocks))
  {
    return 1;
  }

  // Allocate n floats on host
  hp = (float *)calloc(1, sizeof(float));
  hs = (float *)calloc(1, sizeof(float));

  // Create pseudo-random number generator
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  
  // Set seed
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

  // Allocate n floats on device
  CUDA_CALL(cudaMalloc((void **)&da, n * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&db, n * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&dc, n * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&dp, 1 * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&ds, 1 * sizeof(float)));

  // Generate n floats on device
  CURAND_CALL(curandGenerateUniform(gen, da, n));
  CURAND_CALL(curandGenerateUniform(gen, db, n));

  // Run Buffon experiment on gpu
  //int numBlocks = (n + blockSize - 1) / blockSize;
  buffon_exp<<<numBlocks, blockSize>>> (l, n, da, db, dc, dp, ds);

  // Copy device memory to host
  CUDA_CALL(cudaMemcpy(hp, dp, 1 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hs, ds, 1 * sizeof(float), cudaMemcpyDeviceToHost));

  printf ("pi = %f, exp_err = %f, real_err = %f\n", *hp, *hs, fabs (M_PI - *hs));

  /* Cleanup */
  CURAND_CALL(curandDestroyGenerator(gen));
  CUDA_CALL(cudaFree(da));
  CUDA_CALL(cudaFree(db));
  CUDA_CALL(cudaFree(dp));
  CUDA_CALL(cudaFree(ds));
  free(hp);
  free(hs);

  return 0;
}
