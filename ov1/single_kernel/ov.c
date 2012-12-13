/*
 * Optimal Velocity Model Simulator
 * version C
 */


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include "defNumber.h"


double a = 1.0;  // Sensitivity
double s_time = 0;  // Simulation Time

extern char* conv(unsigned int res);

/*
 *  valiable for cuda
 */
CUresult res;
CUdevice dev;
CUcontext ctx;
CUfunction function;
CUmodule module;
CUdeviceptr x_dev, v_dev, error_dev, s_time_dev;
int thread_num, block_num;

/*
 *  valiable for time measuring
 */
struct timeval tv_s, tv_f;
double for_time = 0;




//----------------------------------------------------------------------
/*
 * OV function
 */
inline double
V(double dx){
  return tanh(dx - c)+tanh(c);
}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
/*
 * Initializaiton in order to use kernel program 
 */
void
init_cuda(void){

  thread_num = (N <= 16) ? N : 16 ;  
  block_num = N / (thread_num*thread_num);
  if(N % (thread_num*thread_num) != 0) block_num++;
  
  res = cuInit(0);
  if(res != CUDA_SUCCESS){
    printf("cuInit failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuDeviceGet(&dev, 0);
  if(res != CUDA_SUCCESS){
    printf("cuDeviceGet failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuCtxCreate(&ctx, 0, dev);
  if(res != CUDA_SUCCESS){
    printf("cuCtxCreate failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuModuleLoad(&module, "./cuda_main.cubin");
  if(res != CUDA_SUCCESS){
    printf("cuModuleLoad() failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuModuleGetFunction(&function, module, "cuda_main");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction() failed: res = %s\n",  conv(res));
    exit(1);
  }
  

  /* 
   * preparation for launch kernel 
   */
  res = cuFuncSetSharedSize(function, 0x40);  /* just random */
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetSharedSize() failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuFuncSetBlockShape(function, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape() failed: res = %s\n", conv(res));
    exit(1);
  }

}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
/*
 * get device memory
 */
void
get_dev_mem(void){

  res = cuMemAlloc(&x_dev, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemAlloc(&v_dev, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemAlloc(&error_dev, sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(error) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemAlloc(&s_time_dev, sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(s_time) failed: res = %s\n", conv(res));
    exit(1);
  }
  
}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
/*
 * upload data from host memory to device memory
 */
void
upload(double x[], double v[], int *error, double *s_time){

  res = cuMemcpyHtoD(x_dev, x, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemcpyHtoD(v_dev, v, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  
  res = cuMemcpyHtoD(error_dev, error, sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(error) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemcpyHtoD(s_time_dev, s_time, sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(s_time) failed: res = %s\n", conv(res));
    exit(1);
  }

}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
/*
 * parameter setting
 * provide kernel with needed parameter when it be launched
 */
void
parameter_set(void){

  res = cuParamSeti(function, 0, x_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(function, 4, x_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  
  res = cuParamSeti(function, 8, v_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(function, 12, v_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuParamSetv(function, 16, &a, 8);
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(a) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(function, 24, error_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(function, 28, error_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuParamSeti(function, 32, s_time_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(function, 36, s_time_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuParamSetSize(function, 40);
  if(res != CUDA_SUCCESS){
    printf("cuParaMSetSize() failed: res = %s\n", conv(res));
    exit(1);
  }
  

}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
/*
 * execute kernel function
 */
void
execute_cuda(void){

  
  res = cuLaunchGrid(function, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid() failed: res = %s\n", conv(res));
    exit(1);
  }

  
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS){
    printf("cuCtxSynchronize() failed: res = %s\n", conv(res));
    exit(1);
  }
  
}
//----------------------------------------------------------------------

//----------------------------------------------------------------------
/*
 * download data from device memory to host memory
 */
void
download(double x[], double v[], int *error, double *s_time){

  res = cuMemcpyDtoH(x, x_dev, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemcpyDtoH(v, v_dev, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemcpyDtoH(error, error_dev, sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(error) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemcpyDtoH(s_time, s_time_dev, sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(s_time) failed: res = %s\n", conv(res));
    exit(1);
  }

}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
/*
 * free device memory
 */
void
free_dev_mem(void){

  res = cuMemFree(x_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemFree(v_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemFree(error_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(error) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(s_time_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(s_time) failed: res = %s\n", conv(res));
    exit(1);
  }

}
//----------------------------------------------------------------------

//----------------------------------------------------------------------
/*
 * module unload and destroy context
 */
void
clean_cuda(void){

    res = cuModuleUnload(module);
    if(res != CUDA_SUCCESS){
      printf("cuModuleUnload failed: res = %s\n", conv(res));
      exit(1);
    }
    
    res = cuCtxDestroy(ctx);
    if(res != CUDA_SUCCESS){
      printf("cuCtxDestroy failed: res = %s\n", conv(res));
      exit(1);
    }

}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
/*
 * Initialization
 * Putting small perturbation to the uniform flow
 * with amplitude eps.
 */
void
init(double x[], double v[]){

  const double eps = 0.01;
  double dx = L/(double)N;
  double iv = V(dx);

  int i;

  for(i=0;i<N;i++){
    x[i] = dx*(double)i;
    v[i] = iv;
  }
  x[0] += eps;
  v[0] = V(dx - eps);
  v[N-1] = V(dx + eps);

  /* output constant values */
  printf("# dt = %f\n", dt);
  printf("# density = %f\n", density);
  printf("# a = %f\n", a);
  printf("# N = %d\n", N);
}


//----------------------------------------------------------------------
/*
 * Main Function
 */
int
main(int argc, char *argv[]){
  static double x[N],v[N];
  
  FILE *fp = fopen("time.dat", "w");
  
  int i, j;
  int error=0;

   
  /* initialize x and v values */
  init(x,v);


  /*
   * initilization
   */
  init_cuda();
  
  
  gettimeofday(&tv_s, NULL);  // time measuring start
  
  
  /*
   * get device memory 
   */
  get_dev_mem();
  
  
  /*
   * upload x[], v[] and error_checker 
   */
  upload(x, v, &error, &s_time);
  
  
  /*
   * set parameter 
   */
  parameter_set();
  
  
  /* 
   * execute function 
   */
  execute_cuda();
  
  
  /*
   * download x[], v[] and error_checker 
   */
  download(x, v, &error, &s_time);  
  
  
  /*
   * error management 
   */
  if(error == 1){
    printf("s_time = %f\n", s_time);
    printf("*** Invalid Time Step ***\n");
    exit(1);
  }
  
  /*
   * cleaning up 
   */
  free_dev_mem();  
  
  
  gettimeofday(&tv_f, NULL);  // time measuring end.
  for_time += (tv_f.tv_sec - tv_s.tv_sec)*1000*1000 + (tv_f.tv_usec - tv_s.tv_usec);  // time calculation
    
  
  clean_cuda();
  
  
  /* output calculation data */
  
    for(j=0;j<N;j++){
    printf("%f %.2f\n", x[j], s_time);
    }
  

  
  
  /* output time data */
  fprintf(fp, "N = %d\n", N);
  fprintf(fp, "a = %f\n", a);
  fprintf(fp, "%8.6f[micro sec]\n", for_time);
  
  
  fclose(fp);
  
  printf("# calculation done.\n");
}
//----------------------------------------------------------------------
// End of ov.c
//----------------------------------------------------------------------
