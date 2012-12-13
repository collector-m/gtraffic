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
CUfunction func1, func2, func3, func4;
CUmodule module;
CUdeviceptr x_dev, v_dev, error_dev;
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
  
  res = cuModuleLoad(&module, "./kernel_RungeKutta.cubin");
  if(res != CUDA_SUCCESS){
    printf("cuModuleLoad() failed: res = %s\n", conv(res));
    exit(1);
  }
  
  /* RungeKutta method is devided into 4 phases.  */
  res = cuModuleGetFunction(&func1, module, "RK_phase1");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase1) failed: res = %s\n",  conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func2, module, "RK_phase2");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase2) failed: res = %s\n",  conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func3, module, "RK_phase3");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase3) failed: res = %s\n",  conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func4, module, "RK_phase4");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase4) failed: res = %s\n",  conv(res));
    exit(1);
  }
  

  /* 
   * preparation for launch kernel 
   */

  /* set size of shared memory */
  res = cuFuncSetSharedSize(func1, 0x40);  /* just random */
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetSharedSize() failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetSharedSize(func2, 0x40);  /* just random */
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetSharedSize() failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetSharedSize(func3, 0x40);  /* just random */
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetSharedSize() failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetSharedSize(func4, 0x40);  /* just random */
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetSharedSize() failed: res = %s\n", conv(res));
    exit(1);
  }
  
  /* set Dimensions of block */
  res = cuFuncSetBlockShape(func1, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape() failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func2, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape() failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func3, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape() failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func4, thread_num, thread_num, 1);
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

}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
/*
 * parameter setting
 * provide kernel with needed parameter when it be launched
 */
void
parameter_set_for_func1(void){

  res = cuParamSeti(func1, 0, x_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func1, 4, x_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  
  res = cuParamSeti(func1, 8, v_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func1, 12, v_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuParamSetv(func1, 16, &a, 8);
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(a) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func1, 24, error_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func1, 28, error_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuParamSetSize(func1, 32);
  if(res != CUDA_SUCCESS){
    printf("cuParaMSetSize() failed: res = %s\n", conv(res));
    exit(1);
  }
  

}
//----------------------------------------------------------------------
void
parameter_set_for_func2(void){

  res = cuParamSeti(func2, 0, x_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func2, 4, x_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  
  res = cuParamSeti(func2, 8, v_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func2, 12, v_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuParamSetv(func2, 16, &a, 8);
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(a) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func2, 24, error_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func2, 28, error_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuParamSetSize(func2, 32);
  if(res != CUDA_SUCCESS){
    printf("cuParaMSetSize() failed: res = %s\n", conv(res));
    exit(1);
  }
  

}
//----------------------------------------------------------------------
void
parameter_set_for_func3(void){

  res = cuParamSeti(func3, 0, x_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func3, 4, x_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  
  res = cuParamSeti(func3, 8, v_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func3, 12, v_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuParamSetv(func3, 16, &a, 8);
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(a) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func3, 24, error_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func3, 28, error_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuParamSetSize(func3, 32);
  if(res != CUDA_SUCCESS){
    printf("cuParaMSetSize() failed: res = %s\n", conv(res));
    exit(1);
  }
  

}
//----------------------------------------------------------------------
void
parameter_set_for_func4(void){

  res = cuParamSeti(func4, 0, x_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func4, 4, x_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  
  res = cuParamSeti(func4, 8, v_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func4, 12, v_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuParamSetv(func4, 16, &a, 8);
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(a) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func4, 24, error_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(func4, 28, error_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(error) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuParamSetSize(func4, 32);
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

  res = cuLaunchGrid(func1, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func1) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuLaunchGrid(func2, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func1) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuLaunchGrid(func3, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func1) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuLaunchGrid(func4, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func1) failed: res = %s\n", conv(res));
    exit(1);
  }
  
}  

//----------------------------------------------------------------------
/*
 * execute kernel function
 */
void
synchronization(void){

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


  /* initilization */
  init_cuda();
  
  
  gettimeofday(&tv_s, NULL);  // time measuring start
  
  
  /* get device memory */
  get_dev_mem();
  
  
  /* upload x[], v[] and error_checker */
  upload(x, v, &error, &s_time);
  
  
  /* set parameter */
  parameter_set_for_func1();
  parameter_set_for_func2();
  parameter_set_for_func3();
  parameter_set_for_func4();
  
  
  /* execute function */
  for(i=0; s_time<TIME; i++){
    execute_cuda();
    s_time += dt;
  }
  
  /* synchronization */
  synchronization();

  /* download x[], v[] and error_checker  */
  download(x, v, &error, &s_time);  
  
  
  /* error management */
  if(error == 1){
    printf("s_time = %f\n", s_time);
    printf("*** Invalid Time Step ***\n");
    exit(1);
  }
  
  /* cleaning up */
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
