#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include "2Dimtraffic_sync.h"


char *conv(unsigned int res);
void parameter_set(CUfunction function);
void parameter_set_calc(CUfunction function);


CUresult res;
CUdevice dev;
CUcontext ctx;
CUfunction func_init, func_const;
CUfunction func_exe_phase1_calc, func_exe_phase1;
CUfunction func_exe_phase2_calc, func_exe_phase2;
CUfunction func_exe_phase3_calc, func_exe_phase3;
CUfunction func_exe_phase4_calc, func_exe_phase4;
CUfunction func_calc_neighbor;
CUmodule module;
CUdeviceptr x_dev, y_dev, vx_dev, vy_dev;
int thread_num, block_num;

int N_tmp;

 CUdeviceptr v_dev;
 CUdeviceptr dx_dev; // -0.5*LX<= dx <= 0.5*LX
 CUdeviceptr dy_dev; // -0.5*LY<= dy <= 0.5*LY 
 CUdeviceptr dr_dev;  
 CUdeviceptr pointer_dev;
 CUdeviceptr kx1_dev, kx2_dev, kx3_dev, kx4_dev;
 CUdeviceptr ky1_dev, ky2_dev, ky3_dev, ky4_dev;
 CUdeviceptr kvx1_dev, kvx2_dev, kvx3_dev, kvx4_dev;
 CUdeviceptr kvy1_dev, kvy2_dev, kvy3_dev, kvy4_dev;
 CUdeviceptr sx_dev, sy_dev, svx_dev, svy_dev;


//----------------------------------------------------------------------
/*
 * Initializaiton in order to use kernel program for 2Dim OV simulation 
 */
void trf_init2Dim(int N){

  int byteoffset = 0;
  double dbN;
  N_tmp = N;


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

  res = cuModuleLoad(&module, "./solve_by_cuda_sync.cubin");
  if(res != CUDA_SUCCESS){
    printf("cuModuleLoad() failed: res = %s\n", conv(res));
    exit(1);
  }

  /* get 7 cuda functions for simulation  */
  res = cuModuleGetFunction(&func_init, module, "init_CUDA");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(init) failed: res = %s\n",  conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_const, module, "const_value_set");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(const_value_set) failed: res = %s\n",  conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_exe_phase1_calc, module, "trf_phase1_calc");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase1_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_exe_phase1, module, "trf_phase1");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase1) failed: res = %s\n",  conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_exe_phase2_calc, module, "trf_phase2_calc");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase2_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_exe_phase2, module, "trf_phase2");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase2) failed: res = %s\n",  conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_exe_phase3_calc, module, "trf_phase3_calc");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase3_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_exe_phase3, module, "trf_phase3");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase3) failed: res = %s\n",  conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_exe_phase4_calc, module, "trf_phase4_calc");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase4_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_exe_phase4, module, "trf_phase4");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(phase4) failed: res = %s\n",  conv(res));
    exit(1);
  }

  res = cuModuleGetFunction(&func_calc_neighbor, module, "trf_calc_neighbor");
  if(res != CUDA_SUCCESS){
    printf("cuModuleGetFunction(calc_neighbor failed: res = %s\n", conv(res));
    exit(1);
  }


  /* 
   * preparation for launch kernel 
   */


  /* set Dimensions of block */
  res = cuFuncSetBlockShape(func_init, 1, 1, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(init) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func_const, 1, 1, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(const) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func_exe_phase1_calc, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(exe_phase1_calc) failed: res = %s\n", conv(res));
    exit(1);
  }
 
  res = cuFuncSetBlockShape(func_exe_phase1, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(exe_phase1) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func_exe_phase2_calc, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(exe_phase2_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func_exe_phase2, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(exe_phase2) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func_exe_phase3_calc, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(exe_phase3_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func_exe_phase3, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(exe_phase3) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func_exe_phase4_calc, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(exe_phase4_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func_exe_phase4, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(exe_phase4) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuFuncSetBlockShape(func_calc_neighbor, thread_num, thread_num, 1);
  if(res != CUDA_SUCCESS){
    printf("cuFuncSetBlockShape(calc_neighbor) failed: res = %s\n", conv(res));
    exit(1);
  }



  /* get device memory */
  
    res = cuMemAlloc(&x_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(x) failed: res = %s\n", conv(res));
      exit(1);
    }
  
  
  
    res = cuMemAlloc(&y_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(y) failed: res = %s\n", conv(res));
      exit(1);
    }
  
  
 
    res = cuMemAlloc(&vx_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(vx) failed: res = %s\n", conv(res));
      exit(1);
    }



    res = cuMemAlloc(&vy_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(vy) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&v_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(v) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&dx_dev, N * N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(dx) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&dy_dev, N * N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(dy) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&dr_dev, N * N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(dr) failed: res = %s\n", conv(res));
      exit(1);
    } 

    res = cuMemAlloc(&pointer_dev, N * 6 * sizeof(int));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(pointer) failed: res = %s\n", conv(res));
      exit(1);
    }  

    res = cuMemAlloc(&kx1_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kx1) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kx2_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kx2) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kx3_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kx3) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kx4_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kx4) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&ky1_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(ky1) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&ky2_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(ky2) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&ky3_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(ky3) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&ky4_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(ky4) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kvx1_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kvx1) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kvx2_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kvx2) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kvx3_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kvx3) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kvx4_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kvx4) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kvy1_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kvy1) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kvy2_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kvy2) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kvy3_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kvy3) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&kvy4_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(kvy4) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&sx_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(sx) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&sy_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(sy) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&svx_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(svx) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&svy_dev, N * sizeof(double));
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(svy) failed: res = %s\n", conv(res));
      exit(1);
    }


    /* execute init_CUDA() */
      /* set parameter */
    dbN = (double)N;
    res = cuParamSetv(func_init, byteoffset, &dbN, sizeof(dbN));
    if(res != CUDA_SUCCESS){
      printf("cuParamSetv(N) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(dbN);

    res = cuParamSeti(func_init, byteoffset, v_dev);
    res = cuParamSeti(func_init, byteoffset+4, v_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(v) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(v_dev);

    res = cuParamSeti(func_init, byteoffset, dx_dev);
    res = cuParamSeti(func_init, byteoffset+4, dx_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(dx) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(dx_dev);

    res = cuParamSeti(func_init, byteoffset, dy_dev);
    res = cuParamSeti(func_init, byteoffset+4, dy_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(dy) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(dy_dev);

    res = cuParamSeti(func_init, byteoffset, dr_dev);
    res = cuParamSeti(func_init, byteoffset+4, dr_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(dr) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(dr_dev);

    res = cuParamSeti(func_init, byteoffset, pointer_dev);
    res = cuParamSeti(func_init, byteoffset+4, pointer_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(pointer) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(pointer_dev);

    res = cuParamSeti(func_init, byteoffset, kx1_dev);
    res = cuParamSeti(func_init, byteoffset+4, kx1_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kx1) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kx1_dev);

    res = cuParamSeti(func_init, byteoffset, kx2_dev);
    res = cuParamSeti(func_init, byteoffset+4, kx2_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kx2) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kx2_dev);

    res = cuParamSeti(func_init, byteoffset, kx3_dev);
    res = cuParamSeti(func_init, byteoffset+4, kx3_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kx3) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kx3_dev);

    res = cuParamSeti(func_init, byteoffset, kx4_dev);
    res = cuParamSeti(func_init, byteoffset+4, kx4_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kx4) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kx4_dev);

    res = cuParamSeti(func_init, byteoffset, ky1_dev);
    res = cuParamSeti(func_init, byteoffset+4, ky1_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(ky1) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(ky1_dev);

    res = cuParamSeti(func_init, byteoffset, ky2_dev);
    res = cuParamSeti(func_init, byteoffset+4, ky2_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(ky2) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(ky2_dev);

    res = cuParamSeti(func_init, byteoffset, ky3_dev);
    res = cuParamSeti(func_init, byteoffset+4, ky3_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(ky3) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(ky3_dev);

    res = cuParamSeti(func_init, byteoffset, ky4_dev);
    res = cuParamSeti(func_init, byteoffset+4, ky4_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(ky4) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(ky4_dev);

    res = cuParamSeti(func_init, byteoffset, kvx1_dev);
    res = cuParamSeti(func_init, byteoffset+4, kvx1_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kvx1) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kvx1_dev);

    res = cuParamSeti(func_init, byteoffset, kvx2_dev);
    res = cuParamSeti(func_init, byteoffset+4, kvx2_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kvx2) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kvx2_dev);

    res = cuParamSeti(func_init, byteoffset, kvx3_dev);
    res = cuParamSeti(func_init, byteoffset+4, kvx3_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kvx3) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kvx3_dev);

    res = cuParamSeti(func_init, byteoffset, kvx4_dev);
    res = cuParamSeti(func_init, byteoffset+4, kvx4_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kvx4) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kvx4_dev);

    res = cuParamSeti(func_init, byteoffset, kvy1_dev);
    res = cuParamSeti(func_init, byteoffset+4, kvy1_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kvy1) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kvy1_dev);

    res = cuParamSeti(func_init, byteoffset, kvy2_dev);
    res = cuParamSeti(func_init, byteoffset+4, kvy2_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kvy2) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kvy2_dev);

    res = cuParamSeti(func_init, byteoffset, kvy3_dev);
    res = cuParamSeti(func_init, byteoffset+4, kvy3_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kvy3) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kvy3_dev);

    res = cuParamSeti(func_init, byteoffset, kvy4_dev);
    res = cuParamSeti(func_init, byteoffset+4, kvy4_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(kvy4) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(kvy4_dev);

    res = cuParamSeti(func_init, byteoffset, sx_dev);
    res = cuParamSeti(func_init, byteoffset+4, sx_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(sx) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(sx_dev);

    res = cuParamSeti(func_init, byteoffset, sy_dev);
    res = cuParamSeti(func_init, byteoffset+4, sy_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(sy) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(sy_dev);

    res = cuParamSeti(func_init, byteoffset, svx_dev);
    res = cuParamSeti(func_init, byteoffset+4, svx_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(svx) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(svx_dev);

    res = cuParamSeti(func_init, byteoffset, svy_dev);
    res = cuParamSeti(func_init, byteoffset+4, svy_dev >> 32);
    if(res != CUDA_SUCCESS){
      printf("cuParamseti(svy) failed: res = %s\n", conv(res));
      exit(1);
    }
    byteoffset += sizeof(svy_dev);


    res = cuParamSetSize(func_init, byteoffset);
    if(res != CUDA_SUCCESS){
      printf("cuParaMSetSize(assign) failed: res = %s\n", conv(res));
      exit(1);
    }
    
    res = cuLaunchGrid(func_init, 1, 1);
    if(res != CUDA_SUCCESS){
      printf("cuLaunchGrid(func_init) failed: res = %s\n", conv(res));
      exit(1);
    }
    
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS){
      printf("cuCtxSynchronize(init) failed: res = %s\n", conv(res));
      exit(1);
    }
    
    

}
// ------------------ trf_init2Dim() -------------------------------------
// -----------------------------------------------------------------------





//-------------------------------------------------------------------------------
/*
 *  assign constant values, upload data, 
 *  execute calculation for simulation and download calculaton result  
 */
void trf_OV2Dim(double x[], double y[], double vx[], double vy[], double Lx, double Ly, double dt, double a, double b, double c, double d, double h, double flow, double range, int typeCode){

  int N = N_tmp; 
  double dbtypeCode;
  int byteoffset = 0;

  /* assign constant values and execute Initializaiton function in kernel*/

  res = cuParamSetv(func_const, byteoffset, &Lx, sizeof(Lx));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(Lx) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(Lx);

  res = cuParamSetv(func_const, byteoffset, &Ly, sizeof(Ly));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(Ly) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(Ly);

  res = cuParamSetv(func_const, byteoffset, &dt, sizeof(dt));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(dt) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(dt);

  res = cuParamSetv(func_const, byteoffset, &a, sizeof(a));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(a) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(a);


  res = cuParamSetv(func_const, byteoffset, &b, sizeof(b));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(b) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(b);


  res = cuParamSetv(func_const, byteoffset, &c, sizeof(c));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(c) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(c);


  res = cuParamSetv(func_const, byteoffset, &d, sizeof(d));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(d) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(d);


  res = cuParamSetv(func_const, byteoffset, &h, sizeof(h));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(h) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(h);


  res = cuParamSetv(func_const, byteoffset, &flow, sizeof(flow));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(flow) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(flow);


  res = cuParamSetv(func_const, byteoffset, &range, sizeof(range));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(range) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(range);


  //printf("typeCode_lib = %d\n", typeCode);
  dbtypeCode  = (double)typeCode;
  res = cuParamSetv(func_const, byteoffset, &dbtypeCode, sizeof(dbtypeCode));
  if(res != CUDA_SUCCESS){
    printf("cuParamSetv(typeCode) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(dbtypeCode);


  res = cuParamSetSize(func_const, byteoffset);
  if(res != CUDA_SUCCESS){
    printf("cuParaMSetSize(const) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuLaunchGrid(func_const, 1, 1);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func_const) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS){
    printf("cuCtxSynchronize(const) failed: res = %s\n", conv(res));
    exit(1);
  }

  // ---------------- assign constant values finish --------------------




  /* upload data */
  res = cuMemcpyHtoD(x_dev, x, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemcpyHtoD(y_dev, y, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(y) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemcpyHtoD(vx_dev, vx, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(vx) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemcpyHtoD(vy_dev, vy, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(vy) failed: res = %s\n", conv(res));
    exit(1);
  }

  /* parameter set */
  parameter_set(func_exe_phase1_calc);
  parameter_set(func_exe_phase1);
  parameter_set_calc(func_exe_phase2_calc);
  parameter_set(func_exe_phase2);
  parameter_set_calc(func_exe_phase3_calc);
  parameter_set(func_exe_phase3);
  parameter_set_calc(func_exe_phase4_calc);
  parameter_set(func_exe_phase4);


  /* execute funciton */
  res = cuLaunchGrid(func_exe_phase1_calc, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func1_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  parameter_set_calc(func_calc_neighbor);
  res = cuLaunchGrid(func_calc_neighbor, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(calc_neighbor) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuLaunchGrid(func_exe_phase1, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func1) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuLaunchGrid(func_exe_phase2_calc, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func2_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  parameter_set_calc(func_calc_neighbor);
  res = cuLaunchGrid(func_calc_neighbor, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(calc_neighbor) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuLaunchGrid(func_exe_phase2, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func2) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuLaunchGrid(func_exe_phase3_calc, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func3_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  parameter_set_calc(func_calc_neighbor);
  res = cuLaunchGrid(func_calc_neighbor, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(calc_neighbor) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuLaunchGrid(func_exe_phase3, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func3) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuLaunchGrid(func_exe_phase4_calc, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func4_calc) failed: res = %s\n", conv(res));
    exit(1);
  }

  parameter_set_calc(func_calc_neighbor);
  res = cuLaunchGrid(func_calc_neighbor, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(calc_neighbor) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuLaunchGrid(func_exe_phase4, 1, block_num);
  if(res != CUDA_SUCCESS){
    printf("cuLaunchGrid(func4) failed: res = %s\n", conv(res));
    exit(1);
  }

  
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS){
    printf("cuCtxSynchronize(exe) failed: res = %s\n", conv(res));
     exit(1);
    }

  /* download calculation results */
  res = cuMemcpyDtoH(x, x_dev, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemcpyDtoH(y, y_dev, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(y) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemcpyDtoH(vx, vx_dev, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(vx) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemcpyDtoH(vy, vy_dev, N * sizeof(double));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(vy) failed: res = %s\n", conv(res));
    exit(1);
  }
  

  
}
// ---------------------------- trf_OV2Dim() -------------------------------------
// -------------------------------------------------------------------------------




//---------------------------------------------------------
/*
 *  execute cleaning function in kernel, free device memory 
 *   module unload and destroy context
 */
void trf_exit2Dim(void){


  /* free device memory */
  res = cuMemFree(x_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(x) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemFree(y_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(y) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(vx_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(vx) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemFree(vy_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(vy) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(v_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(v) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(dx_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(dx) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(dy_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(dy) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(dr_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(dr) failed: res = %s\n", conv(res));
    exit(1);
  } 

  res = cuMemFree(pointer_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(pointer) failed: res = %s\n", conv(res));
    exit(1);
  }  

  res = cuMemFree(kx1_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kx1) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kx2_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kx2) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kx3_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kx3) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kx4_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kx4) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(ky1_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(ky1) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(ky2_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(ky2) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(ky3_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(ky3) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(ky4_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(ky4) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kvx1_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kvx1) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kvx2_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kvx2) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kvx3_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kvx3) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kvx4_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kvx4) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kvy1_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kvy1) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kvy2_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kvy2) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kvy3_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kvy3) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(kvy4_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(kvy4) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(sx_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(sx) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(sy_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(sy) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(svx_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(svx) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(svy_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(svy) failed: res = %s\n", conv(res));
    exit(1);
  }



  /* unload module */
  res = cuModuleUnload(module);
  if(res != CUDA_SUCCESS){
    printf("cuModuleUnload failed: res = %s\n", conv(res));
    exit(1);
    }
    
  /* destroy context */
  res = cuCtxDestroy(ctx);
  if(res != CUDA_SUCCESS){
      printf("cuCtxDestroy failed: res = %s\n", conv(res));
      exit(1);
  }

}
// -------------------- trf_exit2Dim() ---------------------
// ---------------------------------------------------------









//---------------------------------------------------------
/*
 *  parameter setting function
 */
void
parameter_set(CUfunction function){

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
  
  
  res = cuParamSeti(function, 8, y_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(function, 12, y_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuParamSeti(function, 16, vx_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(vx) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(function, 20, vx_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(vx) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  
  res = cuParamSeti(function, 24, vy_dev);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(vv) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuParamSeti(function, 28, vy_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(v) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuParamSetSize(function, 32);
  if(res != CUDA_SUCCESS){
    printf("cuParaMSetSize() failed: res = %s\n", conv(res));
    exit(1);
  }
  

}
//----------------------------------------------------------------------


//---------------------------------------------------------
/*
 *  parameter setting function for *_calc
 */
void
parameter_set_calc(CUfunction function){

  int byteoffset = 0;

  res = cuParamSeti(function, byteoffset, vx_dev);
  res = cuParamSeti(function, byteoffset+4, vx_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(vx) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(vx_dev);  
  
  res = cuParamSeti(function, byteoffset, vy_dev);
  res = cuParamSeti(function, byteoffset+4, vy_dev >> 32);
  if(res != CUDA_SUCCESS){
    printf("cuParamSeti(vy) failed: res = %s\n", conv(res));
    exit(1);
  }
  byteoffset += sizeof(vy_dev);

  res = cuParamSetSize(function, byteoffset);
  if(res != CUDA_SUCCESS){
    printf("cuParamSetSize() failed: res = %s\n", conv(res));
    exit(1);
  }
  

}
//----------------------------------------------------------------------


//---------------------------------------------------------
/*
 *  result converter function
 *   this function CUresult values(unsigned int) into messages(char*)
 */
char s[256];

char* conv(unsigned int res){
  
  switch(res){
  case 0: 
    sprintf(s, "CUDA_SUCCESS");
    break;
  case 1: 
    sprintf(s, "CUDA_ERROR_INVALID_VALUE");
    break;
  case 2: 
    sprintf(s, "CUDA_ERROR_OUT_OF_MEMORY");
    break;
  case 3: 
    sprintf(s, "CUDA_ERROR_NOT_INITIALIZED");
    break;
  case 4: 
    sprintf(s, "CUDA_ERROR_DEINITIALIZED");
    break;
  case 100: 
    sprintf(s, "CUDA_ERROR_NO_DEVICE");
    break;
  case 101: 
    sprintf(s, "CUDA_ERROR_INVALID_DEVICE");
    break;
  case 200: 
    sprintf(s, "CUDA_ERROR_INVALID_IMAGE");
    break;
  case 201: 
    sprintf(s, "CUDA_ERROR_INVALID_CONTEXT");
    break;
  case 202: 
    sprintf(s, "CUDA_ERROR_CONTEXT_ALREADY_CURRENT");
    break;
  case 205: 
    sprintf(s, "CUDA_ERROR_MAP_FAILD");
    break;
  case 206: 
    sprintf(s, "CUDA_ERROR_UNMAP_FAILED");
    break;
  case 207: 
    sprintf(s, "CUDA_ERROR_ARRAY_IS_MAPPED");
    break;
  case 208: 
    sprintf(s, "CUDA_ERROR_ALREADY_MAPPED");
    break;
  case 209: 
    sprintf(s, "CUDA_ERROR_NO_BINARY_FOR_GPU");
    break;
  case 210: 
    sprintf(s, "CUDA_ERROR_ALREADY_ACQUIRED");
    break;
  case 211: 
    sprintf(s, "CUDA_ERROR_NOT_MAPPED");
    break;
  case 300: 
    sprintf(s, "CUDA_ERROR_INVALID_SOURCE");
    break;
  case 301: 
    sprintf(s, "CUDA_ERROR_FILE_NOT_FOUND");
    break;
  case 400: 
    sprintf(s, "CUDA_ERROR_INVALID_HANDLE");
    break;
  case 500: 
    sprintf(s, "CUDA_ERROR_NOT_FOUND");
    break;
  case 600: 
    sprintf(s, "CUDA_ERROR_NOT_READY");
    break;
  case 700: 
    sprintf(s, "CUDA_ERROR_LAUNCH_FAILED");
    break;
  case 701: 
    sprintf(s, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES");
    break;
  case 702: 
    sprintf(s, "CUDA_ERROR_LAUNCH_TIMEOUT");
    break;
  case 703: 
    sprintf(s, "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING");
    break;
  case 999: 
    sprintf(s, "CUDA_ERROR_UNKNOWN");
    break;

  default:
    sprintf(s, "not defined value of CUresult");
    break;
    
  }
  
  return s;
  
}


