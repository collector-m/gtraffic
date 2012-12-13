#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "defNumber.h"

/*
 *  valiables for Rungekutta method
 */
__device__
double kx1[N], kv1[N], kx2[N], kv2[N], kx3[N], kv3[N], kx4[N], kv4[N];
__device__
double tx[N], tv[N], tx2[N], tv2[N], tx3[N], tv3[N]; 


//---------------------------------
/*
 * OV function
 */
__device__
double 
V(double dx){
  return tanh(dx - c)+tanh(c);
}
//---------------------------------


//------------------------------------------------------------------------------------
/*
 * One part of culculation of RungeKutta method
 * 
 * This function is called in the middle of itgRungeKutta() several times
 */
__device__
void 
calcf(double x[], double v[], double fx[], double fv[], int idx, double a, int *error)
{
  
  double dx;

     
  if(idx != N-1){
    dx = x[idx+1] - x[idx];

    if(dx < 0){
      dx += (double)L;
    }
    
  }else{ // idx == N-1 then 
    dx = x[0] - x[N-1];

    if(dx < 0){
      dx += (double)L;
    }
  }
  
  
  
  //error management
  if(dx < 0){
    *error = 1;
  }


  //OV Function
  fv[idx] = a * ( V(dx) - v[idx] );
  fx[idx] = v[idx];
}
//------------------------------------------------------------------------------------



//------------------------------------------------------------------------------------
/*
 * RungeKutta method
 *
 * NOTE:RungeKutta method is devided 4 phases.
 */
extern "C"
__global__
void 
RK_phase1(double *x, double *v, double a, int *error)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;

 
  if(idx < N)
    calcf(x, v, kx1, kv1, idx, a, error);  // calculate kx1 and kv1 values respectively


  if(*error == 1) return;  // error management  
 
  if(idx < N){
    tx[idx] = x[idx] + kx1[idx]*dt*0.5;
    tv[idx] = v[idx] + kv1[idx]*dt*0.5;
  }

}

// ------------------------------------------------------------------------------------------
// ----------------------  global_synchronization here  -------------------------------------
// ------------------------------------------------------------------------------------------
    
extern "C"
__global__
void
RK_phase2(double *x, double *v, double a, int *error){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;


  if(idx < N) 
    calcf(tx, tv, kx2, kv2, idx, a, error);   // calculate kx2 and kv2 values respectively  

  if(*error == 1) return;  // error management  

  if(idx < N){
    tx2[idx] = x[idx] + kx2[idx]*dt*0.5;
    tv2[idx] = v[idx] + kv2[idx]*dt*0.5;
  }

}

// ------------------------------------------------------------------------------------------
// ----------------------  global_synchronization here  -------------------------------------
// ------------------------------------------------------------------------------------------
 
extern "C"
__global__
void
RK_phase3(double *x, double *v, double a, int *error){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;


  if(idx < N) 
    calcf(tx2, tv2, kx3, kv3, idx, a, error);   // calculate kx3 and kv3 values respectively  

  if(*error == 1) return;  // error management  

  if(idx < N){
    tx3[idx] = x[idx] + kx3[idx]*dt;
    tv3[idx] = v[idx] + kv3[idx]*dt;
  }
  
}

// ------------------------------------------------------------------------------------------
// ----------------------  global_synchronization here  -------------------------------------
// ------------------------------------------------------------------------------------------

extern "C"
__global__
void
RK_phase4(double *x, double *v, double a, int *error){  

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;
  
  
  if(idx < N) 
    calcf(tx3, tv3, kx4, kv4, idx, a, error);   // calculate kx4 and kv4 values respectively   
   
  if(*error == 1) return;  // error management    

  // update x and v values
  if(idx < N){
    x[idx] += (kx1[idx] + 2.0*kx2[idx] + 2.0*kx3[idx] + kx4[idx]) / 6.0 * dt;
    v[idx] += (kv1[idx] + 2.0*kv2[idx] + 2.0*kv3[idx] + kv4[idx]) / 6.0 * dt;
    if(x[idx] > L) x[idx] -= L;  //Treat Periodic Boundary Condition    
   }

}
  
