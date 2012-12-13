#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "defNumber.h"

/*
 *  valiables for Rungekutta method
 */
__device__
double kx1[N], kv1[N], kx2[N], kv2[N], kx3[N], kv3[N], kx4[N], kv4[N], tx[N], tv[N]; 


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


/*__device__
int global_sync=0;

__device__
void
global_synchronization(void){
  //int global_sync=0;

  if (threadIdx.x == 0 && threadIdx.y == 0){
    atomicAdd(&global_sync, 1);
    if (blockIdx.x == 0 && blockIdx.y == 0){
      while(atomicAdd(&global_sync,0) < gridDim.x*gridDim.y);//numberOfNeurons/threadsPerBlocks);
      atomicExch(&global_sync,0);
    }else{
      while(atomicAdd(&global_sync,0) > 0);
    }
  }
  __syncthreads();
  global_sync = 0;

}
*/

__device__
int sync_counter=0;

__device__
void
global_synchronization(void){

  // the number of all threads in the grid
  int thread_num = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
  
  atomicAdd(&sync_counter, 1); // increment counter exclusively
  

  while(atomicAdd(&sync_counter, 0) != thread_num);  // wait until all threads achieve here 


  if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    atomicExch(&sync_counter, 0);  // representative thread reset counter value 
  
    //__syncthreads(); 
   
}



//---------------------------------------------------------------------------------
/*
 * RungeKutta method
 *
 * NOTE:All method processes(except __syncthreads()) are done by threads which have
 * thread number(idx) under N.  Threads which have thread number over N do nothing.
 */
__device__
void 
itgRungeKutta(double *x, double *v, double a, int *error, int idx)
{



  if(idx < N)
    calcf(x, v, kx1, kv1, idx, a, error);  // calculate kx1 and kv1 values respectively
  
 
  if(idx < N){
    tx[idx] = x[idx] + kx1[idx]*dt*0.5;
    tv[idx] = v[idx] + kv1[idx]*dt*0.5;
  }

  
    
  // -----------------------
  global_synchronization();  
  // -----------------------  


  if(idx < N) 
    calcf(tx, tv, kx2, kv2, idx, a, error);   // calculate kx2 and kv2 values respectively  

  // -----------------------
  global_synchronization();  
  // -----------------------  


 
  if(idx < N){
    tx[idx] = x[idx] + kx2[idx]*dt*0.5;
    tv[idx] = v[idx] + kv2[idx]*dt*0.5;
  }



  // -----------------------
  global_synchronization();  
  // -----------------------  
  
  


  if(idx < N) 
    calcf(tx, tv, kx3, kv3, idx, a, error);   // calculate kx3 and kv3 values respectively  

  // -----------------------
  global_synchronization();  
  // -----------------------  
    
  
  if(idx < N){
    tx[idx] = x[idx] + kx3[idx]*dt;
    tv[idx] = v[idx] + kv3[idx]*dt;
  }


  // -----------------------
  global_synchronization();  
  // -----------------------  

  
  
  
  __syncthreads();
  if(idx < N) 
    calcf(tx, tv, kx4, kv4, idx, a, error);   // calculate kx4 and kv4 values respectively   
  __syncthreads();
  
  
  
  // update x and v values
  if(idx < N){
    x[idx] += (kx1[idx] + 2.0*kx2[idx] + 2.0*kx3[idx] + kx4[idx]) / 6.0 * dt;
    v[idx] += (kv1[idx] + 2.0*kv2[idx] + 2.0*kv3[idx] + kv4[idx]) / 6.0 * dt;
    if(x[idx] > L) x[idx] -= L;  //Treat Periodic Boundary Condition    
   }


}
//---------------------------------------------------------------------------------


//---------------------------------------------------------------------------------
/*
 * main function in kernel
 */
extern "C"
__global__
void
cuda_main(double *x, double *v, double a, int *error, double *s_time)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;
  
  int t;


  
  //for(t=0; *s_time<TIME; t++){
  for(t=0; t<(double)TIME/(double)dt; t++){

    //__syncthreads();  
    global_synchronization();

    /* error management */
    if (*error == 1) {
      return;
    }

    /* calculate x and v values*/
    itgRungeKutta(x, v, a, error, idx);


    /* the only thread which thread number is 0 increment s_time */
    if(idx == 0)
      *s_time += dt;
      
    //__syncthreads();  
    global_synchronization();

    }
  
}
