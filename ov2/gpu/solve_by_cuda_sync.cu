#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

/*
 *  constant values
 */
__device__ int N, typeCode;
__device__ double Lx, Ly, dt, a, b, c, d, h, flow, range;

/*
 *  valiables for calculation
 */
__device__ double *v;
__device__ double *dx; // -0.5*LX<= dx <= 0.5*LX
__device__ double *dy; // -0.5*LY<= dy <= 0.5*LY 
__device__ double *dr;  
__device__ int *pointer;
__device__ double *kx1, *kx2, *kx3, *kx4;
__device__ double *ky1, *ky2, *ky3, *ky4;
__device__ double *kvx1, *kvx2, *kvx3, *kvx4;
__device__ double *kvy1, *kvy2, *kvy3, *kvy4;
__device__ double *sx, *sy, *svx, *svy;



// ---------------------------------------------------
/* assignment constant values and get device memory */
extern "C"
__global__
void
init_CUDA(double N_in,  double *v_in,  double *dx_in, double *dy_in, double *dr_in, int *pointer_in, double *kx1_in, double *kx2_in, double *kx3_in, double *kx4_in, double *ky1_in, double *ky2_in, double *ky3_in, double *ky4_in, double *kvx1_in, double *kvx2_in, double *kvx3_in, double *kvx4_in, double *kvy1_in, double *kvy2_in, double *kvy3_in, double *kvy4_in, double *sx_in, double *sy_in, double *svx_in, double *svy_in)
{
  
  int i;
  
  if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0){ // representative thread in the kernel do this function
    
    N = (int)N_in;
    
    /* get device memory and initialization */
    v = v_in;
    for(i=0; i<N; i++)
      v[i] = (double)0;
    
    dx = dx_in;
    dy = dy_in;
    dr = dr_in;
    for(i=0; i<N*N; i++){
      dx[i] = (double)0;
      dy[i] = (double)0;
      dr[i] = (double)0;
    }

    pointer = pointer_in;
    for(i=0; i<N*6; i++){
      pointer[i] = 0;
    }
    
    kx1 = kx1_in;
    kx2 = kx2_in;
    kx3 = kx3_in;
    kx4 = kx4_in;
    
    ky1 = ky1_in;
    ky2 = ky2_in;
    ky3 = ky3_in;
    ky4 = ky4_in;
    
    kvx1 = kvx1_in;
    kvx2 = kvx2_in;
    kvx3 = kvx3_in;
    kvx4 = kvx4_in;
    
    kvy1 = kvy1_in;
    kvy2 = kvy2_in;
    kvy3 = kvy3_in;
    kvy4 = kvy4_in;
    
    sx = sx_in;
    sy = sy_in;
    svx = svx_in;
    svy = svy_in;
   
    for(i=0; i<N; i++){
      kx1[i] = (double)0;
      kx2[i] = (double)0;
      kx3[i] = (double)0;
      kx4[i] = (double)0;
      
      ky1[i] = (double)0;
      ky2[i] = (double)0;
      ky3[i] = (double)0;
      ky4[i] = (double)0;
      
      kvx1[i] = (double)0;
      kvx2[i] = (double)0;
      kvx3[i] = (double)0;
      kvx4[i] = (double)0;
      
      kvy1[i] = (double)0;
      kvy2[i] = (double)0;
      kvy3[i] = (double)0;
      kvy4[i] = (double)0;
      
      sx[i] = (double)0;
      sy[i] = (double)0;
      svx[i] = (double)0;
      svy[i] = (double)0;
    }
    
  }  // --------- if(blockIdx.x == 0 && …) --------------
   
}
// ------------------------ init_CUDA() -------------------------------------------
// -------------------------------------------------------------------------------- 




// --------------------------------------------------------
// dr calculation
__device__
void 
calc_dr(double x[],double y[], int idx){
  
  for(int j=N-1;j>0;j--){
    dx[idx*N+j] = x[j]-x[idx];
    dy[idx*N+j] = y[j]-y[idx];
    //periodic boudary condition
    if(dx[idx*N+j] > Lx*0.5) {
      dx[idx*N+j] = dx[idx*N+j] -Lx;
    }else if(dx[idx*N+j] <= -Lx*0.5){
      dx[idx*N+j] = Lx + dx[idx*N+j];	
    }
    if(dy[idx*N+j] >Ly*0.5) {
      dy[idx*N+j] = dy[idx*N+j] -Ly ;
    }else if(dy[idx*N+j] < -Ly*0.5){
      dy[idx*N+j] = Ly + dy[idx*N+j];	
    } 
    dr[idx*N+j] = sqrt(dx[idx*N+j]*dx[idx*N+j]+dy[idx*N+j]*dy[idx*N+j]);
    dx[j*N+idx] = -dx[idx*N+j];
    dy[j*N+idx] = -dy[idx*N+j];
    dr[j*N+idx] = dr[idx*N+j];
  }
  
}
// ---------------------------- calc_dr() ---------------------------------
// ------------------------------------------------------------------------



// ------------------------------------------------------------------------
// -------------------------- v calculation -------------------------------
__device__
void 
calc_v(int idx, double vx[], double vy[]){
  
  v[idx]=sqrt(vx[idx]*vx[idx]+vy[idx]*vy[idx]);
}
// -------------------------- calc_v() ------------------------------------
// ------------------------------------------------------------------------



// ------------------------------------------------------------------------
// ----------------------- neighbors calculation --------------------------
__device__
void 
calc_neighbors(int idx, double vx[], double vy[]){


  double theta;
  int j;
 
  for(j=0; j<6; j++){
    pointer[idx*6+j]= -1;
  }
 

  for(j=0; j<N; j++){
    if(idx==j)continue;
    if(dr[idx*N+j] > range) continue;
    
    theta = atan2(dy[idx*N+j],dx[idx*N+j]) - atan2(vy[idx],vx[idx]); //-M_PI< theta<=M_PI
    
    if(theta > M_PI){
      theta = theta - 2.0*M_PI;
    }else if(theta <= -M_PI){
      theta = theta + 2.0*M_PI;
    }
    
    if(-M_PI/2.0 <=theta && theta< -M_PI/6.0){  //1
      
      if(pointer[idx*6+0]==-1) pointer[idx*6+0]=j;
      else if(dr[idx*N+j] < dr[idx*N+pointer[idx*6+0]]) pointer[idx*6+0]=j;
      
    }else if(-M_PI/6.0 <=theta && theta< M_PI/6.0){ //2 :進行方向±π/6
      
      if(pointer[idx*6+1]==-1) pointer[idx*6+1]=j;						
      else if(dr[idx*N+j] < dr[idx*N+pointer[idx*6+1]]) pointer[idx*6+1]=j;
      
    }else if(M_PI/6.0 <=theta && theta< M_PI/2.0){ //3
      
      if(pointer[idx*6+2]==-1) pointer[idx*6+2]=j;
      else if(dr[idx*N+j] < dr[idx*N+pointer[idx*6+2]]) pointer[idx*6+2]=j;
      
    }else if(M_PI/2.0 <=theta && theta< 5.0*M_PI/6.0){ //4
      
      if(pointer[idx*6+3]==-1) pointer[idx*6+3]=j;
      else if(dr[idx*N+j] < dr[idx*N+pointer[idx*6+3]]) pointer[idx*6+3]=j;
      
    }else if(5.0*M_PI/6.0 <=theta || theta< -5.0*M_PI/6.0){ //5
      
      if(pointer[idx*6+4]==-1) pointer[idx*6+4]=j;
      else if(dr[idx*N+j] < dr[idx*N+pointer[idx*6+4]]) pointer[idx*6+4]=j;
      
    }else if( -5.0*M_PI/6.0 <=theta && theta< -M_PI/2.0){ //6
      
      if(pointer[idx*6+5]==-1) pointer[idx*6+5]=j;
      else if(dr[idx*N+j] < dr[idx*N+pointer[idx*6+5]]) pointer[idx*6+5]=j;
      
    }
  }
  
} 
// ----------------------- calc_neighbors() ---------------------------------
// --------------------------------------------------------------------------



//---------------------------------------------------------------------------
//OV function and mathematics functions
__device__
double OV_tanh(double x){
  return (exp(2.0*x)-1.0)/(exp(2.0*x)+1.0);
}


__device__
double OV_cos(int i,int j, double vx[], double vy[]){
  if(i==j) return 0.0;
  else return (dx[i*N+j]*vx[i]+dy[i*N+j]*vy[i])/(dr[i*N+j]*v[i]);
}

__device__
double nx(int i, int j){
  if(i==j) return 0.0;
  else return dx[i*N+j]/dr[i*N+j];
}

__device__
double ny(int i, int j){
  if(i==j) return 0.0;
  return dy[i*N+j]/dr[i*N+j];
}

__device__
double OV(double dr){
  if(dr<=range) return h * (OV_tanh (b*(dr-d))+c);
  else return 0.0;
}
// -------------- OV function and mathematics functions --------------------
// ----------------------------------------------------------


// -------------------------------------------------------------------------
// -------------------- function called by rungekutta method ---------------
__device__
void 
func(double x[],double y[],double vx[], double vy[],double fx[],double fy[],double fvx[],double fvy[], int idx, double vel_x[], double vel_y[]){  
  
  double sumForceX = 0.0;
  double sumForceY = 0.0;

  int j;
  int ptr;

  /*
    calc_dr(x,y, idx);
    calc_v(idx, vel_x, vel_y);
    -------------------------------------------------------------------------
    ****************** need global synchronization here *********************
    -------------------------------------------------------------------------
  */
  
  
  // OV equation
  if(typeCode == 1){  // type-1 : interact with all
    
    for(j=0;j<N;j++){
      if(idx==j) continue;
      if(dr[idx*N+j] > range) continue;
      sumForceX += OV(dr[idx*N+j]) *(1.0+OV_cos(idx,j, vel_x, vel_y)) *0.5* nx(idx,j);
      sumForceY += OV(dr[idx*N+j]) *(1.0+OV_cos(idx,j, vel_x, vel_y)) *0.5* ny(idx,j);
    }
    
    sumForceX +=0.75*vx[idx]/v[idx]; //self-driving force
    sumForceY +=0.75*vy[idx]/v[idx]; //self-driving force	
    
    fvx[idx] = a * ( sumForceX-vx[idx] + flow);
    fvy[idx] = a * ( sumForceY-vy[idx]);
    sumForceX = 0.0;
    sumForceY = 0.0;
    
  }else if(typeCode == 2){  // type-2 : interact with 6-neighbors
    
    /*
      calc_neighbors(idx, vel_x, vel_y);
      -------------------------------------------------------------------------
      ****************** need global synchronization here *********************
      -------------------------------------------------------------------------
    */
    
    for(j=0;j<6;j++){
      if(idx==j) continue;
      if((ptr = pointer[idx*6+j]) == -1) continue;
      /*      
	      if(pointer[idx*6+j]==-1) continue;
            
	      sumForceX += OV(dr[idx*N+pointer[idx*6+j]]) * (1.0+OV_cos(idx,pointer[idx*6+j], vel_x, vel_y))*0.5 * nx(idx,pointer[idx*6+j]);
	      sumForceY += OV(dr[idx*N+pointer[idx*6+j]]) * (1.0+OV_cos(idx,pointer[idx*6+j], vel_x, vel_y))*0.5 * ny(idx,pointer[idx*6+j]);
     
	      ptr = pointer[idx*6+j];
      */    
      sumForceX += OV(dr[idx*N+ptr]) * (1.0+OV_cos(idx,ptr, vel_x, vel_y))*0.5 * nx(idx,ptr);
      sumForceY += OV(dr[idx*N+ptr]) * (1.0+OV_cos(idx,ptr, vel_x, vel_y))*0.5 * ny(idx,ptr);
    }
    
    sumForceX +=0.75*vx[idx]/v[idx]; //self-driving force
    sumForceY +=0.75*vy[idx]/v[idx]; //self-driving force	
    
    fvx[idx] = a * ( sumForceX-vx[idx] + flow);
    fvy[idx] = a * ( sumForceY-vy[idx]);	
    sumForceX = 0.0;
    sumForceY = 0.0;
    
  }
  
  fx[idx] = vx[idx];
  fy[idx] = vy[idx];
}
// ----------------------------- func() ------------------------------------
// -------------------------------------------------------------------------


extern "C"
__global__
void
const_value_set(double Lx_in, double Ly_in, double dt_in, double a_in, double b_in, double c_in, double d_in, double h_in, double flow_in, double range_in, double typeCode_in){

  /* assignment constant values */
  if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0){
    Lx = Lx_in;
    Ly = Ly_in;
    dt = dt_in;
    a = a_in;
    b = b_in;
    c = c_in;
    d = d_in;
    h = h_in;
    flow = flow_in;
    range = range_in;
    typeCode = (int)typeCode_in;
    
  }
 
}    


// -----------------------------------------------------------
//Runge_kutta solution (divided into 4 phases 8 patrs) 
// -----------------------------------------------------------
extern "C"
__global__
void
trf_phase1_calc(double x[], double y[], double vx[], double vy[])
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;

  if(idx < N){  
    calc_dr(x,y, idx);
    calc_v(idx, vx, vy);
  }
}


extern "C"
__global__
void 
trf_phase1(double x[] , double y[], double vx[], double vy[]){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;

  //-------rungekutta calculation phase 1--------------
  if(idx < N){
    func(x, y, vx, vy, kx1, ky1, kvx1, kvy1, idx, vx, vy);

    sx[idx] = x[idx] +  kx1[idx]*dt*0.5;
    sy[idx] = y[idx] +  kx1[idx]*dt*0.5;
    svx[idx] = vx[idx] +  kvx1[idx]*dt*0.5;
    svy[idx] = vy[idx] +  kvy1[idx]*dt*0.5;
  }
  //----------------------------------------------------
}


extern "C"
__global__
void
trf_phase2_calc(double vx[], double vy[])
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;
 
  if(idx < N){
    calc_dr(sx,sy, idx);
    calc_v(idx, vx, vy);
  }
}

extern "C"
__global__
void
trf_phase2(double x[], double y[], double vx[], double vy[]){

  // return;


  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;

  //-------rungekutta calculation phase 2--------------
  if(idx < N){  
    func(sx, sy, svx, svy, kx2, ky2, kvx2, kvy2, idx, vx, vy);
    
    sx[idx] = x[idx] +  kx2[idx]*dt*0.5;
    sy[idx] = y[idx] +  kx2[idx]*dt*0.5;
    svx[idx] = vx[idx] +  kvx2[idx]*dt*0.5;
    svy[idx] = vy[idx] +  kvy2[idx]*dt*0.5;
  }
  //---------------------------------------------------
}

extern "C"
__global__
void
trf_phase3_calc(double vx[], double vy[])
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;
  
  if(idx < N){
    calc_dr(sx,sy, idx);
    calc_v(idx, vx, vy);
  }
}


extern "C"
__global__
void
trf_phase3(double x[], double y[], double vx[], double vy[]){

  // return;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;


  //-------rungekutta calculation phase 3--------------
  if(idx < N){
    func(sx, sy, svx, svy, kx3, ky3, kvx3, kvy3, idx, vx, vy);
    
    sx[idx] = x[idx] +  kx3[idx]*dt;
    sy[idx] = y[idx] +  ky3[idx]*dt;
    svx[idx] = vx[idx] +  kvx3[idx]*dt;
    svy[idx] = vy[idx] +  kvy3[idx]*dt;
  }
  //---------------------------------------------------
}

extern "C"
__global__
void
trf_phase4_calc(double vx[], double vy[])
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;
  
  if(idx < N){
    calc_dr(sx,sy, idx);
    calc_v(idx, vx, vy);
  }
}


extern "C"
__global__
void
trf_phase4(double x[] , double y[], double vx[], double vy[]){

  //return;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;


  //-------rungekutta calculation phase 4--------------
  if(idx < N){
    func(sx, sy, svx, svy, kx4, ky4, kvx4, kvy4, idx, vx, vy);
    
    /* update x, y, vx, vy values */
    x[idx] += (kx1[idx]+2.0*kx2[idx]+2.0*kx3[idx]+kx4[idx])/6.0*dt;
    y[idx] += (ky1[idx]+2.0*ky2[idx]+2.0*ky3[idx]+ky4[idx])/6.0*dt;
    vx[idx] += (kvx1[idx]+2.0*kvx2[idx]+2.0*kvx3[idx]+kvx4[idx])/6.0*dt;
    vy[idx] += (kvy1[idx]+2.0*kvy2[idx]+2.0*kvy3[idx]+kvy4[idx])/6.0*dt;
    //periodic boudary condition
    if(x[idx]<0.0){
      x[idx] = x[idx] + Lx;
    }else if (x[idx]>=Lx){
      x[idx] = x[idx] - Lx;
    }
    if(y[idx]<0.0){
      y[idx] = y[idx] + Ly;
    }else if(y[idx]>=Ly){
      y[idx] = y[idx] - Ly;
    }
  }
  //------------------------------------------------------
  /*
    if(idx == 0){
    printf("Lx = %f Ly = %f dt = %f a = %f b = %f c = %f d = %f \nh = %f flow = %f range = %f typeCode = %d\n", Lx, Ly, dt, a, b, c, d, h, flow, range, typeCode);
    }*/
  
}


/*
 * this function call "calc_neighbor()" if typeCode = 2
 * (i.e. interaction = 6-neighbors)
 */
extern "C"
__global__
void
trf_calc_neighbor(double vx[], double vy[]){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (gridDim.y*blockDim.y) + j;
  
  if(idx < N && typeCode == 2)
    calc_neighbors(idx, vx, vy);
  
}
