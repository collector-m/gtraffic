//----------------------------------------------------------------------
/* Optimal Velocity Model Simulator
 Author :Hiroshi Watanabe
 URL: http://www.phys.cs.is.nagoya-u.ac.jp/~watanabe/
 $Id: ov.cc 248 2005-12-12 17:22:03Z kaityo $

References:
[1] M. Bando, K. Hasebe, A. Nakayama, A. Shibata, and Y. Sugiyama, 
   Jpn. J. Ind. Appl. Math. 11, 203 (1994).
[2] M. Bando, K. Hasebe, A. Nakayama, A. Shibata, and Y. Sugiyama,
   Phys. Rev. E 51, 1035 (1995).
[3] T. S. Komatsu, S. Sasa, Phys. Rev. E 52 5574 (1995).
*/
//----------------------------------------------------------------------

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <fstream>

using namespace std;

// Constant Variables

const int N = 10000;         // Number of vehicles
const double dt = 0.05;   // Time Interval
const double L = 2*N;      // System Size
const double c = 2.0;

const int TIME = 200;     // Total Time to simulate
const int INTERVAL = 10;  // Time interval to observe

const double density = L/(double)N;

// Grobal Variables
double a = 1.0;      // Sensitivity
double s_time = 0;   // Simulation Time

// variable for time measuring
struct timeval tv_s, tv_f;
double for_time = 0;

// variable for recording temporally coodinate and velocity
#define RECNUM 1000
double xrec[RECNUM], vrec[RECNUM];


//----------------------------------------------------------------------
/**
 * Random Number Generator
 */
inline double
myrand(void){
  return (double)rand()/(double)RAND_MAX;
}
//----------------------------------------------------------------------
/**
 * OV function
 */
inline double
V(double dx){
  return tanh(dx - c)+tanh(c);
}
//----------------------------------------------------------------------
/**
 * Calculate Differential of x and v
 * Input: x,v
 * Output: x' = fx, v' = fv
 */
void
calcf(double x[], double v[], double fx[], double fv[]){
  
  
  double dx;
  
  
  for(int i=0;i<N;i++){
    if(i != N-1){
      dx = x[i+1] - x[i];
      if(dx < 0){ 
	dx += (double)L;
      }
    }else{
      dx = x[0] - x[N-1];
      if(dx < 0){ 
	dx += (double)L;
      }    
    }
    
    
    if(dx <0){
      printf("dx = %f\n", dx);
      cerr << "*** Invalid Time Step ***" << endl;
      exit(1);
    }
    
    
    // OV Function
    fv[i] = a * (V(dx) - v[i]);
        
  }
  for(int i=0;i<N;i++){
    fx[i] = v[i];
  }
  
}
//----------------------------------------------------------------------
/**
 * Integration Scheme
 * Euler Method (1st Order)
 */

/*-----------------------  BE NOT USED  -------------------------------*/
/*
void
integrate_Euler(double x[], double v[]){
  static double fv[N];
  static double fx[N];

  calcf(x,v,fx,fv);

  for(int i=0;i<N;i++){
    x[i] += fx[i]*dt;
    v[i] += fv[i]*dt;
  }

}
*/
//----------------------------------------------------------------------
/**
 * Integration Scheme
 * Runge-Kutta (4th Order)
 */
void
integrate_RungeKutta(double x[], double v[]){
  static double kx1[N],kv1[N];
  static double kx2[N],kv2[N];
  static double kx3[N],kv3[N];
  static double kx4[N],kv4[N];
  static double tx[N],tv[N];


  gettimeofday(&tv_s, NULL);  // time measuring
  

  
  calcf(x,v,kx1,kv1);

  
  for(int i=0;i<N;i++){
    tx[i] = x[i] +  kx1[i]*dt*0.5;
    tv[i] = v[i] +  kv1[i]*dt*0.5;
  }



  calcf(tx,tv,kx2,kv2);


  for(int i=0;i<N;i++){
    tx[i] = x[i] +  kx2[i]*dt*0.5;
    tv[i] = v[i] +  kv2[i]*dt*0.5;
  }


  calcf(tx,tv,kx3,kv3);


  for(int i=0;i<N;i++){
    tx[i] = x[i] +  kx3[i]*dt;
    tv[i] = v[i] +  kv3[i]*dt;
  }


  calcf(tx,tv,kx4,kv4);


  for(int i=0;i<N;i++){
    x[i] += (kx1[i]+2.0*kx2[i]+2.0*kx3[i]+kx4[i])/6.0*dt;
    v[i] += (kv1[i]+2.0*kv2[i]+2.0*kv3[i]+kv4[i])/6.0*dt;
  }

  gettimeofday(&tv_f, NULL);  // time measuring
  for_time += (tv_f.tv_sec - tv_s.tv_sec)*1000*1000 + (tv_f.tv_usec - tv_s.tv_usec);

}
//----------------------------------------------------------------------
/**
 * Integration
 */
void
integrate(double x[], double v[]){


  
  integrate_RungeKutta(x,v);



  
  for(int i=0;i<N;i++){
    if(x[i] > L) //Treat Periodic Boundary Condition
      x[i] -= L;
  }
  
  s_time += dt;
}
//----------------------------------------------------------------------
/**
 * Initialization
 * Putting small perturbation to the uniform flow
 * with amplitude eps.
 */
void
init(double x[], double v[]){

  const double eps = 0.01;
  double dx = L/(double)N;
  double iv = V(dx);




  for(int i=0;i<N;i++){
    //x[i] = L/(double)N*(double)i + eps*myrand();
    //v[i] = iv;
    x[i] = dx*(double)i;
    v[i] = iv;
  }
  x[0] += eps;
  v[0] = V(dx - eps);
  v[N-1] = V(dx + eps);


  cout << "# dt = " << dt << endl;
  cout << "# density = " << density << endl;
  cout << "# a = " << a << endl;
  cout << "# N = " << N << endl;
}
//----------------------------------------------------------------------
/**
 * Main Function
 */
int
main(void){

  FILE *fp = fopen("time.dat", "w");

  FILE *recfile = fopen("recfile_CPU.dat", "w");
  int rec_counter = 0;
  static double x[N],v[N];

  
  init(x,v);

  for(int i=0;s_time<TIME;i++){
    integrate(x,v);
    if(rec_counter < RECNUM){
      xrec[rec_counter] = x[0];
      vrec[rec_counter] = v[0];
      rec_counter++;
    }
    if(i%INTERVAL){
      for(int j=0;j<N;j++){
        //cout << x[j] << " " << s_time << endl;
	
	if(s_time > (double)TIME)  // for debug
	printf("%f %.2f\n", x[j], s_time);
	
	

      }
    }
  }
 
  for(int i=0; i<RECNUM; i++){
    fprintf(recfile, "%.6f %.6f\n", xrec[i], vrec[i]);
  }

  fprintf(fp, "N = %d\n", N);
  fprintf(fp, "a = %f\n", a);
  fprintf(fp, "%8.6f[micro sec]\n", for_time);

  fclose(fp);
  fclose(recfile);
  printf("# calculation done.\n");

}
//----------------------------------------------------------------------
// End of ov.cc
//----------------------------------------------------------------------
