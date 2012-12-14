#ifndef TRAFFIC_SYNC_H
#define TRAFFIC_SYNC_H

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

  void trf_init2Dim(int N);
  /**************************************************
   * This function do...
   * ------------------------------------------------
   * 1) initialization to use CUDA kernel
   * 2) upload constant values
   * 3) execute initialization function in kernel
   ***************************************************
   * Parameters
   * -------------------------------------------------
   * N : number of object
   ****************************************************/


  //  void trf_OV2Dim(double x[], double y[], double vx[], double vy[], int N, double Lx, double Ly, double dt, double a, double b, double c, double d, double h, double flow, double range, int typeCode);
  void trf_OV2Dim(double x[], double y[], double vx[], double vy[], double Lx, double Ly, double dt, double a, double b, double c, double d, double h, double flow, double range, int typeCode);
  /*******************************************
   * This function do...
   * -----------------------------
   * 1) upload object's position and velocity
   * 2) execute calculation of simulation
   * 3) download updated position and velocity
   *********************************************
   * Parameters
   * ------------------------------------------- 
   * x[] : position_x
   * y[] : position_y
   * vx[] : velocity_x
   * vy[] : velocity_y
   * Lx : area
   * Ly : area
   * dt : time step
   * a : sensitivity
   * b, c, d, h : constant values for OV function
   * flow : 
   * range : 
   * typeCode : interaction type 
   *********************************************/  


  void trf_exit2Dim(void);
  /*********************************************
   * This function do...
   * -------------------------------------------
   * 1) execute exit function in kernel
   * 2) cleaning up after use kernel functions
   *********************************************
   * Parameter
   * -------------------------------------------
   * None
   **********************************************/
  
#ifdef __cplusplus
}
#endif /* #ifdef __cplusplus */

#endif /* #ifndef TRAFFIC_SYNC_H */
