#include <stdio.h>

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
