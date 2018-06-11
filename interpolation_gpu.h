#ifndef INTERPOLATION_GPU_H
#define INTERPOLATION_GPU_H
#ifdef __APPLE__ 
#include <OpenCL/cl.h> 
#elif defined(__linux__) 
#include <CL/cl.h> 
#include <CL/opencl.h> 
#endif 
#include <ocl_macros.h>
#include <stdio.h>
#include <ocl_macros.h>
#include "utils.h"

void init(size_t groupSizeX, size_t groupSizeY, size_t& ResImageH, size_t& ResImageW, cl_context& context, cl_command_queue& commandQueue, cl_program& program, char* filename, Image& image, cl_mem& inputImageBuffer, cl_mem& widthBuffer, cl_mem& heightBuffer, cl_mem& angleBuffer, cl_mem& depthBuffer, cl_mem& CurProbeRadiusBuffer, cl_mem& outputImageBuffer);
cl_kernel createKernel(cl_program& program, char* kernel_name, cl_mem& inputImageBuffer, cl_mem& widthBuffer, cl_mem& heightBuffer, cl_mem& angleBuffer, cl_mem& depthBuffer, cl_mem& CurProbeRadiusBuffer);

#endif // INTERPOLATION_GPU_H
