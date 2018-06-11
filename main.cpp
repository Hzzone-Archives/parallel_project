#include <stdio.h>
#ifdef __APPLE__ 
#include <OpenCL/cl.h> 
#elif defined(__linux__) 
#include <CL/cl.h> 
#include <CL/opencl.h> 
#endif 

#include "utils.h"
#include "interpolation_cpu.h"
#include "interpolation_gpu.h"
#include <stdlib.h>
#include <time.h>
#include "ocl_macros.h"

int main(int argc, char *argv[])
{
    Image *img = new Image();
    // 以二进制方式读取文件
    FILE *file;
    if (!(file = fopen("image.dat", "rb")))
    {
        perror("fopen()");
        return NULL;
    }
    // 获取图片信息
    int res = fscanf(file, "%d %d %f %f %f\n",
                    &(img->width), &(img->height),
                    &(img->depth), &(img->angle),
                    &(img->radius));
    if (5 != res) {
        printf("获取图片信息出错!");
        return 0;
    }
    // 读取图片并转换成opencv图片对象
    int cnt = img->width*img->height;
    img->data = (float*)malloc(cnt*sizeof(float));
    fread(img->data, sizeof(float), cnt, file);
    IplImage* im = arr2img(img->data, img->height, img->width);
    cv::imwrite("原图片.bmp", cv::cvarrToMat(im));

    Mat mat1;
    Mat mat2;
    Mat mat3;
    IplImage* imsc;
    double start_time=0.0, end_time=0.0;
    double time1 = 0.0;
    double time2 = 0.0;
    double time3 = 0.0;

    int counts = 1;
    for(int i=0; i<counts; i++) {

        //////
        start_time = clock();
        imsc = ScanConvCurve_B(img);
        end_time = clock();
        time1 += double(end_time-start_time)/CLOCKS_PER_SEC;
        mat1 = cvarrToMat(imsc);


        /////
        start_time = clock();
        imsc = Inter_Linear(img);
        end_time = clock();
        time2 += double(end_time-start_time)/CLOCKS_PER_SEC;
        mat2 = cvarrToMat(imsc);

        ////

        start_time = clock();
        imsc = Bi_cubic(img);
        end_time = clock();
        time3 += double(end_time-start_time)/CLOCKS_PER_SEC;
        mat3 = cvarrToMat(imsc);

        printf("#");
    }

    printf("\n");

    printf("插值结果如下：\n");


    cv::imwrite("邻近插值.bmp", mat1);
    cv::imwrite("双线性插值.bmp", mat2);
    cv::imwrite("双三次插值.bmp", mat3);

//    printf("邻近插值的psnr：%f，ssim：%f", getPSNR(mat1, mat3), getSSIM(mat1, mat3));
//    printf("双线性插值的psnr：%f，ssim：%f", getPSNR(mat2, mat3), getSSIM(mat2, mat3));
    printf("算法\t平台\t平均时间(ms)\tSSIM\tPSNR\tMSE\n");
    printf("邻近插值\t%s\t%.2f\t\t%.2f\t%.2f\t%.2f\t\n", "CPU", time1/counts*1000.0, getSSIM(mat1, mat3), getPSNR(mat1, mat3), getMSE(mat1, mat3));
    printf("双线性插值\t%s\t%.2f\t\t%.2f\t%.2f\t%.2f\t\n", "CPU", time2/counts*1000.0, getSSIM(mat2, mat3), getPSNR(mat2, mat3), getMSE(mat2, mat3));
    printf("双三次插值\t%s\t%.2f\t\t%.2f\t%.2f\t%.2f\t\n", "CPU", time3/counts*1000.0, getSSIM(mat3, mat3), getPSNR(mat3, mat3), getMSE(mat3, mat3));

    cvReleaseImage(&im);
    cvReleaseImage(&imsc);


    cl_int status = 0;
    char kernels[2][10] = {"BiLinear", "BiNearst"};
    size_t groupSizeX = 16, groupSizeY = 16;
    size_t ResImageH, ResImageW;

    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_mem inputImageBuffer, widthBuffer, heightBuffer, angleBuffer, depthBuffer, CurProbeRadiusBuffer, outputImageBuffer;

    init(groupSizeX, groupSizeY, ResImageH, ResImageW, context, commandQueue, program, "kernel/BiLinear.cl", *img, inputImageBuffer, widthBuffer, heightBuffer, angleBuffer, depthBuffer, CurProbeRadiusBuffer, outputImageBuffer);

    cl_kernel kernel = createKernel(program, "interpolation_kernel", inputImageBuffer, widthBuffer, heightBuffer, angleBuffer, depthBuffer, CurProbeRadiusBuffer);

    // Execute the OpenCL kernel on the list
    cl_event ndrEvt;

    //2D Kernel Setting
    size_t globalThreads[] = {
        ResImageW,
        ResImageH
    };
    size_t localThreads[] = { groupSizeX, groupSizeY };
    status = clEnqueueNDRangeKernel(
        commandQueue,
        kernel,
        2,
        NULL,
        globalThreads,
        localThreads,
        0,
        NULL,
        &ndrEvt);
    LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed.");

    status = clFinish(commandQueue);
    LOG_OCL_ERROR(status, "clFinish Failed.");

    //Compute kernel's execution time
    clWaitForEvents(1, &ndrEvt);
    cl_ulong startTime, endTime;
    clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &startTime, NULL);
    clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &endTime, NULL);
    cl_ulong kernelExecTimeNs = endTime - startTime;
    printf("Local_size: %d * %d \nKernel运行时间 :%8.6f ms\n\n", groupSizeX, groupSizeY, kernelExecTimeNs*1e-6);






    status = clReleaseKernel(kernel);
    status |= clReleaseProgram(program);
    status |= clReleaseMemObject(inputImageBuffer);
    status |= clReleaseMemObject(widthBuffer);
    status |= clReleaseMemObject(heightBuffer);
    status |= clReleaseMemObject(angleBuffer);
    status |= clReleaseMemObject(depthBuffer);
    status |= clReleaseMemObject(CurProbeRadiusBuffer);
    status |= clReleaseMemObject(outputImageBuffer);
    status |= clReleaseCommandQueue(commandQueue);
    status |= clReleaseContext(context);
    LOG_OCL_ERROR(status, "clean Failed.");

    return 0;


}
