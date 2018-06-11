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
        exit(1);
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

    Mat mat;
    Mat cubic_mat;
    IplImage* imsc;
    double start_time=0.0, end_time=0.0;
    double cpu_total_time = 0.0;
    double gpu_time = 0.0;
    double cpu_time = 0.0;

    int counts = 100;

    cl_int status = 0;

    size_t groupSizeX = 16, groupSizeY = 16;
    size_t ResImageH, ResImageW;

    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_mem inputImageBuffer, widthBuffer, heightBuffer, angleBuffer, depthBuffer, CurProbeRadiusBuffer, outputImageBuffer;

    init(groupSizeX, groupSizeY, ResImageH, ResImageW, context, commandQueue, program, "kernel/BiLinear.cl", *img, inputImageBuffer, widthBuffer, heightBuffer, angleBuffer, depthBuffer, CurProbeRadiusBuffer, outputImageBuffer);

    double ssim = 0.0, mse = 0.0, psnr = 0.0;


    printf("插值结果如下：\n");


    //    cv::imwrite("双线性插值.bmp", mat2);
    //    cv::imwrite("双三次插值.bmp", mat3);

    printf("算法\t平台\t平均时间(ms)\tSSIM\tPSNR\tMSE\n");

    //双三次插值
    for (int i=0; i<counts; i++)
    {
        start_time = clock();
        imsc = Bi_cubic(img);
        end_time = clock();
        cpu_total_time += double(end_time-start_time)/CLOCKS_PER_SEC;
        cubic_mat = cvarrToMat(imsc);
        ssim += getSSIM(cubic_mat, cubic_mat);
        mse += getMSE(cubic_mat, cubic_mat);
        psnr += getMSE(cubic_mat, cubic_mat);
    }


    cv::imwrite("双三次插值_cpu.bmp", cubic_mat);


    printf("双三次插值\t%s\t%.2f\t\t%.2f\t%.2f\t%.2f\t\n", "CPU", cpu_total_time/counts*1000.0, ssim/counts, psnr/counts, mse/counts);


    ssim = 0.0, mse = 0.0, psnr = 0.0, cpu_total_time = 0.0;

    //邻近插值
    for (int i=0; i<counts; i++)
    {
        start_time = clock();
        imsc = ScanConvCurve_B(img);
        end_time = clock();
        cpu_total_time += double(end_time-start_time)/CLOCKS_PER_SEC;
        mat = cvarrToMat(imsc);
        ssim += getSSIM(mat, cubic_mat);
        mse += getMSE(mat, cubic_mat);
        psnr += getMSE(mat, cubic_mat);
    }

    cv::imwrite("邻近插值_cpu.bmp", mat);

    printf("邻近插值\t%s\t%.2f\t\t%.2f\t%.2f\t%.2f\t\n", "CPU", cpu_total_time/counts*1000.0, ssim/counts, psnr/counts, mse/counts);



    ssim = 0.0, mse = 0.0, psnr = 0.0, cpu_total_time = 0.0;

    //双线性插值
    for (int i=0; i<counts; i++)
    {
        start_time = clock();
        imsc = Inter_Linear(img);
        end_time = clock();
        cpu_total_time += double(end_time-start_time)/CLOCKS_PER_SEC;
        mat = cvarrToMat(imsc);
        ssim += getSSIM(mat, cubic_mat);
        mse += getMSE(mat, cubic_mat);
        psnr += getMSE(mat, cubic_mat);
    }

    cv::imwrite("双线性插值_cpu.bmp", mat);


    cpu_time = cpu_total_time/counts*1000.0;

    printf("双线性插值\t%s\t%.2f\t\t%.2f\t%.2f\t%.2f\t\n", "CPU", cpu_time, ssim/counts, psnr/counts, mse/counts);



    cl_kernel kernel = createKernel(program, "interpolation_kernel", inputImageBuffer, widthBuffer, heightBuffer, angleBuffer, depthBuffer, CurProbeRadiusBuffer);

    gpu_time = run_kernel("双线性插值", cubic_mat, commandQueue, kernel, outputImageBuffer, ResImageW, ResImageH, groupSizeX, groupSizeY, counts);

    printf("加速比: %4.3f\n", cpu_time/gpu_time);






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
