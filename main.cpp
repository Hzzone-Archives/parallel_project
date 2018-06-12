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

//    int counts = 100;
    int counts = 5;

    cl_int status = 0;

//    size_t groupSizeX = 1, groupSizeY = 1;
//    size_t groupSizeX = 2, groupSizeY = 2;
//    size_t groupSizeX = 4, groupSizeY = 4;
    size_t groupSizeX = 8, groupSizeY = 8;
//    size_t groupSizeX = 16, groupSizeY = 16;
//    size_t groupSizeX = 32, groupSizeY = 32;
    size_t ResImageH, ResImageW;

    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_mem memObjects[16];//内存对象

    cl_kernel kernel;

    init(groupSizeX, groupSizeY, ResImageH, ResImageW, context, commandQueue, program, "kernel.cl", img, memObjects);

    double ssim = 0.0, mse = 0.0, psnr = 0.0;


    printf("插值结果如下：\n");


    //    cv::imwrite("双线性插值.bmp", mat2);
    //    cv::imwrite("双三次插值.bmp", mat3);

    printf("算法\t平台\t平均时间(ms)\tSSIM\tPSNR\tMSE\n");

//    printf("%d %d %f %f %f\n", img->width, img->height, img->angle, img->depth, img->radius);

//    for (int i=0;i<100;i++)
//        printf("%f\n", img->data[i]);


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

    cpu_time = cpu_total_time/counts*1000.0;
    printf("双三次插值\t%s\t%.2f\t\t%.2f\t%.2f\t%.2f\t\n", "CPU", cpu_time, ssim/counts, psnr/counts, mse/counts);

    kernel = createKernel(program, "Bi_cubic", memObjects);

    gpu_time = run_kernel("双三次插值", cubic_mat, commandQueue, kernel, memObjects[15], ResImageW, ResImageH, groupSizeX, groupSizeY, counts);

    printf("加速比: %4.3f\n", cpu_time/gpu_time);



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

    cpu_time = cpu_total_time/counts*1000.0;

    printf("邻近插值\t%s\t%.2f\t\t%.2f\t%.2f\t%.2f\t\n", "CPU", cpu_time, ssim/counts, psnr/counts, mse/counts);

    kernel = createKernel(program, "ScanConvCurve_B", memObjects);

    gpu_time = run_kernel("邻近插值", cubic_mat, commandQueue, kernel, memObjects[15], ResImageW, ResImageH, groupSizeX, groupSizeY, counts);

    printf("加速比: %4.3f\n", cpu_time/gpu_time);



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



    kernel = createKernel(program, "Inter_Linear", memObjects);

    gpu_time = run_kernel("双线性插值", cubic_mat, commandQueue, kernel, memObjects[15], ResImageW, ResImageH, groupSizeX, groupSizeY, counts);

    printf("加速比: %4.3f\n", cpu_time/gpu_time);






    status = clReleaseKernel(kernel);
    status |= clReleaseProgram(program);
    for (int i = 0; i < 16; i++) {
        clReleaseMemObject(memObjects[i]);
    }
    status |= clReleaseCommandQueue(commandQueue);
    status |= clReleaseContext(context);
    LOG_OCL_ERROR(status, "clean Failed.");

    return 0;


}
