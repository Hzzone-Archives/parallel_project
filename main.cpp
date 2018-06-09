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

int main(int argc, char *argv[])
{
    Image *img = new Image();
    // 以二进制方式读取文件
    FILE *file;
    if (!(file = fopen("/home/hzzone/parallel_project/image.dat", "rb")))
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

    int counts = 100;
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

    return 0;


}
