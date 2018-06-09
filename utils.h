#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;

// 图片对象的结构体
struct Image {
    float *data;
    int width, height;
    float depth, angle, radius;
    Image():width(0),height(0),depth(0.0),angle(0.0),radius(0.0){}
};


int sign(int x);

// 把float数组转换成opencv图片对象
IplImage* arr2img(float* data, int width, int height);

// 计算PSNR
double getPSNR(const Mat& I1, const Mat& I2);

// 计算SSIM
double getSSIM(const Mat& i1, Mat& i2);

double getMSE (const Mat& I1, const Mat& I2);


#endif // UTILS_H
