#ifndef INTERPOLATION_CPU_H
#define INTERPOLATION_CPU_H

#include "utils.h"

// 邻近插值法转换图片
IplImage* ScanConvCurve_B(Image *img);

// 双线性插值法转换图片
IplImage* Inter_Linear(Image *img);

// 双三次插值基函数
float weights(float x);

// 双三次插值法转换图片
IplImage* Bi_cubic(Image *img);


#endif // INTERPOLATION_CPU_H
