#include "interpolation_cpu.h"

// 邻近插值法转换图片
IplImage* ScanConvCurve_B(Image *img) {
    int iSmapleNum = img->height;
    int iLine = img->width;
    int Lnum = iLine;
    int Snum = iSmapleNum;

    int iBHeightResolve = 1024; // 预定义的图像高度
    float cuEPSILON = 1e-6;
    float alpha = -0.5; // cubic 插值系数 : -1, -0.75 or -0.5.
    float ProbeRadiusPixel = img->radius * Snum / img->depth;
    float SectorRadiusPiexl = ProbeRadiusPixel + Snum;  // 扇区半径的像素数
    float StartAngle = -img->angle, EndAngle = img->angle;
    float AveIntervalAngleReciprocal = (Lnum - 1) / (img->angle * 2); // 单位角度有多少条数据
    int ResImageH = iBHeightResolve;
    float Ratio = ResImageH / (SectorRadiusPiexl - cos(img->angle)*ProbeRadiusPixel + 1); // 定义的高度/实际高度的像素数=缩放比例
    int ResImageW = round(sin(img->angle)*SectorRadiusPiexl * 2 * Ratio + 1); // 图像宽度
    ProbeRadiusPixel = ProbeRadiusPixel*Ratio;    // 按比例缩放后的探测半径
    SectorRadiusPiexl = SectorRadiusPiexl *Ratio; // 按比例缩放后的扇区半径
    Ratio = 1 / Ratio;   // 原高度/定义高度

    // 坐标转化参数
    int SampleOriginX = ResImageW / 2;
    int SampleOriginY = 0;
    int TranformHor = 0;
    int TranformVec = SectorRadiusPiexl - ResImageH; // 扇区半径-图像高度

    float *SCRes = new float[ResImageH * ResImageW];
    for (int iX = 0; iX < ResImageW; iX++) {
        for (int iY = 0; iY < ResImageH; iY++) {
            int fHitX = iX - SampleOriginX + TranformHor;// 采样位置以交点为原点的成像位置x
            int fHitY = iY - SampleOriginY + TranformVec; // 采样位置以交点为原点的成像位置y
            // Depth
            float fHitPointNorm = sqrt(fHitX*fHitX + fHitY*fHitY); // 距交点距离
            // 若映射后的在该圆环内
            if (fHitPointNorm > ProbeRadiusPixel - cuEPSILON &&  fHitPointNorm < SectorRadiusPiexl + cuEPSILON) {
                // 映射后的点的角度
                float fSamplePointAngle = acos(fHitY / fHitPointNorm - cuEPSILON)*sign(fHitX); // Cos求角度
                // 若映射后的点的角度没有超出范围
                if (fSamplePointAngle > StartAngle - cuEPSILON && fSamplePointAngle < EndAngle + cuEPSILON) {
                    // CublicSample
                    float PosX = (fSamplePointAngle - StartAngle)*AveIntervalAngleReciprocal; // 第几条数据
                    float PosY = (fHitPointNorm - ProbeRadiusPixel)*Ratio;  // 第几条数据的第几个数据
                    int PosXint1 = PosX - floor(PosX) > 0.5 ? ceil(PosX) : floor(PosX);
                    int PosYint1 = PosY - floor(PosY) > 0.5 ? ceil(PosY) : floor(PosY);
                    // 若转换后该点区间在范围内
                    if(PosXint1 >= 0 && PosXint1<iLine && PosYint1 >= 0 && PosYint1<iSmapleNum)
                        SCRes[iY*ResImageW + iX] = (img->data)[PosXint1*img->height+PosYint1];
                }
            }
        }
    }
    return arr2img(SCRes, ResImageW, ResImageH);
}

// 双线性插值法转换图片
IplImage* Inter_Linear(Image *img) {
    int iSmapleNum = img->height;
    int iLine = img->width;
    int Lnum = iLine;
    int Snum = iSmapleNum;

    int iBHeightResolve = 1024; // 预定义的图像高度
    float cuEPSILON = 1e-6;
    float alpha = -0.5; // cubic 插值系数 : -1, -0.75 or -0.5.
    float ProbeRadiusPixel = img->radius * Snum / img->depth;
    float SectorRadiusPiexl = ProbeRadiusPixel + Snum;  // 扇区半径的像素数
    float StartAngle = -img->angle, EndAngle = img->angle;
    float AveIntervalAngleReciprocal = (Lnum - 1) / (img->angle * 2); // 单位角度有多少条数据
    int ResImageH = iBHeightResolve;
    float Ratio = ResImageH / (SectorRadiusPiexl - cos(img->angle)*ProbeRadiusPixel + 1); // 定义的高度/实际高度的像素数=缩放比例
    int ResImageW = round(sin(img->angle)*SectorRadiusPiexl * 2 * Ratio + 1); // 图像宽度
    ProbeRadiusPixel = ProbeRadiusPixel*Ratio;    // 按比例缩放后的探测半径
    SectorRadiusPiexl = SectorRadiusPiexl *Ratio; // 按比例缩放后的扇区半径
    Ratio = 1 / Ratio;   // 原高度/定义高度

    // 坐标转化参数
    int SampleOriginX = ResImageW / 2;
    int SampleOriginY = 0;
    int TranformHor = 0;
    int TranformVec = SectorRadiusPiexl - ResImageH; // 扇区半径-图像高度

    float *SCRes = new float[ResImageH * ResImageW];
    int count = 0;
    for (int iX = 0; iX < ResImageW; iX++) {
        for (int iY = 0; iY < ResImageH; iY++) {
            int fHitX = iX - SampleOriginX + TranformHor;// 采样位置以交点为原点的成像位置x
            int fHitY = iY - SampleOriginY + TranformVec; // 采样位置以交点为原点的成像位置y
                                                          // Depth
            float fHitPointNorm = sqrt(fHitX*fHitX + fHitY*fHitY); // 距交点距离
                                                                   // 若映射后的在该圆环内
            if (fHitPointNorm > ProbeRadiusPixel - cuEPSILON &&  fHitPointNorm < SectorRadiusPiexl + cuEPSILON) {
                // 映射后的点的角度
                float fSamplePointAngle = acos(fHitY / fHitPointNorm - cuEPSILON)*sign(fHitX); // Cos求角度
                                                                                               // 若映射后的点的角度没有超出范围
                if (fSamplePointAngle > StartAngle - cuEPSILON && fSamplePointAngle < EndAngle + cuEPSILON) {
                    // CublicSample
                    float PosX = (fSamplePointAngle - StartAngle)*AveIntervalAngleReciprocal; // 第几条数据
                    float PosY = (fHitPointNorm - ProbeRadiusPixel)*Ratio;  // 第几条数据的第几个数据
                    int PosXint1 = floor(PosX);
                    int PosYint1 = floor(PosY);
                    float u = PosX - PosXint1;
                    float v = PosY - PosYint1;
                    // 若转换后该点区间在范围内
                    if (PosXint1 >= 0 && PosXint1 < iLine - 1 && PosYint1 >= 0 && PosYint1 < iSmapleNum - 1) {
                        SCRes[iY*ResImageW + iX] = (1 - u)*(1 - v)*(img->data)[PosXint1*img->height + PosYint1] +
                            (1 - u)*v*(img->data)[PosXint1*img->height + (PosYint1 + 1)] +
                            u*(1 - v)*(img->data)[(PosXint1 + 1)*img->height + PosYint1] +
                            u*v*(img->data)[(PosXint1 + 1)*img->height + (PosYint1 + 1)];
                    }
                    if(PosXint1 == iLine-1 || PosYint1==iSmapleNum-1)
                        SCRes[iY*ResImageW + iX] = (img->data)[PosXint1*img->height + PosYint1];
                }
            }
        }
    }
    return arr2img(SCRes, ResImageW, ResImageH);
}

// 双三次插值基函数
float weights(float x) {
    float abs_x = abs(x);//取x的绝对值
    float a = -0.5;
    if (abs_x <= 1.0)
        return (a + 2)*pow(abs_x, 3) - (a + 3)*pow(abs_x, 2) + 1;
    else if (abs_x <= 2.0)
        return a*pow(abs_x, 3) - 5 * a*pow(abs_x, 2) + 8 * a*abs_x - 4 * a;
    else
        return 0.0;
}

// 双三次插值法转换图片
IplImage* Bi_cubic(Image *img) {
    int iSmapleNum = img->height;
    int iLine = img->width;
    int Lnum = iLine;
    int Snum = iSmapleNum;

    int iBHeightResolve = 1024; // 预定义的图像高度
    float cuEPSILON = 1e-6;
    float alpha = -0.5; // cubic 插值系数 : -1, -0.75 or -0.5.
    float ProbeRadiusPixel = img->radius * Snum / img->depth;
    float SectorRadiusPiexl = ProbeRadiusPixel + Snum;  // 扇区半径的像素数
    float StartAngle = -img->angle, EndAngle = img->angle;
    float AveIntervalAngleReciprocal = (Lnum - 1) / (img->angle * 2); // 单位角度有多少条数据
    int ResImageH = iBHeightResolve;
    float Ratio = ResImageH / (SectorRadiusPiexl - cos(img->angle)*ProbeRadiusPixel + 1); // 定义的高度/实际高度的像素数=缩放比例
    int ResImageW = round(sin(img->angle)*SectorRadiusPiexl * 2 * Ratio + 1); // 图像宽度
    ProbeRadiusPixel = ProbeRadiusPixel*Ratio;    // 按比例缩放后的探测半径
    SectorRadiusPiexl = SectorRadiusPiexl *Ratio; // 按比例缩放后的扇区半径
    Ratio = 1 / Ratio;   // 原高度/定义高度

                         // 坐标转化参数
    int SampleOriginX = ResImageW / 2;
    int SampleOriginY = 0;
    int TranformHor = 0;
    int TranformVec = SectorRadiusPiexl - ResImageH; // 扇区半径-图像高度

    float *SCRes = new float[ResImageH * ResImageW];
    int count = 0;
    for (int iX = 0; iX < ResImageW; iX++) {
        for (int iY = 0; iY < ResImageH; iY++) {
            int fHitX = iX - SampleOriginX + TranformHor;// 采样位置以交点为原点的成像位置x
            int fHitY = iY - SampleOriginY + TranformVec; // 采样位置以交点为原点的成像位置y
                                                          // Depth
            float fHitPointNorm = sqrt(fHitX*fHitX + fHitY*fHitY); // 距交点距离
                                                                   // 若映射后的在该圆环内
            if (fHitPointNorm > ProbeRadiusPixel - cuEPSILON &&  fHitPointNorm < SectorRadiusPiexl + cuEPSILON) {
                // 映射后的点的角度
                float fSamplePointAngle = acos(fHitY / fHitPointNorm - cuEPSILON)*sign(fHitX); // Cos求角度
                                                                                               // 若映射后的点的角度没有超出范围
                if (fSamplePointAngle > StartAngle - cuEPSILON && fSamplePointAngle < EndAngle + cuEPSILON) {
                    // CublicSample
                    float PosX = (fSamplePointAngle - StartAngle)*AveIntervalAngleReciprocal; // 第几条数据
                    float PosY = (fHitPointNorm - ProbeRadiusPixel)*Ratio;  // 第几条数据的第几个数据
                    int PosXint1 = floor(PosX);
                    int PosYint1 = floor(PosY);
                    float w[16];
                    float res = 0;
                    // 若转换后该点区间在范围内
                    if (PosXint1 >= 1 && PosXint1 < iLine-2 && PosYint1 >= 1 && PosYint1 < iSmapleNum-2) {
                        for (int i = -1;i < 3;i++) {
                            for (int j = -1;j < 3;j++) {
                                res += weights(PosX - (PosXint1+i))*weights(PosY - (PosYint1+j)) *
                                    (img->data)[(PosXint1+i)*img->height + PosYint1+j];
                            }
                        }
                        SCRes[iY*ResImageW + iX] = res > 0 ? res : 0;
                    }
                    else {
                        SCRes[iY*ResImageW + iX] = (img->data)[PosXint1*img->height + PosYint1];
                    }
                }
            }
        }
    }
    return arr2img(SCRes, ResImageW, ResImageH);
}

