#include "utils.h"


int sign(int x) {
    if (x > 0) return 1;
    else if (x < 0) return -1;
    else return 0;
}

// 把float数组转换成opencv图片对象
IplImage* arr2img(float* data, int width, int height) {
    IplImage* img = cvCreateImage(CvSize(width, height), 8, 1);
    for (int j = 0;j<height;j++)
    {
        for (int i = 0;i<width;i++)
            img->imageData[j*width + i] = data[j*width + i];
    }
    return img;
}

// 计算PSNR
double getPSNR(const Mat& I1, const Mat& I2) {
    Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    Scalar s = sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];
    if (sse <= 1e-10)
        return 0;
    else {
        double mse = sse / (double)(I1.channels()*I1.total());
        double psnr = 10.0*log10(255 * 255 / mse);
        return psnr;
    }

}

// 计算SSIM
double getSSIM(const Mat& i1, Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    Mat I1_2 = I1.mul(I1);
    Mat I2_2 = I2.mul(I2);
    Mat I1_I2 = I1.mul(I2);
    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigam2_2, sigam12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigam2_2, Size(11, 11), 1.5);
    sigam2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigam12, Size(11, 11), 1.5);
    sigam12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigam12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigam2_2 + C2;
    t1 = t1.mul(t2);

    Mat ssim_map;
    divide(t3, t1, ssim_map);
    Scalar mssim = mean(ssim_map);

    double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3;
    return ssim;
}

// 计算MSE
double getMSE (const Mat& I1, const Mat& I2){
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);        // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        return mse;
    }
}
