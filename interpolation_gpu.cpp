#include "interpolation_gpu.h"

void init(size_t groupSizeX, size_t groupSizeY, size_t& ResImageH, size_t& ResImageW, cl_context& context, cl_command_queue& commandQueue, cl_program& program, char* filename, Image* img, cl_mem* memObjects)
{
    cl_int status = 0;
    cl_device_type dType = CL_DEVICE_TYPE_GPU;
    cl_platform_id *platforms = NULL;
    cl_device_id   device;
    cl_uint     num_platforms;


    //Setup the OpenCL Platform,
    //Get the first available platform. Use it as the default platform
    status = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platforms);
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    LOG_OCL_ERROR(status, "clGetPlatformIDs Failed.");

    //Get the first available device
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    LOG_OCL_ERROR(status, "clGetDeviceIDs Failed.");

    //Query Platform Information
    char queryBuffer[1024];
    status = clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, 1024, &queryBuffer, NULL);
    LOG_OCL_ERROR(status, "clGetPlatformInfo Failed.");
    printf("clGetPlatformInfo: %s\n", queryBuffer);

    //Query Device Information
    cl_device_type deviceType;
    char deviceName[1024], deviceVendor[1024];
    status = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 1024, &deviceVendor, NULL);
    printf("CL_DEVICE_VENDOR: %s\n", deviceVendor);
    status = clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, &deviceName, NULL);
    printf("CL_DEVICE_NAME: %s\n", deviceName);
    status = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL);
    printf("CL_DEVICE_TYPE: %d\n", deviceType);

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms[0],
        0
    };
    context = clCreateContextFromType(
        cps,
        dType,
        NULL,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateContextFromType Failed.");

//    return context;

    // Create command queue
    commandQueue = clCreateCommandQueue(context,
        device,
        CL_QUEUE_PROFILING_ENABLE,
        &status);
    LOG_OCL_ERROR(status, "clCreateCommandQueue Failed.");


    int iSmapleNum = img->height;
    int iLine = img->width;
    int Lnum = iLine;
    int Snum = iSmapleNum;

    int iBHeightResolve = 1024; // 预定义的图像高度
    //    float cuEPSILON = 1e-6;
    //    float alpha = -0.5; // cubic 插值系数 : -1, -0.75 or -0.5.
    float ProbeRadiusPixel = img->radius * Snum / img->depth;
    float SectorRadiusPiexl = ProbeRadiusPixel + Snum;  // 扇区半径的像素数
    float StartAngle = -img->angle, EndAngle = img->angle;
    float AveIntervalAngleReciprocal = (Lnum - 1) / (img->angle * 2); // 单位角度有多少条数据
    ResImageH = iBHeightResolve;
    float Ratio = ResImageH / (SectorRadiusPiexl - cos(img->angle)*ProbeRadiusPixel + 1); // 定义的高度/实际高度的像素数=缩放比例
    ResImageW = round(sin(img->angle)*SectorRadiusPiexl * 2 * Ratio + 1); // 图像宽度
    ProbeRadiusPixel = ProbeRadiusPixel*Ratio;    // 按比例缩放后的探测半径
    SectorRadiusPiexl = SectorRadiusPiexl *Ratio; // 按比例缩放后的扇区半径
    Ratio = 1 / Ratio;   // 原高度/定义高度

    // 坐标转化参数
    int X = ResImageW / 2;
    int Y = 0;
    int TranformHor = 0;
    int TranformVec = SectorRadiusPiexl - ResImageH; // 扇区半径-图像高度

    float *SCRes = new float[ResImageH * ResImageW];

    //创建内存对象
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * iSmapleNum * iLine, img->data, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(int), &iSmapleNum, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(int), &iLine, NULL);
    memObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(int), &ResImageW, NULL);
    memObjects[4] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(int), &ResImageH, NULL);
    memObjects[5] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(int), &X, NULL);
    memObjects[6] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(int), &Y, NULL);
    memObjects[7] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(int), &TranformHor, NULL);
    memObjects[8] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(int), &TranformVec, NULL);
    memObjects[9] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float), &SectorRadiusPiexl, NULL);
    memObjects[10] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float), &ProbeRadiusPixel, NULL);
    memObjects[11] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float), &StartAngle, NULL);
    memObjects[12] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float), &EndAngle, NULL);
    memObjects[13] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float), &AveIntervalAngleReciprocal, NULL);
    memObjects[14] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float), &Ratio, NULL);
    memObjects[15] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    sizeof(float) * ResImageH * ResImageW, NULL, NULL);



//    status = clFinish(commandQueue);
//    LOG_OCL_ERROR(status, "clFinish Failed while writing the image data and parameters.");


    ifstream kernelFile(filename, ios::in);
    if (!kernelFile.is_open())
    {
        cerr << "Failed to open file for reading: " << filename << endl;
    }
    ostringstream oss;
    oss << kernelFile.rdbuf();
    string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1,
        (const char **)&srcStr, NULL, &status);
    LOG_OCL_ERROR(status, "clCreateProgramWithSource Failed.");

    // Build the program
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        printf("%s", buildLog);
        clReleaseProgram(program);
    }


}

cl_kernel createKernel(cl_program& program, char* kernel_name, cl_mem* memObjects)
{
    cl_int status = 0;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &status);
    LOG_OCL_ERROR(status, "clCreateKernel interpolation_kernel Failed.");

    // Set the arguments of the kernel
//    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&inputImageBuffer);
//    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&outputImageBuffer);
//    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&widthBuffer);
//    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&heightBuffer);
//    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&angleBuffer);
//    status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&depthBuffer);
//    status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&CurProbeRadiusBuffer);
    for (int i = 0; i < 16; i++) {
        status |= clSetKernelArg(kernel, i, sizeof(cl_mem), &memObjects[i]);
    }

    LOG_OCL_ERROR(status, "clSetKernelArg Failed.");

    return kernel;

}

double run_kernel(char* msg, Mat& org_mat, cl_command_queue& commandQueue, cl_kernel& kernel, cl_mem& outputImageBuffer, size_t ResImageW, size_t ResImageH, size_t groupSizeX, size_t groupSizeY, int counts)
{
    cl_int status = 0;
    //2D Kernel Setting
    size_t globalThreads[] = {
        ResImageW,
        ResImageH
    };
    size_t localThreads[] = { groupSizeX, groupSizeY };
    double kernelExecTimeNs = 0.0;

    float res[ResImageH*ResImageW];
    float ssim = 0.0, psnr = 0.0, mse = 0.0;

    Mat mat;

    for (int i=0; i<counts; i++) {
        // Execute the OpenCL kernel on the list
        cl_event ndrEvt;

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
        kernelExecTimeNs += endTime - startTime;

        //Read the Data back into the host memory.
        cl_event readEvt;
        status = clEnqueueReadBuffer(
        commandQueue,
        outputImageBuffer,
        CL_TRUE,
        0,
        ResImageH * ResImageW* sizeof(float),
        res,
        0,
        NULL,
        &readEvt);
        LOG_OCL_ERROR(status, "clEnqueueReadBuffer of outputImg Failed.");

        status = clWaitForEvents(1, &readEvt);
        //status = clFinish(commandQueue);
        LOG_OCL_ERROR(status, "clWaitForEvents for readEvt.");

        IplImage* img = arr2img(res, ResImageW, ResImageH);
//        printf("%d %d %d %d\n", img->width, img->height, org_mat.cols, org_mat.rows);

        mat = cvarrToMat(img);
//        printf("%d %d %d %d\n", ResImageW, ResImageH, org_mat.cols, org_mat.rows);

        ssim += getSSIM(mat, org_mat);
        psnr += getPSNR(mat, org_mat);
        mse += getMSE(mat, org_mat);

    }


    printf("%s\t%s\t%.2f\t\t%.2f\t%.2f\t%.2f\t\n", msg, "GPU", kernelExecTimeNs*1e-6/counts, ssim/counts, psnr/counts, mse/counts);

    char file_name [80];
    strcpy (file_name, msg);
    strcat (file_name, "_gpu.bmp");

    cv::imwrite(file_name, mat);

    return kernelExecTimeNs*1e-6/counts;


}


