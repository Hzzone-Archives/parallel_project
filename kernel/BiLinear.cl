#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void interpolation_kernel(global float *inData, global float *outData, 
									global int* width, global int* height,
									global float* angle, global float* depth, global float* radius)
{
	int iX = get_global_id(0); //width
	int iY = get_global_id(1); //height
	
	// 参数准备
	int Lnum = *width;
	int Snum = *height;
	float Angle = *angle;
	float Depth = *depth;
	float CurProbeRadius = *radius;
	int iBHeightResolve = 1024;
	float cuEPSILON = 1e-6;
	float alpha = -0.5; 
	float ProbeRadiusPixel = CurProbeRadius * Snum / Depth;
	float SectorRadiusPixel = ProbeRadiusPixel + Snum;
	float StartAngle = -Angle;
	float EndAngle = Angle;
	float AveIntervalAngleReciprocal = (Lnum - 1) / (Angle * 2);
	int ResImageH = iBHeightResolve;
	float Ratio = ResImageH / (SectorRadiusPixel - cos(Angle)*ProbeRadiusPixel + 1);
	int ResImageW = round(sin(Angle)*SectorRadiusPixel * 2 * Ratio + 1);
	ProbeRadiusPixel = ProbeRadiusPixel*Ratio;
	SectorRadiusPixel = SectorRadiusPixel *Ratio;
	Ratio = 1 / Ratio;

	// DEBUG
	// printf("%d %d %f %f %f",Lnum,Snum,Angle,Depth,CurProbeRadius);
	// printf("%f %f %f",inData[7682],inData[9874],inData[12398]);
	

	float SampleOriginX = ResImageW / 2;
	float SampleOriginY = 0;
	float TranformHor = 0;
	float TranformVec = SectorRadiusPixel - ResImageH;
	
	float PosX = 0;
	float PosY = 0;
	int PosXint1 = 0;
	int PosYint1 = 0;
	float fHitX = 0;
	float fHitY = 0;
	float fHitPointNorm = 0;
	float fSamplePointAngle = 0;
	int sign;

	// 采样位置以交点为原点的成像位置坐标
	fHitX = iX - SampleOriginX + TranformHor;  
	fHitY = iY - SampleOriginY + TranformVec;  
	
	// 深度计算
	fHitPointNorm = sqrt(fHitX*fHitX + fHitY*fHitY);

	// 若映射后的在该圆环内
	if (fHitPointNorm > ProbeRadiusPixel - cuEPSILON &&  fHitPointNorm < SectorRadiusPixel + cuEPSILON) {
		// 符号判断
		if(fHitX>0) sign=1;
		else if(fHitX==0) sign=0;
		else sign=-1;

		// 映射后的角度计算
		fSamplePointAngle = acos(fHitY / fHitPointNorm - cuEPSILON)*sign; 
		
		// 映射后点的角度没有超出范围
		if (fSamplePointAngle > StartAngle - cuEPSILON && fSamplePointAngle < EndAngle + cuEPSILON) {
			
			PosX = (fSamplePointAngle - StartAngle)*AveIntervalAngleReciprocal;
			PosY = (fHitPointNorm - ProbeRadiusPixel)*Ratio; 
			PosXint1 = floor(PosX);
			PosYint1 = floor(PosY);

			float a = PosX-PosXint1;
			float b = PosY-PosYint1;
					
			if (PosXint1 >= 0 && PosXint1<Lnum && PosYint1 >= 0 && PosYint1<Snum)
			if(PosXint1 == 0 && PosXint1==Lnum-1 && PosYint1 == 0 && PosYint1==Snum-1 ){
				printf("reach");
			}
				outData[iX*ResImageH+iY] = inData[PosXint1*Snum+PosYint1]*(1-a)*(1-b)+
										   inData[(PosXint1+1)*Snum+PosYint1]*a*(1-b)+
										   inData[PosXint1*Snum+PosYint1+1]*(1-a)*b+
										   inData[(PosXint1+1)*Snum+PosYint1+1]*a*b;					
		}			
	}
}