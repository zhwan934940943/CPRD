#define _CRT_SECURE_NO_DEPRECATE

#include <opencv2/opencv.hpp>

#include "dataset_test.h"


int main()
{
	DatasetTest test;
	test.LPWTest_Haar();
	//test.SwirskiTest_Haar();
	//test.samplesTest_Haar();
	//test.LPWTest_PuRe();
	
	cv::destroyAllWindows();
	return system("pause");
}