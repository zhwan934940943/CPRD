#define _CRT_SECURE_NO_DEPRECATE

#include <opencv2/highgui.hpp>

#include "dataset_test.h"


void test()
{
	double ratioS[] = { 1.4, 2, 3, 4, 5, 6, 7 };
	for (int i = 0; i < sizeof(ratioS) / sizeof(ratioS[0]); i++)
		cout << i << endl;
}


int main()
{
	//test();

	DatasetTest test;
	double ratioS[] = { 1.4 };
	//double ratioS[] = { 1.1,1.2,1.3,1.4,1.5, 1.6, 1.7, 1.8, 1.9,2, 3, 4, 5, 6, 7 };
	for (int i = 0; i < sizeof(ratioS) / sizeof(ratioS[0]); i++)
		test.LPWTest_Haar(ratioS[i]);

	//test.SwirskiTest_Haar();
	//test.samplesTest_Haar();
	//test.LPWTest_PuRe();

	cv::destroyAllWindows();
	return system("pause");
}