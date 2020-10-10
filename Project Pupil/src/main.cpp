#define _CRT_SECURE_NO_DEPRECATE

#include <opencv2/opencv.hpp>

#include <my_lib.h>

#include "m_pupil_extraction.h"
#include "dataset_test.h"



using namespace std;
using namespace cv;
using namespace mycv;


void imgTest()
{
	string dirname = "samples/";
	string filename = "imagelist.xml";
	vector<string> imagelist;
	readStringList_xml(dirname + filename, imagelist);

	for (int i = 0; i < imagelist.size(); ++i)//imagelist.size()
	{
		Mat img;
		cout << endl << imagelist[i] << endl;
		measureTime([&]()
		{
			img = imread(dirname + imagelist[i]);
		}, "imread\t");

		PupilExtractionMethod detector;
		measureTime([&]() {detector.detect(img); }, "detector\t");
		imshow("img", img);
		waitKey(2000);
	}
}

void imgTest2()
{
	string dirname = "C:/KernelData/0 code lib/pupil datasets/Swiriski datasets/p1-left/frames/";
	VideoCapture imgcap(dirname + "0-eye.png");//read从当前帧开始读
	if (!imgcap.isOpened())
		throw("cann't open img capture!");
	Mat frame;
	while (imgcap.read(frame))
	{
		PupilExtractionMethod detector;
		measureTime([&]() {detector.detect(frame); }, "detector\t");
		waitKey(1000);
	}


}



int main()
{
	//imgTest();
	DatasetTest test;
	//test.SwirskiTest_Haar();
	//test.LPWTest_Haar_iris();
	test.LPWTest_Haar();
	//test.LPWTest_PuRe();
	//test.PupilnetDatasetTest_Haar();

	//test.HaarTest();
	destroyAllWindows();
	return system("pause");
}