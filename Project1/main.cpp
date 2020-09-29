#define _CRT_SECURE_NO_DEPRECATE

#include <opencv2/opencv.hpp>

#include <my_lib.h>
#include "src/pupil_extraction.h"
#include "src/dataset_test.h"
#include "PuRe/PuRe.h"
#include <QDebug>


#include <StereoReconstructor.h>



using namespace std;
using namespace cv;
using namespace mycv;


void imgTest()
{
	string dirname = libpath + "data_eye/";
	string filename = "imagelist.xml";
	vector<string> imagelist;
	readStringList(dirname + filename, imagelist);

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





void test()
{
	//Rect并集测试
	Rect a(0, 0, 10, 10);
	Rect b(10, 10, 10, 10);
	Rect c = a | b;

	StereoReconstructor stereor;
	stereor.dirname = "C:/Users/w9349/OneDrive - Platinum/mycodelib/stereoCalibration/stereo calibration20171130/";

	//双目重建测试
	stereor.alg = StereoReconstructor::STEREO_BM;
	stereor.img1_filename = stereor.dirname + "left/left01.png";
	stereor.img2_filename = stereor.dirname + "right/right01.png";
	stereor.testSparse();
	

	//双目标定测试
	//stereor.imagelistfn = "stereo_calib.xml";
	//stereor.boardSize = Size(8, 5);
	//stereor.squareSize = 20.0;
	//stereor.stereoCalib(false, true, false);
}

int main()
{
	//test();
	//stereoMatch();
	//SwirskiTest();
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