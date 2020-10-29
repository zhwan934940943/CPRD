#define _CRT_SECURE_NO_DEPRECATE

#include <opencv2/highgui.hpp>

#include "dataset_test.h"
#include <direct.h>
#include <iostream>
#include <windows.h>
using namespace std;
void test()
{
	double k = 1.4;
	double r = 1.6;
	cout << k << "	" << r << endl;


	std::stringstream ss;
	ss << setprecision(3) << k<<r;

	system("mkdir sample");

	string folderPath = "D:\\OneDrive - Platinum\\mycodelib\\pupil detection\\results\\Haar\\r";
	string folderPath2 = "D:\\test";
	//mkdir(folderPath.c_str());
	string cmd = "mkdir -p " + folderPath;
	system(cmd.c_str());
	//if (0 != access(folderPath.c_str(), 0))
	//{
	//	 if this folder not exist, create a new one.
	//	mkdir(folderPath.c_str());   // 返回 0 表示创建成功，-1 表示失败
	//	换成 ::_mkdir  ::_access 也行，不知道什么意思
	//}

	string errorfile_name = "./r/" + ss.str() + ".txt";
	ofstream fout(errorfile_name);
	fout << k;
	fout.close();
}


int main()
{
	//test();

	DatasetTest test;
	test.LPWTest_Haar();
	//test.SwirskiTest_Haar();
	//test.samplesTest_Haar();
	//test.LPWTest_PuRe();
	
	cv::destroyAllWindows();
	return system("pause");
}