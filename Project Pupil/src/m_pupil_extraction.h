#pragma once

#ifndef M_PupilExtraction_H_
#define M_PupilExtraction_H_


#include <QString>
#include <bitset>
#include <iostream>
#include <numeric>
#include <random>

#include <opencv2/opencv.hpp>

#include <my_lib.h>
#include "m_PupilDetectorHaar.h"





using namespace std;
//using namespace cv;
using namespace mycv;
//using namespace mymath;


class PupilExtractionMethod
{
public:
	void detect(Mat& img_in)
	{
		checkImg(img_in);
		Mat img_gray, img_BGR;
		img2GrayBGR(img_in, img_gray, img_BGR);

		section("1 detect pupil region");
		PupilDetectorHaar haar;
		haar.ratio_outer_ = 2;
		haar.kf_ = 1;
		haar.useSquareHaar_ = false;
		haar.useInitRect_ = false;

		haar.detect(img_gray);

		haar.drawCoarse(img_BGR);
		rectangle(img_BGR, haar.pupil_rect_fine_, BLUE, 1, 8);

		Rect pupil_rect = haar.pupil_rect_fine_;
		Mat img_pupil = Mat(img_gray, pupil_rect);

		//1.2 smoothing
		Mat img_blur;
		measureTime([&]() {filterImg(img_pupil, img_blur); }, "filter\t");
		img_pupil = img_blur;

		//1.3 accurate region with mask
		//这种方法更符合人的直觉
		Mat img_bw, img_bw_glint;
		measureTime([&]()
		{
			detectPupilMask(img_pupil, img_bw, 1);
			detectGlintMask(img_pupil, img_bw_glint);
			img_bw = img_bw - img_bw_glint;
		}, "bw\t");

		section("2 detect pupil contour with Canny");
		Mat edges, edges_;
		Rect inlinerRect = haar.pupil_rect_fine_ - haar.pupil_rect_fine_.tl();
		Rect inlinerRect2 = (haar.pupil_rect_fine_&haar.pupil_rect_coarse_) - haar.pupil_rect_fine_.tl();

		measureTime([&]() {detectPupilContour(img_pupil, edges, inlinerRect);
		edges_ = edges & img_bw;
		edges_ = edges_ > 200; //仅使用强edge，可以减少部分outliers
		}, "canny\t");


		//contour 过滤，去除小的curves
		vector<cv::Vec4i> hierarchy;
		vector<vector<Point> > curves;
		findContours(edges_, curves, hierarchy, CV_RETR_LIST,
			CV_CHAIN_APPROX_TC89_KCOS);
		//CV_CHAIN_APPROX_TC89_KCOS
		//CV_CHAIN_APPROX_SIMPLE
		vector<vector<Point> > candidates;
		Mat edgesFiltered = Mat::zeros(edges_.size(), CV_8UC1);
		for (int i = 0; i < curves.size(); i++)
		{
			if (curves[i].size() > 5)
			{
				candidates.push_back(curves[i]);
				for (int j = 0; j < curves[i].size(); j++)
				{
					edgesFiltered.at<uchar>(curves[i][j]) = 255;
				}
			}
		}
		edges_ = edgesFiltered;

		Mat dst;
		showGradient(edges_, dst);


		section("3 filter and fit pupil ellipse with RANSAC");
		int K;//iterations
		{
			double p = 0.99;	// success rate 0.99
			double e = 0.6;		// outlier ratio, 0.7效果很好，但是时间长
			K = cvRound(log(1 - p) / log(1 - pow(1 - e, 5)));
		}
		measureTime([&]() {fitPupilEllipse2(edges_, inlinerRect,
			ellipse_rect, K);
		ellipse_rect.center = ellipse_rect.center + Point2f(pupil_rect.x, pupil_rect.y);
		ellipse(img_BGR, ellipse_rect, RED);
		}, "fitpupil\t");


		namedWindow("edges");
		namedWindow("edges+");

		moveWindow("edges", 0, 550);
		moveWindow("edges+", 600, 550);

		imshow("edges", edges);
		imshow("edges+", edges_);
	}



	void detectPupilMask(const Mat& img_gray, Mat& img_bw, int illuminationFlag = 0)
	{
		if (!illuminationFlag)
		{
			//利用中间区域确定二值化阈值
			int thresh;
			{
				Mat img_tmp = Mat(img_gray, rectScale(Rect(0, 0, img_gray.cols,
					img_gray.rows), 1));
				Mat hist;
				int binwidth = 1;
				//不考虑亮斑
				histogram(img_tmp, hist, Range(0, 200), 200 / binwidth, Mat(), 1);
				float eta_best;
				int thresh_otsu = otsu(hist, eta_best);
				thresh = thresh_otsu * binwidth;

				//上述操作与下列等效
				/*Mat dst;
				float t = threshold(img_tmp, dst, 125, 255, THRESH_OTSU);*/
			}
			threshold(img_gray, img_bw, thresh, 255, cv::THRESH_BINARY_INV);

			Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(9, 9));


			//小亮斑的孔洞填充
			cv::morphologyEx(img_bw, img_bw, cv::MORPH_CLOSE, kernel);
			//erode消除睫毛影响
			cv::erode(img_bw, img_bw, kernel, Point(-1, -1), 2);

			//确定包含pupil的区域
			Mat labels, stats, centroids;
			cv::connectedComponentsWithStats(img_bw, labels, stats, centroids);
			Mat area = stats.col(cv::CC_STAT_AREA);
			area.at<int>(0, 0) = 0;
			Point maxLoc;
			double maxval;
			mymax(area, maxLoc, maxval);
			img_bw = labels == maxLoc.y;

			//膨胀区域以增强检测稳定性
			Mat img_bw_tmp;
			dilate(img_bw, img_bw_tmp, kernel, Point(-1, -1), 3);//3

			img_bw = img_bw_tmp - img_bw;
		}
		else //处理光照非均匀
		{
			int thresh;
			{
				Mat img_tmp = Mat(img_gray, rectScale(Rect(0, 0, img_gray.cols,
					img_gray.rows), 1));
				Mat hist;
				int binwidth = 1;
				//不考虑亮斑
				histogram(img_tmp, hist, Range(0, 256), 256 / binwidth, Mat(), 1);
				float eta_best;
				int thresh_otsu = otsu(hist, eta_best);
				thresh = thresh_otsu * binwidth;

				//上述操作与下列等效
				/*Mat dst;
				float t = threshold(img_tmp, dst, 125, 255, THRESH_OTSU);*/
			}
			threshold(img_gray, img_bw, thresh, 255, cv::THRESH_BINARY_INV);
			Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
			dilate(img_bw, img_bw, kernel, Point(-1, -1), 1);

			//threshold(img_gray, img_bw, 200, 255, THRESH_BINARY_INV);
		}
	}

	void detectGlintMask(const Mat& img_gray, Mat& img_bw)
	{
		int thresh = 200;
		threshold(img_gray, img_bw, thresh, 255, cv::THRESH_BINARY);
		Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(3, 3));
		cv::dilate(img_bw, img_bw, kernel, Point(-1, -1), 1);
	}




	/* try different methods to filter eye image.
	*/
	void filterImg(const Mat& img_gray, Mat& img_blur)
	{
		//1 Gaussian
		cv::GaussianBlur(img_gray, img_blur, Size(5, 5), 0, 0);

		//2 mean shift: can narrow edges.
		//Mat temp;
		//cvtColor(img_blur, temp, CV_GRAY2BGR);
		////measureTime([&]() {bilateralFilter(img_blur, temp, 5, 100, 1, 4); }, "bilateral\t");
		//measureTime([&]() {pyrMeanShiftFiltering(temp, temp, 20, 20, 2); }, "mean shift\t");
		//cvtColor(temp, img_blur, CV_BGR2GRAY); 

		//3 morphology operation
		//close operation: weaken thin eyelashes
		//open operation: weaken small light spots
		Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(7, 7));
		Mat dst;
		cv::morphologyEx(img_blur, dst, cv::MORPH_CLOSE, kernel, Point(-1, -1), 1);
		cv::morphologyEx(dst, dst, cv::MORPH_OPEN, kernel, Point(-1, -1), 1);
		Mat tmp = dst - img_gray;
		img_blur = dst;
	}

	void detectPupilContour(const Mat& img_gray, Mat& edges, const Rect& pupil_rect)
	{
		//方法1：固定阈值
		/*double thresh = 59;
		cv::Canny(img_gray, edges, thresh*0.5, thresh, 3, true);*/

		////方法2：自适应阈值
		Mat_<uchar> edges_withAdaptiveThresh;
		{
			Sobel(img_gray, img_gradX, img_gradX.depth(), 1, 0);
			Sobel(img_gray, img_gradY, img_gradY.depth(), 0, 1);
		}

		double adp_thresh = canny_my2(img_gradX, img_gradY, edges_withAdaptiveThresh,
			0, 0.5, pupil_rect);
		cout << "auto thresh (inlier rect) = " << adp_thresh << endl;
		edges = edges_withAdaptiveThresh;

		Mat_<uchar> edges_withAdaptiveThresh2;
		double adp_thresh2 = canny_my(img_gray, edges_withAdaptiveThresh2,
			0, 0.5, 0);
		cout << "auto thresh2 = " << adp_thresh2 << endl;
		imshow("edges_withAdaptiveThresh2", edges_withAdaptiveThresh2);

		/*namedWindow("edges2");
		moveWindow("edges2", 250, 500);
		imshow("edges2", edges_withAdaptiveThresh);*/

		////方法3：
		//Mat edges_pure = canny_pure(img_gray, true, true, 64, 0.7f, 0.5f);
		/*namedWindow("edges3");
		moveWindow("edges3", 500, 500);
		imshow("edges3", edges_pure);*/
		//edges = edges_pure;

	}

	void fitPupilEllipse(const Mat& edges, RotatedRect& ellipse_rect, int K = 10)
	{
		// initial params
		//由于下面的椭圆拟合函数需要float类型，所以这里都为这种类型
		Mat_<float> edge_data; //N x 2
		{
			for (int i = 0; i < edges.rows; i++)
				for (int j = 0; j < edges.cols; j++)
					if (edges.at<uchar>(i, j) > 0)
					{
						Mat temp = (Mat_<float>(1, 2) << j, i);
						edge_data.push_back(temp);
					}
		}

		if (edge_data.rows < 5)
		{
			cout << "RANSAC: not enough edge data!" << endl;
			return;
		}

		int sup = 0;
		Mat_<int> best_indexs;
		double tau_pixel = 5;
		for (int kIterations = 0; kIterations < K; kIterations++)
		{
			// 1 random five samples from edge_data
			Mat_<float> sample;
			{
				Mat_<int> sample_indexs(1, 5);
				randomNoRepeat(sample_indexs, 0, edge_data.rows - 1);
				for (int i = 0; i < 5; i++)
					sample.push_back(edge_data.row(sample_indexs(i))); //
			}

			// fitting
			RotatedRect elp_rotatedRect = fitEllipseDirect(sample);
			MyEllipse elp(elp_rotatedRect);

			//显示椭圆以及样本点
			/*Mat img_temp;
			cvtColor(edges, img_temp, CV_GRAY2BGR);
			ellipse(img_temp, elp_rotatedRect, RED);
			for (int i = 0; i < 5; i++)
			{
				auto f = elp.approxDistancePointEllipse2(sample.row(i).t());
				drawMarker(img_temp, Point(sample(i, 0), sample(i, 1)), RED);
				cout << "distance = " << f << endl;
			}*/

			//利用Rect范围约束，提前结束循环
			Rect rect = Rect(0, 0, edges.cols, edges.rows);
			Rect rect_half = rectScale(rect, 0.5);

			//TODO 可能将所有的情况都被提前拒绝了，那样就无法拟合了
			Rect elp_boundingRect = elp_rotatedRect.boundingRect();
			/*if (elp_boundingRect.x < 0 ||
				elp_boundingRect.y < 0 ||
				elp_boundingRect.x + elp_boundingRect.width > rect.width ||
				elp_boundingRect.y + elp_boundingRect.height > rect.height)
				continue;*/

				//if (elp_rotatedRect.center.x < rect_half.x || 
				//	elp_rotatedRect.center.y < rect_half.y ||
				//	elp_rotatedRect.center.x > rect_half.x + rect_half.width ||
				//	elp_rotatedRect.center.y > rect_half.y + rect_half.height)
				//	continue;


			// 2 inliers
			int count = 0;
			Mat_<int> indexs;

			//方法1：矩阵计算inliers，速度很快
			Mat distance = elp.approxDistancePointEllipse2(edge_data.t());
			Mat inliers = (distance < tau_pixel) / 255;
			for (int i = 0; i < inliers.cols; i++)
				if (inliers.at<uchar>(0, i) > 0)
					indexs.push_back(i);
			Scalar tmp = sum(inliers);
			count = tmp(0);

			//方法2：单个点分别计算inliers
			//耗时
			//int count2 = 0;
			//for (int i = 0; i < edge_data.rows; i++)
			//{
			//	//double f = elp.distancePointEllipse(edge_data.row(i).t());
			//	double f = elp.approxDistancePointEllipse(edge_data.row(i).t());
			//	//cout << f2 - f << endl;
			//	
			//	if (f < tau_pixel)
			//	{
			//		++count2;
			//		Mat temp = (Mat_<int>(1, 1) << i);
			//		indexs.push_back(temp);
			//	}
			//}


			// 3 support function	
			if (count >= sup)
			{
				sup = count;
				best_indexs = indexs;
			}
		}// end iterations
		Mat_<float> best_data;
		for (int i = 0; i < best_indexs.rows; i++)
		{
			best_data.push_back(edge_data.row(best_indexs(i)));
		}
		ellipse_rect = fitEllipseDirect(best_data);
	}

	void fitPupilEllipse2(const Mat& edges, const Rect& inlinerRect,
		RotatedRect& ellipse_rect, int K = 10)
	{
		// initial params
		Mat_<double> edge_data; //N x 2
		{
			for (int i = 0; i < edges.rows; i++)
				for (int j = 0; j < edges.cols; j++)
					if (edges.at<uchar>(i, j) > 0)
					{
						Mat temp = (Mat_<double>(1, 2) << j, i);
						edge_data.push_back(temp);
					}
		}

		Mat_<double> edge_data2; //N x 2
		{
			int rmin = inlinerRect.y, rmax = inlinerRect.y + inlinerRect.height;
			int cmin = inlinerRect.x, cmax = inlinerRect.x + inlinerRect.width;
			for (int i = rmin; i < rmax; i++)
				for (int j = cmin; j < cmax; j++)
					if (edges.at<uchar>(i, j) > 0)
					{
						Mat temp = (Mat_<double>(1, 2) << j, i);
						edge_data2.push_back(temp);
					}
		}

		if (edge_data2.rows < 5)
		{
			cout << "RANSAC: not enough edge data!" << endl;
			return;
		}

		double tau_pixel = 3.0;
		int sup = 0;
		Mat_<int> best_indexs;
		for (int kIterations = 0; kIterations < K; kIterations++)
		{
			// 1 random five samples from edge_data
			Mat_<double> sample; //5 x 2
			{
				Mat_<int> sample_indexs(1, 5);
				randomNoRepeat(sample_indexs, 0, edge_data2.rows - 1);
				for (int i = 0; i < 5; i++)
					sample.push_back(edge_data2.row(sample_indexs(i)));
			}

			// fitting
			RotatedRect elp_rotatedRect;
			{
				Mat sample2;
				//椭圆拟合不能使用<double>类型
				sample.convertTo(sample2, CV_32F);
				//fitEllipse更为直接
				elp_rotatedRect = fitEllipse(sample2);
				//fitEllipseDirect得到的椭圆可能无效，需要提前排除
				/*if (isnan(elp_rotatedRect.size.width) || isnan(elp_rotatedRect.size.height))
					continue;*/
			}
			MyEllipse elp(elp_rotatedRect);


			//显示椭圆以及样本点
			Mat img_temp;
			{
				cvtColor(edges, img_temp, CV_GRAY2BGR);
				ellipse(img_temp, elp_rotatedRect, RED);
				for (int i = 0; i < 5; i++)
					drawMarker(img_temp, Point(sample(i, 0), sample(i, 1)), RED);
			}


			//1.2 提前结束循环
			//TODO: 可能将所有的情况都被提前拒绝了，那样就无法拟合了

			//利用长短轴比例，不能过小
			if (elp.s_ / elp.l_ < 0.2)
				continue;

			//利用样本点到椭圆的距离
			{
				Mat distance_sample = elp.approxDistancePointEllipse2(sample.t());
				Scalar tmp_1 = sum(distance_sample > 2.0);
				if (tmp_1(0) > 0)
					continue;
			}

			//利用Rect范围约束
			Rect rect = Rect(0, 0, edges.cols, edges.rows);
			Rect elp_boundingRect = elp_rotatedRect.boundingRect();
			int margin = 20;//留有一定的裕度
			if (elp_boundingRect.x < 0 - margin ||
				elp_boundingRect.y < 0 - margin ||
				elp_boundingRect.x + elp_boundingRect.width > rect.width + margin ||
				elp_boundingRect.y + elp_boundingRect.height > rect.height + margin)
				continue;


			//利用样本点的椭圆梯度与图像梯度的一致性提前结束
			int validnum = getGradientCoincideCount(elp, sample.t());
			if (validnum < 5)
				continue;


			// 2 inliers
			int count = 0;
			Mat_<int> inliers_indexs;

			//方法1：矩阵计算inliers，速度很快
			Mat distance = elp.approxDistancePointEllipse2(edge_data.t());
			Mat inliers = (distance < tau_pixel) / 255;
			Mat_<double> inliers_data;
			for (int i = 0; i < inliers.cols; i++)
				if (inliers.at<uchar>(0, i) > 0)
				{
					inliers_indexs.push_back(i);
					inliers_data.push_back(edge_data.row(i));
				}
			Scalar tmp = sum(inliers);
			count = tmp(0);
			count = getGradientCoincideCount(elp, inliers_data.t());

			// 3 support function	
			if (count >= sup)
			{
				sup = count;
				best_indexs = inliers_indexs;
				if (count > 0.8*edge_data.rows) //提前结束
					break;
			}
		}// end iterations

		if (!sup) //所有椭圆全部拒绝
		{
			cout << "no ellipse" << endl;
			ellipse_rect.center = Point(edges.cols, edges.rows);
			return;
		}

		Mat_<float> best_data, best_data2;
		for (int i = 0; i < best_indexs.rows; i++)
			best_data.push_back(edge_data.row(best_indexs(i)));
		best_data.convertTo(best_data2, CV_32F);
		ellipse_rect = fitEllipseDirect(best_data2);
	}




	/**
	points, 2 x N
	*/
	int getGradientCoincideCount(MyEllipse elp, const Mat_<double>& points)
	{
		int sumcol = points.cols;
		Mat_<double> elp_gradient = elp.getGradient(points);
		Mat_<double> img_gradient(2, sumcol);
		Mat_<double> costheta(1, sumcol);
		for (int i = 0; i != sumcol; ++i)
		{
			img_gradient(0, i) = img_gradX(Point(points(0, i), points(1, i)));
			img_gradient(1, i) = img_gradY(Point(points(0, i), points(1, i)));
			costheta(0, i) = elp_gradient.col(i).dot(img_gradient.col(i)) /
				(norm(elp_gradient.col(i))*norm(img_gradient.col(i)));
		}

		Scalar tmp_1 = sum(costheta > 0.86);//0.707
		int validnum = tmp_1(0) / 255;
		return validnum;		
	}


	void showGradient(const Mat& edges, Mat& dst)
	{
		Mat_<double> edge_data; //N x 2
		{
			for (int i = 0; i < edges.rows; i++)
				for (int j = 0; j < edges.cols; j++)
					if (edges.at<uchar>(i, j) > 0)
					{
						Mat temp = (Mat_<double>(1, 2) << j, i);
						edge_data.push_back(temp);
					}
		}

		int rows = edge_data.rows;
		Mat_<double> img_gradient(rows, 2);
		cv::cvtColor(edges, dst, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < rows; i+=3)
		{
			img_gradient(i, 0) = img_gradX(Point(edge_data(i, 0), edge_data(i, 1)))/8;
			img_gradient(i, 1) = img_gradY(Point(edge_data(i, 0), edge_data(i, 1)))/8;
			line(dst, Point(edge_data(i, 0), edge_data(i, 1)), Point(edge_data(i, 0), 
				edge_data(i, 1))+Point(img_gradient(i, 0), img_gradient(i, 1)), RED);

		}
		
	}

	RotatedRect ellipse_rect;

private:
	//图像梯度，在RANSAC需要使用，因此设为全局变量
	Mat_<float> img_gradX;
	Mat_<float> img_gradY;

};


#endif