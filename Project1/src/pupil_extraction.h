#pragma once

#ifndef PupilExtraction_H_
#define PupilExtraction_H_


#include <QString>
#include <bitset>
#include <iostream>
#include <numeric>
#include <random>

#include <opencv2/opencv.hpp>

#include <my_lib.h>
#include "src/m_PupilDetectorHaar.h"
//#include <m_PupilDetectorHaar.cpp> //由于项目中没有添加，所以这里必须包含


#include <pupiltracker/PupilTracker.h>
#include <tbb/tbb.h>
#include <pupiltracker/utils.h>
#include <boost/foreach.hpp>

using namespace std;
//using namespace cv;
using namespace mycv;
using namespace mymath;


class PupilExtractionMethod
{

public:
	void detect(Mat& img_in)
	{
		checkImg(img_in);
		Mat img_gray, img_BGR;
		img2GrayBGR(img_in, img_gray, img_BGR);


		/*Mat img_blur;
		measureTime([&]() {filterImg(img_gray, img_blur); }, "filter\t");
		img_gray = img_blur;*/

		section("1 detect pupil region");
		//1.1 coarse region with Haar
		PupilDetectorHaar haar;
		measureTime([&]() {detectPupilRegion(img_gray, haar);
		haar.draw(img_BGR, 0);
		}, "haar\t");

		Rect pupil_rect = rectScale(haar.outer_rect2_, 1)
			&haar.outer_rect_; //Rect(0, 0, img_gray.cols, img_gray.rows);
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
		Rect inlinerRect = haar.pupil_rect2_ - haar.outer_rect2_.tl();
		Rect inlinerRect2 = (haar.pupil_rect2_&haar.pupil_rect_) - haar.outer_rect2_.tl();

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

	void detectPupilRegion(const Mat& img_gray, PupilDetectorHaar& haar)
	{
		HaarParams params;
		params.width_min = 31;
		params.width_max = 120; //240的一半
		params.wh_step = 4; //影响程序的执行速度
		params.ratio = 2;

		haar.detect(img_gray, params);


		//section("2.1 show haar results");
		//{
		//	Mat img_haar;
		//	cvtColor(img_gray, img_haar, CV_GRAY2BGR);
		//	haar.draw(img_haar);
		//	namedWindow("Eye with haar features");
		//	moveWindow("Eye with haar features", 0, 0);
		//	imshow("Eye with haar features", img_haar);
		//	waitKey(30);
		//}
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

	void filterImg(const Mat& img_gray, Mat& img_blur)
	{
		cv::GaussianBlur(img_gray, img_blur, Size(5, 5), 0, 0);

		//mean shift 可以将边缘变窄
		//Mat temp;
		//cvtColor(img_blur, temp, CV_GRAY2BGR);
		////measureTime([&]() {bilateralFilter(img_blur, temp, 5, 100, 1, 4); }, "bilateral\t");
		//measureTime([&]() {pyrMeanShiftFiltering(temp, temp, 20, 20, 2); }, "mean shift\t");
		//cvtColor(temp, img_blur, CV_BGR2GRAY); 

		//close操作可以弱化睫毛，效果很好
		//open操作可以弱化小亮斑
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



	bool fitPupilEllipseSwirski(Mat& img_blur, Mat& edges, cv::RotatedRect& elPupil)
	{
		// initial params
		//由于下面的椭圆拟合函数需要float类型，所以这里都为这种类型
		vector<Point2f> edgePoints; //N x 2
		{
			for (int i = 0; i < edges.rows; i++)
				for (int j = 0; j < edges.cols; j++)
					if (edges.at<uchar>(i, j) > 0)
					{
						Point2f temp(j, i);
						edgePoints.push_back(temp);
					}
		}

		std::vector<cv::Point2f> inliers;

		// Desired probability that only inliers are selected
		const double p = 0.999;
		// Probability that a point is an inlier
		double w = 30.0 / 100.0;
		// Number of points needed for a model
		const int n = 5;

		pupiltracker::TrackerParams params;
		params.Radius_Min = 460 / 2 / 8;//460是图像像素宽度
		params.Radius_Max = 460 / 2 / 3;
		//为了适应我的方法，这里修改
		float d = sqrt(pow(edges.rows, 2) + pow(edges.cols, 2));
		params.Radius_Min = min(edges.rows, edges.cols) / 4; 
		params.Radius_Max = d/2;

		params.CannyBlur = 1;
		params.CannyThreshold1 = 20;
		params.CannyThreshold2 = 40;
		params.StarburstPoints = 0;

		params.PercentageInliers = 30;
		params.InlierIterations = 2;
		params.ImageAwareSupport = true;
		params.EarlyTerminationPercentage = 95;
		params.EarlyRejection = true;
		params.Seed = -1;


		if (edgePoints.size() >= n) // Minimum points for ellipse
		{
			// RANSAC!!!

			double wToN = std::pow(w, n);
			int k = static_cast<int>(std::log(1 - p) / std::log(1 - wToN) + 2 * std::sqrt(1 - wToN) / wToN);


			//size_t threshold_inlierCount = std::max<size_t>(n, static_cast<size_t>(out.edgePoints.size() * 0.7));

			// Use TBB for RANSAC
			struct EllipseRansac_out {
				std::vector<cv::Point2f> bestInliers;
				cv::RotatedRect bestEllipse;
				double bestEllipseGoodness;
				int earlyRejections;
				bool earlyTermination;

				EllipseRansac_out() : bestEllipseGoodness(-std::numeric_limits<double>::infinity()), earlyTermination(false), earlyRejections(0) {}
			};

			struct EllipseRansac {
				const pupiltracker::TrackerParams& params;
				const std::vector<cv::Point2f>& edgePoints;
				int n;
				const cv::Rect& bb;
				const cv::Mat_<float>& mDX;
				const cv::Mat_<float>& mDY;
				int earlyRejections;
				bool earlyTermination;

				EllipseRansac_out out;

				EllipseRansac(
					const pupiltracker::TrackerParams& params,
					const std::vector<cv::Point2f>& edgePoints,
					int n,
					const cv::Rect& bb,
					const cv::Mat_<float>& mDX,
					const cv::Mat_<float>& mDY)
					: params(params), edgePoints(edgePoints), n(n), bb(bb), mDX(mDX), mDY(mDY), earlyTermination(false), earlyRejections(0)
				{
				}

				EllipseRansac(EllipseRansac& other, tbb::split)
					: params(other.params), edgePoints(other.edgePoints), n(other.n), bb(other.bb), mDX(other.mDX), mDY(other.mDY), earlyTermination(other.earlyTermination), earlyRejections(other.earlyRejections)
				{
					//std::cout << "Ransac split" << std::endl;
				}

				void operator()(const tbb::blocked_range<size_t>& r)
				{
					if (out.earlyTermination)
						return;
					//std::cout << "Ransac start (" << (r.end()-r.begin()) << " elements)" << std::endl;
					for (size_t i = r.begin(); i != r.end(); ++i)
					{
						// Ransac Iteration
						// ----------------
						std::vector<cv::Point2f> sample;
						if (params.Seed >= 0)
							sample = pupiltracker::randomSubset(edgePoints, n, static_cast<unsigned int>(i + params.Seed));
						else
							sample = pupiltracker::randomSubset(edgePoints, n);

						cv::RotatedRect ellipseSampleFit = fitEllipse(sample);
						// Normalise ellipse to have width as the major axis.
						if (ellipseSampleFit.size.height > ellipseSampleFit.size.width)
						{
							ellipseSampleFit.angle = std::fmod(ellipseSampleFit.angle + 90, 180);
							std::swap(ellipseSampleFit.size.height, ellipseSampleFit.size.width);
						}

						cv::Size s = ellipseSampleFit.size;
						// Discard useless ellipses early
						if (!ellipseSampleFit.center.inside(bb)
							|| s.height > params.Radius_Max * 2
							|| s.width > params.Radius_Max * 2
							|| s.height < params.Radius_Min * 2 && s.width < params.Radius_Min * 2
							|| s.height > 4 * s.width
							|| s.width > 4 * s.height
							)
						{
							// Bad ellipse! Go to your room!
							continue;
						}

						// Use conic section's algebraic distance as an error measure
						pupiltracker::ConicSection conicSampleFit(ellipseSampleFit);

						// Check if sample's gradients are correctly oriented
						if (params.EarlyRejection)
						{
							bool gradientCorrect = true;
							BOOST_FOREACH(const cv::Point2f& p, sample)
							{
								cv::Point2f grad = conicSampleFit.algebraicGradientDir(p);
								float dx = mDX(cv::Point(p.x, p.y));
								float dy = mDY(cv::Point(p.x, p.y));

								float dotProd = dx * grad.x + dy * grad.y;

								gradientCorrect &= dotProd > 0;
							}
							if (!gradientCorrect)
							{
								out.earlyRejections++;
								continue;
							}
						}

						// Assume that the sample is the only inliers

						cv::RotatedRect ellipseInlierFit = ellipseSampleFit;
						pupiltracker::ConicSection conicInlierFit = conicSampleFit;
						std::vector<cv::Point2f> inliers, prevInliers;

						// Iteratively find inliers, and re-fit the ellipse
						for (int i = 0; i < params.InlierIterations; ++i)
						{
							// Get error scale for 1px out on the minor axis
							cv::Point2f minorAxis(-std::sin(PI / 180.0*ellipseInlierFit.angle), std::cos(PI / 180.0*ellipseInlierFit.angle));
							cv::Point2f minorAxisPlus1px = ellipseInlierFit.center + (ellipseInlierFit.size.height / 2 + 1)*minorAxis;
							float errOf1px = conicInlierFit.distance(minorAxisPlus1px);
							float errorScale = 1.0f / errOf1px;

							// Find inliers
							inliers.reserve(edgePoints.size());
							const float MAX_ERR = 2;
							BOOST_FOREACH(const cv::Point2f& p, edgePoints)
							{
								float err = errorScale * conicInlierFit.distance(p);

								if (err*err < MAX_ERR*MAX_ERR)
									inliers.push_back(p);
							}

							if (inliers.size() < n) {
								inliers.clear();
								continue;
							}

							// Refit ellipse to inliers
							ellipseInlierFit = fitEllipse(inliers);
							conicInlierFit = pupiltracker::ConicSection(ellipseInlierFit);

							// Normalise ellipse to have width as the major axis.
							if (ellipseInlierFit.size.height > ellipseInlierFit.size.width)
							{
								ellipseInlierFit.angle = std::fmod(ellipseInlierFit.angle + 90, 180);
								std::swap(ellipseInlierFit.size.height, ellipseInlierFit.size.width);
							}
						}
						if (inliers.empty())
							continue;

						// Discard useless ellipses again
						s = ellipseInlierFit.size;
						if (!ellipseInlierFit.center.inside(bb)
							|| s.height > params.Radius_Max * 2
							|| s.width > params.Radius_Max * 2
							|| s.height < params.Radius_Min * 2 && s.width < params.Radius_Min * 2
							|| s.height > 4 * s.width
							|| s.width > 4 * s.height
							)
						{
							// Bad ellipse! Go to your room!
							continue;
						}

						// Calculate ellipse goodness
						double ellipseGoodness = 0;
						if (params.ImageAwareSupport)
						{
							BOOST_FOREACH(cv::Point2f& p, inliers)
							{
								cv::Point2f grad = conicInlierFit.algebraicGradientDir(p);
								float dx = mDX(p);
								float dy = mDY(p);

								double edgeStrength = dx * grad.x + dy * grad.y;

								ellipseGoodness += edgeStrength;
							}
						}
						else
						{
							ellipseGoodness = inliers.size();
						}

						if (ellipseGoodness > out.bestEllipseGoodness)
						{
							std::swap(out.bestEllipseGoodness, ellipseGoodness);
							std::swap(out.bestInliers, inliers);
							std::swap(out.bestEllipse, ellipseInlierFit);

							// Early termination, if 90% of points match
							if (params.EarlyTerminationPercentage > 0 && out.bestInliers.size() > params.EarlyTerminationPercentage*edgePoints.size() / 100)
							{
								earlyTermination = true;
								break;
							}
						}

					}
					//std::cout << "Ransac end" << std::endl;
				}

				void join(EllipseRansac& other)
				{
					//std::cout << "Ransac join" << std::endl;
					if (other.out.bestEllipseGoodness > out.bestEllipseGoodness)
					{
						std::swap(out.bestEllipseGoodness, other.out.bestEllipseGoodness);
						std::swap(out.bestInliers, other.out.bestInliers);
						std::swap(out.bestEllipse, other.out.bestEllipse);
					}
					out.earlyRejections += other.out.earlyRejections;
					earlyTermination |= other.earlyTermination;

					out.earlyTermination = earlyTermination;
				}
			};


			cv::Mat_<float> mPupilSobelX;
			cv::Mat_<float> mPupilSobelY;
			cv::Sobel(img_blur, mPupilSobelX, CV_32F, 1, 0, 3);
			cv::Sobel(img_blur, mPupilSobelY, CV_32F, 0, 1, 3);
			Rect bbPupil(0, 0, img_blur.cols, img_blur.rows);
			EllipseRansac ransac(params, edgePoints, n, bbPupil, mPupilSobelX, mPupilSobelY);
			try
			{
				tbb::parallel_reduce(tbb::blocked_range<size_t>(0, k, k / 8), ransac);
			}
			catch (std::exception& e)
			{
				const char* c = e.what();
				std::cerr << e.what() << std::endl;
			}
			inliers = ransac.out.bestInliers;


			int earlyRejections = ransac.out.earlyRejections;
			bool earlyTermination = ransac.out.earlyTermination;


			cv::RotatedRect ellipseBestFit = ransac.out.bestEllipse;
			pupiltracker::ConicSection conicBestFit(ellipseBestFit);
			std::vector<pupiltracker::EdgePoint> edgePoints2;
			BOOST_FOREACH(const cv::Point2f& p, edgePoints)
			{
				cv::Point2f grad = conicBestFit.algebraicGradientDir(p);
				float dx = mPupilSobelX(p);
				float dy = mPupilSobelY(p);

				edgePoints2.push_back(pupiltracker::EdgePoint(p, dx*grad.x + dy * grad.y));
			}

			elPupil = ellipseBestFit;
			/*elPupil.center.x += roiPupil.x;
			elPupil.center.y += roiPupil.y;*/
		}

		if (inliers.size() == 0)
			return false;

		cv::Point2f pPupil = elPupil.center;

		/*out.pPupil = pPupil;
		out.elPupil = elPupil;
		out.inliers = inliers;*/

		return true;
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