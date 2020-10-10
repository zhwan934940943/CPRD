#pragma once

#ifndef UTILS_H_
#define UTILS_H_



#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string>

//use ellipse fitting by Swirski 2012
#include "Swirski2012/PupilTracker.h"
#include "Swirski2012/utils.h"
#include <tbb/tbb.h>
#include <boost/foreach.hpp>

#include<opencv2/opencv.hpp>

using namespace std;

using cv::Mat;


/** Reads string list.
e.g.,
	string dirname = libpath + "data_eye/";
	string filename = "imagelist.txt";
	vector<string> imagelist;
	readStringList(dirname + filename, imagelist);
*/
inline bool readStringList_txt(const string& pathname, vector<string>& stringlist)
{
	stringlist.resize(0);
	ifstream fin(pathname);
	{
		if (!fin.is_open())
			throw("cannot open file");
	}

	string str;
	while (fin >> str)
	{
		stringlist.push_back(str);
	}

	return true;
}


/* checks the validity of the import image
possible error reasons: false dir, file error.
*/
inline void checkImg(const Mat& img)
{
	if (img.empty())
		throw "error: Import image fail!";//runtime_error("error");
}

/* Transforms image to grayscale.
@param src input image: RGB, grayscale, or 4 channels with alpha.
@param imgGray output image: the same depth (UINT8 or float) as src.
*/
inline void img2Gray(const Mat& src, Mat& imgGray)
{
	if (src.channels() == 1)
		imgGray = src;
	else if (src.channels() == 3)
		cv::cvtColor(src, imgGray, cv::COLOR_BGR2GRAY);
	else if (src.channels() == 4)
		cv::cvtColor(src, imgGray, cv::COLOR_BGRA2GRAY);
	else
		throw std::runtime_error("Unsupported number of channels");
}

inline void img2BGR(const Mat& src, Mat& img_color)
{
	if (src.channels() == 3)
		img_color = src;
	else if (src.channels() == 1)
		cv::cvtColor(src, img_color, cv::COLOR_GRAY2BGR);
	else
		throw std::runtime_error("Unsupported number of channels");
}




/** Scales a rect.
	1 [default], center scale
	2 an image scale
*/
inline Rect rectScale(const Rect& rect, const float& ratio, 
	bool center_flag = true, bool useSquareHaar = true)
{
	int width = rect.width;
	int height = rect.height;
	if (center_flag)
	{
		//Scales a rect based on the center.
		//Rect()内小数的默认处理方式是ceil
		if (useSquareHaar)
			return Rect(rect.x - (ratio - 1)*width / 2, rect.y - (ratio - 1)*height / 2, \
				width*ratio, height*ratio);
		else //horizontal Haar
			return Rect(rect.x - (ratio - 1)*width / 2, rect.y, width*ratio, height);
	}
	else
	{
		//Scales a rect based on the lefttop.
		return Rect(rect.x*ratio, rect.y*ratio, width*ratio, height*ratio);
	}

}



/* Changes putText() to number.
@ params org Position to put number.
*/
inline void putNumber(const Mat& src, const double& number, Point org, Scalar color = RED)
{
	if (number != NULL)
	{
		char buffer[10];
		sprintf_s(buffer, "%.0f", number);
		putText(src, buffer, org, cv::FONT_HERSHEY_SIMPLEX, 0.8, color);
	}
}


//Filter edges
//strategy: the size of edge must be larger than 5 elements.
inline void edgesFilter(Mat& edges)
{
	//contour 过滤，去除小的curves
	vector<cv::Vec4i> hierarchy;
	vector<vector<Point> > curves;
	cv::findContours(edges, curves, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	//CV_CHAIN_APPROX_TC89_KCOS
	//CV_CHAIN_APPROX_SIMPLE

	vector<vector<Point> > candidates;
	Mat edgesFiltered = Mat::zeros(edges.size(), CV_8UC1);
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
	edges = edgesFiltered;
}


/** Canny detector by PuRe 2018, the high threshold is based on gradient magnititude percentage.
Mat detectedEdges = canny_pure(img, true, true, 64, 0.7f, 0.5f);
*/
inline Mat canny_pure(const Mat &in, bool blurImage, bool useL2, int bins, float nonEdgePixelsRatio, float lowHighThresholdRatio)
{
	(void)useL2;
	/* 1
	 * Smoothing and directional derivatives
	 * TODO: adapt sizes to image size
	 */
	Mat blurred;
	if (blurImage) {
		Size blurSize(5, 5);
		cv::GaussianBlur(in, blurred, blurSize, 1.5, 1.5, cv::BORDER_REPLICATE);
	}
	else
		blurred = in;

	Mat_<float> dx, dy;
	Sobel(blurred, dx, dx.type(), 1, 0, 7, 1, 0, cv::BORDER_REPLICATE);
	Sobel(blurred, dy, dy.type(), 0, 1, 7, 1, 0, cv::BORDER_REPLICATE);

	/*
	 *  Magnitude
	 */
	double minMag = 0;
	double maxMag = 0;
	float *p_res;
	float *p_x, *p_y; // result, x, y

	Mat_<float> img_mag;
	cv::magnitude(dx, dy, img_mag);

	// Normalization
	cv::minMaxLoc(img_mag, &minMag, &maxMag);
	img_mag = img_mag / maxMag;


	/* 2
	 *  Threshold selection based on the magnitude histogram
	 */
	float low_th = 0;
	float high_th = 0;

	// Histogram //统计梯度直方图，但是可以直接用calcHist()
	int *histogram = new int[bins]();
	Mat res_idx = (bins - 1) * img_mag; //value range [0,bins-1]
	res_idx.convertTo(res_idx, CV_16U);
	short *p_res_idx = 0;
	for (int i = 0; i < res_idx.rows; i++)
	{
		p_res_idx = res_idx.ptr<short>(i);
		for (int j = 0; j < res_idx.cols; j++)
			histogram[p_res_idx[j]]++;
	}

	// Ratio //自动确定canny的阈值
	int sum = 0;
	int nonEdgePixels = nonEdgePixelsRatio * in.rows * in.cols;
	for (int i = 0; i < bins; i++)
	{
		sum += histogram[i];
		if (sum > nonEdgePixels)
		{
			high_th = float(i + 1) / bins;
			break;
		}
	}
	low_th = lowHighThresholdRatio * high_th;


	/*3
	 *  Non maximum supression
	 */
	const float tg22_5 = 0.4142135623730950488016887242097f;
	const float tg67_5 = 2.4142135623730950488016887242097f;
	uchar *_edgeType;
	float *p_res_b, *p_res_t;
	Mat_<uchar> edgeType(img_mag.size());
	edgeType.setTo(0);
	for (int i = 1; i < img_mag.rows - 1; i++)
	{
		_edgeType = edgeType.ptr<uchar>(i);

		p_res = img_mag.ptr<float>(i);
		p_res_t = img_mag.ptr<float>(i - 1);
		p_res_b = img_mag.ptr<float>(i + 1);

		p_x = dx.ptr<float>(i);
		p_y = dy.ptr<float>(i);

		for (int j = 1; j < img_mag.cols - 1; j++)
		{
			float m = p_res[j];
			if (m < low_th)
				continue;

			float iy = p_y[j];
			float ix = p_x[j];
			float y = abs((double)iy);
			float x = abs((double)ix);

			uchar val = p_res[j] > high_th ? 255 : 128;

			float tg22_5x = tg22_5 * x;
			if (y < tg22_5x)
			{
				if (m > p_res[j - 1] && m >= p_res[j + 1])
					_edgeType[j] = val;
			}
			else
			{
				float tg67_5x = tg67_5 * x;
				if (y > tg67_5x)
				{
					if (m > p_res_b[j] && m >= p_res_t[j])
						_edgeType[j] = val;
				}
				else
				{
					if ((iy <= 0) == (ix <= 0))
					{
						if (m > p_res_t[j - 1] && m >= p_res_b[j + 1])
							_edgeType[j] = val;
					}
					else
					{
						if (m > p_res_b[j - 1] && m >= p_res_t[j + 1])
							_edgeType[j] = val;
					}
				}
			}
		}
	}

#ifdef MY_DEBUG
	Mat edgeType_temp = edgeType;
#endif

	/*4
	 *  Hystheresis
	 */
	int pic_x = edgeType.cols;
	int pic_y = edgeType.rows;
	int area = pic_x * pic_y;
	int lines_idx = 0;
	int idx = 0;

	vector<int> lines;
	Mat_<uchar> edge(img_mag.size());
	edge.setTo(0);
	for (int i = 1; i < pic_y - 1; i++) {
		for (int j = 1; j < pic_x - 1; j++) {

			if (edgeType.data[idx + j] != 255 || edge.data[idx + j] != 0)
				continue;

			edge.data[idx + j] = 255;
			lines_idx = 1;
			lines.clear();
			lines.push_back(idx + j);
			int akt_idx = 0;

			while (akt_idx < lines_idx) {
				int akt_pos = lines[akt_idx];
				akt_idx++;

				if (akt_pos - pic_x - 1 < 0 || akt_pos + pic_x + 1 >= area)
					continue;

				for (int k1 = -1; k1 < 2; k1++)
					for (int k2 = -1; k2 < 2; k2++) {
						if (edge.data[(akt_pos + (k1*pic_x)) + k2] != 0 || edgeType.data[(akt_pos + (k1*pic_x)) + k2] == 0)
							continue;
						edge.data[(akt_pos + (k1*pic_x)) + k2] = 255;
						lines.push_back((akt_pos + (k1*pic_x)) + k2);
						lines_idx++;
					}
			}
		}
		idx += pic_x;
	}
#ifdef MY_DEBUG
	Mat edg_temp = edge;
#endif
	return edge;
}



inline bool fitPupilEllipseSwirski(const Mat& img_blur, Mat& edges, cv::RotatedRect& elPupil)
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
	params.Radius_Max = d / 2;

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





#endif