// Zhonghua Wan @ 20190408
// Class for pupil features extracion
//modify time: 2020.10.7
// modify time: 2019.4.8
// create time: 2019.1.19


//------------------------------------------------
//					an example
//------------------------------------------------
//PupilDetectorHaar haar;
//haar.kf_ = 1.4; //1,1.1,1.2,1.3,1.4,...
//haar.ratio_outer_ = 1.42;//1.42, 2, 3, 4, 5, 6, 7
//haar.useSquareHaar_ = 0;
//haar.useInitRect_ = false;
//haar.xystep_ = 2;//2,3,4,...
//haar.whstep_ = 2;//2,3,4,...
//
//Mat img_gray;
//haar.detect(img_gray);
//
//cv::RotatedRect ellipse_rect;
//Point2f center_fitting;
//bool flag = haar.extractEllipse(img_gray, haar.pupil_rect_fine_,
//	ellipse_rect, center_fitting);
//
////show
//Mat img_coarse, img_fine;
//cv::cvtColor(img_gray, img_coarse, CV_GRAY2BGR);
//
//int thickness = 2;
//haar.drawCoarse(img_coarse);
//img_fine = img_coarse.clone();
//rectangle(img_fine, haar.pupil_rect_fine_, BLUE, thickness, 8);
//
//ellipse(img_fine, ellipse_rect, RED, thickness);
//drawMarker(img_fine, center_fitting, RED, cv::MARKER_CROSS, 20, thickness);
//
//imshow("Results", img_fine);
//waitKey(500);


#ifndef M_PupilDetectorHaar_H_
#define M_PupilDetectorHaar_H_

#include <opencv2/opencv.hpp>

#include "utils.h"


//#define HAAR_TEST
//#define UNIT_TEST
//optimization: exhaustive algorithm
//1 response,
//2 haarResponseMap (hot map)
//3 Eye image with haar rect


using namespace std;
using cv::Mat;
using cv::Mat_;
using cv::Point;
using cv::Scalar;

namespace myColor
{
	const Scalar red = Scalar(0, 0, 255);
	const Scalar blue = Scalar(255, 0, 0);
};


//TODO: old version, delete
class HaarParams
{
public:
	HaarParams() : initRectFlag(false), squareHaarFlag(true),
		outer_ratio(1.42), kf(1),
		width_min(31), width_max(120), wh_step(4), xy_step(4), roi(Rect(0, 0, 0, 0)),
		init_rect(Rect(0, 0, 0, 0)), mu_inner(50), mu_outer(200)
	{};

	bool squareHaarFlag;
	double outer_ratio; // outer_rect ratio
	double kf;


	// width: --> pupil_rect
	int width_min; //默认不使用原始分辨率的width
	int width_max;
	int wh_step; //wh的间隔步长

	int xy_step; //xy的间隔步长
	Rect roi; //inner rect 的范围，即不能超出该范围，显然(x,y)搜索范围要考虑width

	bool initRectFlag;
	Rect init_rect;
	double mu_inner;
	double mu_outer;
};




/* Detects pupil features using Haar detector.
@param img_gray input image: UINT8 [0, 255]. If img_gray is float [0,1], we can add a function to
	transfrom it to [0,255]. But it is not recommended, as [0,255] has better significance than [0,1].
*/
class PupilDetectorHaar
{
public:
	PupilDetectorHaar() :pupil_rect_coarse_(Rect()), outer_rect_coarse_(Rect()),
		max_response_coarse_(-255), iterate_count_(0) {};

	PupilDetectorHaar(const Mat &img_gray) : PupilDetectorHaar()
	{
		detect(img_gray);
	}

	void detect(const Mat &img_gray)
	{
		//guarantee the img_gray is UIN8 [0,255], not float [0,1].
		CV_Assert(img_gray.depth() == cv::DataType<uchar>::depth);
		frameNum_ += 1;

		//-------------------------- 1 Preprocessing --------------------------
		//1.1 downsampling
		Mat img_down;
		{
			ratio_downsample_ = max(img_gray.cols*1.0f / target_resolution_.width,
				img_gray.rows*1.0f / target_resolution_.height);
			resize(img_gray, img_down, Size(img_gray.cols / ratio_downsample_,
				img_gray.rows / ratio_downsample_));
		}
		imgboundary_ = Rect(0, 0, img_down.cols, img_down.rows);


		//1.2 image I strategy：high intensity suppression
		if (useInitRect_)
		{
			//todo: use max function.
			int tau;
			//tau = max(params.mu_outer, params.mu_inner + 30);
			if (mu_outer0_ - mu_inner0_ > 30)
				tau = mu_outer0_;
			else
				tau = mu_inner0_ + 30;
			filterLight(img_down, img_down, tau);
		}

		//-------------------------- 2 Coarse Detection --------------------------
		coarseDetection(img_down, pupil_rect_coarse_, outer_rect_coarse_,
			max_response_coarse_, mu_inner_, mu_outer_);
		if (useInitRect_ && frameNum_ == 1)
		{
			mu_inner0_ = mu_inner_;
			mu_outer0_ = mu_outer_;
		}


		//-------------------------- 3 Fine Detection --------------------------
		/* No fine detection for poor image quality when the intensity in the pupil region is close to,
		or even greater than, the intensity in the surrounding region.
		*/
		if (mu_outer_ - mu_inner_ < 5)
			pupil_rect_fine_ = pupil_rect_coarse_;
		else
			fineDetection(img_down, pupil_rect_coarse_, pupil_rect_fine_);

		// plot results, coarse and fine rectangles.
#ifdef HAAR_TEST
		Mat img_BGR;
		img2BGR(img_down, img_BGR);
		rectangle(img_BGR, roi_, BLUE, 1, 8);//plot ROI
		rectangle(img_BGR, pupil_rect_coarse_, RED, 1, 8);//plot ROI
		rectangle(img_BGR, pupil_rect_fine_, BLUE, 1, 8);//plot fine 
#endif

	//-------------------------- 4 Postprocessing --------------------------
	//upsample
		pupil_rect_coarse_ = rectScale(pupil_rect_coarse_, ratio_downsample_, false);
		outer_rect_coarse_ = rectScale(outer_rect_coarse_, ratio_downsample_, false);
		pupil_rect_fine_ = rectScale(pupil_rect_fine_, ratio_downsample_, false);

		center_coarse_ = Point2f(pupil_rect_coarse_.x + pupil_rect_coarse_.width*1.0f / 2,
			pupil_rect_coarse_.y + pupil_rect_coarse_.height*1.0f / 2);
		center_fine_ = Point2f(pupil_rect_fine_.x + pupil_rect_fine_.width*1.0f / 2,
			pupil_rect_fine_.y + pupil_rect_fine_.height*1.0f / 2);
	}




	void coarseDetection(const Mat& img_down, Rect& pupil_rect_coarse,
		Rect& outer_rect_coarse, double& max_response_coarse, double& mu_inner, double& mu_outer);


	void fineDetection(const Mat& img_down, const Rect& pupil_rect_coarse, Rect& pupil_rect_fine);


	//plot on the original img_BGR.
	void drawCoarse(Mat& img_BGR, Rect pupil_rect, Rect outer_rect, double max_response, Scalar color = RED)
	{
		int thickness = 1;
		rectangle(img_BGR, pupil_rect, color, thickness, 8);
		rectangle(img_BGR, outer_rect, color, thickness, 8);
		Point center = Point(pupil_rect.x + pupil_rect.width / 2, pupil_rect.y + pupil_rect.height / 2);
		drawMarker(img_BGR, center, color, cv::MARKER_CROSS,20,thickness);
		putNumber(img_BGR, max_response, center, color);
	}

	// overload
	void drawCoarse(Mat& img_BGR)
	{
		drawCoarse(img_BGR, pupil_rect_coarse_, outer_rect_coarse_, max_response_coarse_, GREEN);
	}

	static void filterLight(const Mat& img_gray, Mat& img_blur, int tau)
	{
		//GaussianBlur(img_gray, img_blur, Size(5, 5), 0, 0);

		//mean shift narrow the edges.
		//Mat temp;
		//cvtColor(img_blur, temp, CV_GRAY2BGR);
		////measureTime([&]() {bilateralFilter(img_blur, temp, 5, 100, 1, 4); }, "bilateral\t");
		//measureTime([&]() {pyrMeanShiftFiltering(temp, temp, 20, 20, 2); }, "mean shift\t");
		//cvtColor(temp, img_blur, CV_BGR2GRAY); 

		//close operation, weaken the impact of eyelashes
		//open operation, weaken  little light spots.
		//Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		//Mat dst;
		//morphologyEx(img_blur, dst, MORPH_CLOSE, kernel, Point(-1, -1), 1);
		//morphologyEx(dst, dst, MORPH_OPEN, kernel, Point(-1, -1), 1);
		//Mat tmp = dst - img_gray;
		//img_blur = dst;

		//high intensity suppression
		int col = img_gray.cols;
		int row = img_gray.rows;
		for (int i = 0; i < col; i++)
			for (int j = 0; j < row; j++)
			{
				if (img_gray.at<uchar>(j, i) > tau)
					img_blur.at<uchar>(j, i) = tau;
				else
					img_blur.at<uchar>(j, i) = img_gray.at<uchar>(j, i);
			}
	}






	//not in down sample scale, but in original scale.
	bool extractEllipse(const Mat& img_gray, const Rect& pupil_rect, cv::RotatedRect& ellipse_rect, Point2f& center_fitting)
	{
		center_fitting = center_fine_; //[default]
		if (mu_outer_ - mu_inner_ < 10) //others：0,10,20
			return false;

		Rect boundary(0, 0, img_gray.cols, img_gray.rows);
		double validRatio = 1.2; //or：1.42
		Rect validRect = rectScale(pupil_rect, validRatio)&boundary;
		Mat img_pupil = img_gray(validRect);
		GaussianBlur(img_pupil, img_pupil, Size(5, 5), 0, 0);

		Mat edges_filter;
		detectEdges(img_pupil,edges_filter);

		//imshow("edgesF", edges_filter);

		//Fit ellipse by RANSAC
		//int K;//iterations
		//{
		//	double p = 0.99;	// success rate 0.99
		//	double e = 0.7;		// outlier ratio, 0.7效果很好，但是时间长
		//	K = cvRound(log(1 - p) / log(1 - pow(1 - e, 5)));
		//}
		//RotatedRect ellipse_rect;
		//detector.fitPupilEllipse(edges_filter, ellipse_rect, K);

		
		fitPupilEllipseSwirski(img_pupil, edges_filter, ellipse_rect);
		ellipse_rect.center = ellipse_rect.center + Point2f(validRect.tl());

		if (pupil_rect_fine_.contains(ellipse_rect.center) && (ellipse_rect.size.width > 0))
		{
			center_fitting = ellipse_rect.center;
			return true;
		}
		else
			return false;

		//	//futher: PuRe
		//	//PuRe detectorPuRe;
		//	//Pupil pupil = detectorPuRe.run(img_pupil);
		//	//pupil.center = pupil.center + Point2f(validRect.tl());
	}


	// Extracts edges by Canny.
	void detectEdges(const Mat& img_pupil, Mat& edges_filter)
	{
		double tau1 = 1 - 20.0 / img_pupil.cols;//策略：10，20
		Mat edges = canny_pure(img_pupil, false, false, 64 * 2, tau1, 0.5);

		{
			//1 suppression some edges by high intensity
			int tau;
			//if (haar.mu_outer_ - haar.mu_inner_ > 30)
			//	tau = params.mu_outer;
			//else
			tau = mu_inner_ + 100;
			//PupilDetectorHaar::filterLight(img_t, img_t, tau);

			Mat img_bw;
			threshold(img_pupil, img_bw, tau, 255, cv::THRESH_BINARY);
			Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
			dilate(img_bw, img_bw, kernel, Point(-1, -1), 1);
			edges_filter = edges & ~img_bw;

			//2 filter edges by curves size
			edgesFilter(edges_filter);
		}
	}




private:
	// initial the search ranges of w (width_min_, width_max_) and xy (roi_).
	void initialSearchRange(const Mat& img_down);



	//Descending order (bubbling order)
	//area, index, are column vectors (int)
	void areaSort(Mat& area, Mat& index)
	{
		index = Mat(area.rows, 1, CV_16SC1);
		for (int i = 0; i < area.rows - 1; i++)
			index.at<int>(i) = i;
		for (int i = 0; i < area.rows - 1; i++)
			for (int j = i + 1; j < area.rows; j++)
			{
				if (area.at<int>(i) < area.at<int>(j))
				{
					int tmp = area.at<int>(i);
					area.at<int>(i) = area.at<int>(j);
					area.at<int>(j) = tmp;
					tmp = index.at<int>(i);
					index.at<int>(i) = index.at<int>(j);
					index.at<int>(j) = tmp;
				}
			}
	}



	/* gets Haar response map and maximum.
	optimization with variables (x,y)
	output:	pupil_rect, output rect, mu_inner, mu_outer
	return: max response value
	*/
	double getResponseMap(const Mat &integral_img, \
		int width, int height, double ratio, bool useSquareHaar, double kf, Rect roi, int xystep,
		Rect& pupil_rect, Rect& outer_rect, double& mu_inner, double& mu_outer);

	/* Computes response value with a Haar kernel.
	@param inner_rect
	*/
	double getResponseValue(const Mat& integral_img, Rect& inner_rect, Rect& outer_rect, double kf,
		double& mu_inner, double& mu_outer)
	{
		iterate_count_ += 1;
		// Filters rect, i.e., intersect two Rect.
		Rect boundary(0, 0, integral_img.cols - 1, integral_img.rows - 1);
		outer_rect &= boundary;
		inner_rect &= boundary;
		CV_Assert(outer_rect.width != 0);

		auto outer_bII = getBlockIntegral(integral_img, outer_rect);
		auto inner_bII = getBlockIntegral(integral_img, inner_rect);

		mu_outer = 1.0*(outer_bII - inner_bII) / (outer_rect.area() - inner_rect.area());
		mu_inner = 1.0*inner_bII / inner_rect.area();

		double f = mu_outer - kf * mu_inner;
		return f;
	}


	/*
	@param rect must locate in the range of intergral_img.
	*/
	int getBlockIntegral(const Mat& integral_img, const Rect& rect)
	{
		/* integral_img
			  a(x1-1,y1-1)				   b (x2,y1-1)
						II(x1,y1)----------------
						|						|
						|		   Rect		  height
						|						|
			  c(x1-1,y2)---------width-------d (x2,y2)
		*/

		/* DEPRECATED
		int x1 = rect.x;
		int y1 = rect.y;
		int x2 = x1 + rect.width - 1;
		int y2 = y1 + rect.height - 1;
		int d = integral_img.at<int>(y2, x2);
		int b = y1 ? integral_img.at<int>(y1 - 1, x2) : 0;
		int c = x1 ? integral_img.at<int>(y2, x1 - 1) : 0;
		int a = x1 && y1 ? integral_img.at<int>(y1 - 1, x1 - 1) : 0;
		*/
		int d = integral_img.at<int32_t>(rect.y + rect.height, rect.x + rect.width);
		int c = integral_img.at<int32_t>(rect.y + rect.height, rect.x);
		int b = integral_img.at<int32_t>(rect.y, rect.x + rect.width);
		int a = integral_img.at<int32_t>(rect.y, rect.x);

		int integra_value = d + a - b - c;
		return integra_value;
	}


	//rectlist 存储每个w得到的Rect
	//	rectlist2 存储非极大化抑制之后的Rect
	// 考虑到重叠，使用两次抑制
	void rectSuppression2(vector<Rect>& rectlist, vector<double>& response,
		vector<Rect>& rectlist_out, vector<double>& response_out)
	{
		vector<Rect> rectlist2;
		vector<double> response2;
		rectSuppression(rectlist, response, rectlist2, response2);
		rectSuppression(rectlist2, response2, rectlist_out, response_out);
	}



	void rectSuppression(vector<Rect>& rectlist, vector<double>& response,
		vector<Rect>& rectlist2, vector<double>& response2)
	{
		for (int i = 0; i < rectlist.size(); ++i)
		{
			bool flag_intersect = false; //flag: two rectangles intersect
			for (int j = 0; j < rectlist2.size(); ++j)
			{
				Rect tmp = rectlist[i] & rectlist2[j];
				if (tmp.width) //相交
				{
					flag_intersect = true;
					if (response[i] > response2[j])
					{
						rectlist2[j] = rectlist[i];
						response2[j] = response[i];
					}
					else
						continue;//break
				}

			}
			if (!flag_intersect)
			{
				rectlist2.push_back(rectlist[i]);
				response2.push_back(response[i]);
			}//end j
		}//end i
	}

public:
	//--------------------input parameters--------------------
	//Haar parameters
	double ratio_outer_ = 1.42;
	double kf_ = 1.4;//the weight of the response function
	bool useSquareHaar_ = false;//default [horizontal Haar outer]

	bool useInitRect_ = false;
	Rect init_rect_ = Rect(0, 0, 0, 0); // based on original image resolution, not downsample


	//optimization parameters
	int width_min_ = 31; //img.width/10
	int width_max_ = 240 / 2;//img.height/2
	int whstep_ = 4;
	int xystep_ = 4;

	//dynamic parameters
	int frameNum_ = 0;
	Rect roi_; //process region (mask), can be used for future tracking.
	double mu_inner_ = 50;//the current frame
	double mu_outer_ = 200;

	// only used for high intensity suppression
	double mu_inner0_ = 50;//the first frame
	double mu_outer0_ = 200;

	//--------------------output--------------------
	Rect pupil_rect_coarse_; //coarse
	Rect outer_rect_coarse_;
	double max_response_coarse_ = -255;
	vector<Rect> inner_rectlist_; //detected rect for each w
	Point2f center_coarse_;

	Rect pupil_rect_fine_; //refine
	Point2f center_fine_;

	size_t iterate_count_ = 0;



private:
	Size target_resolution_ = Size(320, 240);
	double ratio_downsample_;
	Rect imgboundary_;
	Rect init_rect_down_;

	// intergral_img: a matrix, not a image, and its values are always large.So we 
	//   set its type <int>
	Mat integral_img_;// size: (M+1)*(N+1)
};

#endif