// Zhonghua Wan @ 20190409
// Class for pupil features extracion
//modify time: 2020.10.7
//modify time: 2019.4.9
//create time: 2019.1.19

#include "m_PupilDetectorHaar.h"

#include <opencv2/opencv.hpp>

#include <my_lib.h>

#define XYSTEP 4
//#define HAAR_TEST
//optimization: exhaustive algorithm
//1 response,
//2 haarResponseMap (hot map)
//3 Eye image with haar rect


using namespace std;


void PupilDetectorHaar::detect(const Mat &img_gray, const HaarParams& params)
{
	//guarantee the img_gray is UIN8 [0,255], not float [0,1].
	CV_Assert(img_gray.depth() == cv::DataType<uchar>::depth);

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
		int tau;
		//tau = max(params.mu_outer, params.mu_inner + 30);
		if (params.mu_outer - params.mu_inner > 30)
			tau = params.mu_outer;
		else
			tau = params.mu_inner + 30;
		filterLight(img_down, img_down, tau);
	}
	
	//-------------------------- 2 Coarse Detection --------------------------
	coarseDetection(img_down);
	

	//-------------------------- 3 Fine Detection --------------------------
	//if the image has bad quality, no fine detection
	if (mu_outer_ - mu_inner_ < 5)
		pupil_rect_fine_ = pupil_rect_coarse_;
	else
		fineDetection(img_down);

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
	outer_rect_fine_ = rectScale(outer_rect_fine_, ratio_downsample_, false);
}


// initial the search ranges of w and xy.
void PupilDetectorHaar::initialSearchRange(const Mat& img_down)
{
	//xy strategy1: [contant margin]， If the feature is close to the image boundary, it will not be robust.
	Rect boundary = imgboundary_;
	//If the pupil is in the edges，margin is set to half of the minimum pupil.
	int margin = img_down.rows / 10 / 2;
	boundary = Rect(margin, margin, img_down.cols - 2 * margin, img_down.rows - 2 * margin);
	
	if (!useInitRect_)
		roi_ = boundary;
	else
	{
		init_rect_down = rectScale(init_rect_, 1 / ratio_downsample_, false);

		//xy strategy 2: adaptive margin
		{
			if (init_rect_down.x < 35)
				boundary |= Rect(0, margin, margin, img_down.rows - 2 * margin);
			if (init_rect_down.y < 35)
				boundary |= Rect(margin, 0, img_down.cols - 2 * margin, margin);
			if (init_rect_down.x + init_rect_down.width > 320 - 35)
				boundary |= Rect(img_down.cols - margin, margin, margin, img_down.rows - 2 * margin);
			if (init_rect_down.y + init_rect_down.height > 320 - 35)
				boundary |= Rect(margin, img_down.rows - margin, img_down.cols - 2 * margin, margin);
		}
		roi_ = rectScale(roi_, 1 / ratio_downsample_, false)&boundary;


		//w strategy
		//w can not be too small, otherwise be easily affected by light spots.
		width_min_ = init_rect_down.width;
		width_max_ = width_min_ * 3.0 / 2;
	}
}




void PupilDetectorHaar::coarseDetection(const Mat& img_down)
{
	initialSearchRange(img_down);

	//2.1 integral image
	cv::integral(img_down, integral_img); // size: (M+1)*(N+1)


#ifdef HAAR_TEST
#define XYSTEP 1 //重新定义XYSTEP，从而可以显示完整的response map
	whstep = 4;

	ofstream fs("Harr_output" + to_string(int(ratio)) + "+" + getCurrentTimeStr());

	fs << "ratio" << ratio << endl;
	fs << "width	response" << endl;
	fs << fixed << setprecision(2);

	Mat img_BGRall;
	img2BGR(img_down, img_BGRall);
#endif

	//2.2 Computes Haar response.
	//search the best w,h,x,y for Haar-like features.
	//detect a best rect for each width to constitute rectlist.
	vector<Rect> rectlistAll;
	vector<double> responselistAll;
	for (int width = width_min_; width <= width_max_; width += whstep_)
	{
		//height is not used in here actually. It is used for furture extension.
		int height_min = width;
		for (int height = height_min; height <= width; height += whstep_)
		{
			Rect pupil_rect, outer_rect;
			double mu_inner, mu_outer;
			auto max_response = getResponseMap(integral_img, ratio_outer_,
				width, height, roi_, useSquareHaar_, pupil_rect, outer_rect, mu_inner, mu_outer);

			if (max_response_coarse_ < max_response)
			{
				pupil_rect_coarse_ = pupil_rect;
				//outer_rect_coarse_ = outer_rect;
				//max_response_coarse_ = max_response;
				//mu_inner_ = mu_inner;
				//mu_outer_ = mu_outer;
			}

			rectlistAll.push_back(pupil_rect);
			responselistAll.push_back(max_response);

#ifdef HAAR_TEST
			Mat img_BGR;
			img2BGR(img_down, img_BGR);
			draw(img_BGR, pupil_rect, outer_rect, max_response);
			draw(img_BGRall, pupil_rect, outer_rect, max_response);
			imshow("Haar features with eye", img_BGR);
			imshow("img BGRall", img_BGRall);
			waitKey(30);
			fs << pupil_rect.x*ratio_downsample << "	" << pupil_rect.y*ratio_downsample << "	"
				<< pupil_rect.width * ratio_downsample << "	" << max_response << endl;
#endif
		}//end for height
	}//end for width

	//plot all detected rect.
#ifdef HAAR_TEST
	Mat img_BGR;
	img2BGR(img_down, img_BGR);
	rectangle(img_BGR, roi_, BLUE, 1, 8);

	//plot all detected rectangles for each w
	for (int i = 0; i < rectlistAll.size(); i++)
	{
		Rect outer_rect;
		outer_rect = rectScale(rectlistAll[i], ratio_outer_, true, useSquareHaar_);
		rectangle(img_BGR, rectlistAll[i], RED, 1, 8);
	}
#endif
	
	//2.3 Rectangle list suppression. 
	if (useInitRect_)
	{
		//non-maximum suppression
		vector<double> responselist;
		rectSuppression(rectlistAll, responselistAll, inner_rectlist_, responselist);
		//Point2f initCenter((params.init_rect.x + params.init_rect.width) / 2,
		//	(params.init_rect.y + params.init_rect.height) / 2);

		//based on the distance to the initial center.
		Point2f initCenter((init_rect_down.x + init_rect_down.width) / 2,
			(init_rect_down.y + init_rect_down.height) / 2);
		double dis = 10000;
		for (int i = 0; i < inner_rectlist_.size(); i++)
		{
#ifdef HAAR_TEST
			Rect outer_rect = rectScale(inner_rectlist_[i], ratio_outer_, true, useSquareHaar_);
			draw(img_BGR, inner_rectlist_[i], outer_rect, responselist[i]);
#endif

			Point2f iCenter(inner_rectlist_[i].x + (inner_rectlist_[i].width) / 2,
				inner_rectlist_[i].y + (inner_rectlist_[i].height) / 2);
			double dis_t = norm(initCenter - iCenter);
			if (dis_t < dis)
			{
				pupil_rect_coarse_ = inner_rectlist_[i];
				dis = dis_t;
			}

			//scale inner rect to the resolution of original image for output.
			inner_rectlist_[i] = rectScale(inner_rectlist_[i], ratio_downsample_, false);
		}

	}
	
	outer_rect_coarse_ = rectScale(pupil_rect_coarse_, ratio_outer_, true, useSquareHaar_)&imgboundary_;
	max_response_coarse_ = getResponseValue(integral_img, pupil_rect_coarse_, outer_rect_coarse_, 
		mu_inner_, mu_outer_);
}







void PupilDetectorHaar::detectIris(const Mat &img_gray, const HaarParams& params)
{
	//TODO(zhwan): guarantee the imgGray UIN8 [0,255], not float [0,1].
	CV_Assert(img_gray.depth() == cv::DataType<uchar>::depth);

	max_response_coarse_ = -255;

	//1 downsampling
	Mat img_down;
	double ratio_downsample;
	{
		Size resolution = Size(320, 240);
		ratio_downsample = max(img_gray.cols*1.0f / resolution.width,
			img_gray.rows*1.0f / resolution.height);
		resize(img_gray, img_down, Size(img_gray.cols / ratio_downsample,
			img_gray.rows / ratio_downsample));
	}


	double ratio = params.outer_ratio;
	whstep_ = params.wh_step;
	xystep_ = params.xy_step;
	kf_ = params.kf;
	//init_rect在params中是原图的scale，在这里是downsample之后的
	Rect init_rect = rectScale(params.init_rect, 1 / ratio_downsample, false);

	//w策略
	int width_min = 31; //img.width/10
	int width_max = 240 / 2;//img.height/2
	int height_min;
	int height_max;

	//xy策略：默认使用固定margin
	//太靠边的位置不考虑，这样不仅提升速度，而且提升稳定性，因为边缘的可能相应值高
	Rect imgboundary = Rect(0, 0, img_down.cols, img_down.rows); //不用margin
	Rect boundary = imgboundary;
	//XY策略1 用margin
	int margin = img_down.rows / 10 / 2;//假设瞳孔在边缘，margin设为最小瞳孔时候的一半
	boundary = Rect(margin, margin, img_down.cols - 2 * margin, img_down.rows - 2 * margin);
	Rect roi = boundary; //直接使用则不采用输入ROI

	if (useInitRect_)
	{
		//图像I策略：光强抑制
		//int tau;
		//if (params.mu_outer - params.mu_inner > 30)
		//	tau = params.mu_outer;
		//else
		//	tau = params.mu_inner + 30;
		//filterLight(img_down, img_down, tau);

		width_min = init_rect.width;//这个不能变小，变小容易受光斑的影响，最后检测的很小
		//width_max = width_min * 3.0 / 2;
		width_max = width_min;
		height_min = init_rect.height;
		height_max = init_rect.height;

		//XY策略2 选择性boundary，考虑init rect靠近边缘
		{
			if (init_rect.x < 35)
				boundary |= Rect(0, margin, margin, img_down.rows - 2 * margin);
			if (init_rect.y < 35)
				boundary |= Rect(margin, 0, img_down.cols - 2 * margin, margin);
			if (init_rect.x + init_rect.width > 320 - 35)
				boundary |= Rect(img_down.cols - margin, margin, margin, img_down.rows - 2 * margin);
			if (init_rect.y + init_rect.height > 320 - 35)
				boundary |= Rect(margin, img_down.rows - margin, img_down.cols - 2 * margin, margin);
		}

		//XY策略3 采用输入ROI
		roi = rectScale(params.roi, 1 / ratio_downsample, false)&boundary;
	}
	//width_max = min(width_max, int(min(img_down.rows, img_down.cols) / ratio));




	//2 integral image
	cv::integral(img_down, integral_img); // size: (M+1)*(N+1)



#ifdef HAAR_TEST
#define XYSTEP 1 //重新定义XYSTEP，从而可以显示完整的response map
	whstep = 4;

	ofstream fs("Harr_output" + to_string(int(ratio)) + "+" + getCurrentTimeStr());

	fs << "ratio" << ratio << endl;
	fs << "width	response" << endl;
	fs << fixed << setprecision(2);

	Mat img_BGRall;
	img2BGR(img_down, img_BGRall);
#endif

	//3 Computes Haar response.
	vector<Rect> rectlist;
	vector<double> responselist;
	vector<double> responselist2;

	// Decreses the search time.
	for (int width = width_min; width <= width_max; width += whstep_)
	{
		//cout<<"这个线程是："<<this_thread::get_id()<<endl;
		//int height_min = width;//heigth的循环实际上在这里没用，因为最大与最小范围是一个
		for (int height = height_min; height <= height_max; height += whstep_)
		{
			Rect pupil_rect, outer_rect;
			//rect与图像boundary相交，而不会超出图像范围
			double mu_inner, mu_outer;
			auto max_response = getResponseMap(integral_img, ratio,
				width, height, roi, useSquareHaar_, pupil_rect, outer_rect, mu_inner, mu_outer);

			if (max_response_coarse_ < max_response)
			{
				max_response_coarse_ = max_response;
				pupil_rect_coarse_ = pupil_rect;
				outer_rect_coarse_ = outer_rect;
				mu_inner_ = mu_inner;
				mu_outer_ = mu_outer;
			}

			rectlist.push_back(pupil_rect);
			responselist.push_back(max_response);

#ifdef HAAR_TEST
			Mat img_BGR;
			img2BGR(img_down, img_BGR);
			draw(img_BGR, pupil_rect, outer_rect, max_response);
			draw(img_BGRall, pupil_rect, outer_rect, max_response);
			imshow("Haar features with eye", img_BGR);
			imshow("img BGRall", img_BGRall);
			waitKey(30);
			fs << pupil_rect.x*ratio_downsample << "	" << pupil_rect.y*ratio_downsample << "	"
				<< pupil_rect.width * ratio_downsample << "	" << max_response << endl;
#endif
		}//end for height
	}//end for width

	Mat img_BGR2;
	img2BGR(img_down, img_BGR2);
	//绘制ROI
	rectangle(img_BGR2, roi, BLUE, 1, 8);

	//rectlist抑制，基于响应值 + 距初始距离
	if (useInitRect_)
	{
		rectSuppression(rectlist, responselist, inner_rectlist_, responselist2);
		Point2f initCenter((params.init_rect.x + params.init_rect.width) / 2,
			(params.init_rect.y + params.init_rect.height) / 2);
		double dis = 10000;
		for (int i = 0; i < inner_rectlist_.size(); i++)
		{
			Rect outer_rect;
			outer_rect = rectScale(inner_rectlist_[i], ratio, true, useSquareHaar_);
			draw(img_BGR2, inner_rectlist_[i], outer_rect, responselist2[i]);

			Point2f iCenter(inner_rectlist_[i].x + (inner_rectlist_[i].width) / 2,
				inner_rectlist_[i].y + (inner_rectlist_[i].height) / 2);
			double dis_t = norm(initCenter - iCenter);
			if (dis_t < dis)
			{
				pupil_rect_coarse_ = inner_rectlist_[i];
				dis = dis_t;
			}

			//将每个inner rect scale到原始图像的尺寸
			inner_rectlist_[i] = rectScale(inner_rectlist_[i], ratio_downsample, false);
		}
		outer_rect_coarse_ = rectScale(pupil_rect_coarse_, ratio, true, useSquareHaar_)&imgboundary;

		max_response_coarse_ = getResponseValue(integral_img, pupil_rect_coarse_, outer_rect_coarse_, mu_inner_, mu_outer_);
	}


	//图像质量太差的时候，不进行优化，不仅没意义，而且可能有反效果
	//if (mu_outer_ - mu_inner_ < 5) //尤其小心出现负值
	//	pupil_rect2_ = pupil_rect_;
	//else
	//	detectToFine2(img_down);
	//rectangle(img_BGR2, pupil_rect2_, BLUE, 1, 8);

	imshow("tmp", img_BGR2);

	//Rect整体放大，所以需要使用另一种scale方式
	pupil_rect_coarse_ = rectScale(pupil_rect_coarse_, ratio_downsample, false);
	outer_rect_coarse_ = rectScale(outer_rect_coarse_, ratio_downsample, false);
	//detectToFine 进一步优化得到
	pupil_rect_fine_ = rectScale(pupil_rect_fine_, ratio_downsample, false);
	outer_rect_fine_ = rectScale(outer_rect_fine_, ratio_downsample, false);
}






/* gets max Haar response.

@param pupil_rect output rect.
*/
double PupilDetectorHaar::getResponseMap(const Mat & integral_img, \
	double ratio, int width, int height, Rect roi, bool useSquareHaar,
	Rect& pupil_rect, Rect& outer_rect, double& mu_inner, double& mu_outer)
{
	//(x,y) is the lefttop corner of pupil_rect.
	//The range of (x,y) has two kinds: the first is the whole image, the second is the valid range.

#ifdef HAAR_TEST
	Mat response_map = Mat::zeros(integral_img.size(), CV_32F);
	xystep = 1;
	roi = Rect(0, 0, integral_img.cols, integral_img.rows);
#endif
	decltype(max_response_coarse_) max_response = -255;

	// Positions that are too close to peripheries are not considered, which not only improves the speed,
	// but also improves the robustness. Because peripheries are dark.
	int xmin = roi.x;
	int xmax = roi.width + roi.x - width;
	int ymin = roi.y;
	int ymax = roi.height + roi.y - height;
	for (int x = xmin; x <= xmax; x += xystep_)
		for (int y = ymin; y <= ymax; y += xystep_)
		{
			Rect pupil_rect0(x, y, width, height);
			Rect outer_rect0 = rectScale(pupil_rect0, ratio, true, useSquareHaar);

			double mu_inner0, mu_outer0;
			auto f = getResponseValue(integral_img, pupil_rect0, outer_rect0, mu_inner0, mu_outer0);
			if (max_response < f)
			{
				max_response = f;
				pupil_rect = pupil_rect0;
				outer_rect = outer_rect0;
				mu_inner = mu_inner0;
				mu_outer = mu_outer0;
			}
#ifdef HAAR_TEST
			response_map.at<float>(y + height / 2, x + width / 2) = f;
#endif
		}
#ifdef HAAR_TEST
	showHotMap(response_map);
#endif

	return max_response;
}



