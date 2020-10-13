#include "m_PupilDetectorHaar.h"

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
		init_rect_down_ = rectScale(init_rect_, 1 / ratio_downsample_, false);

		//xy strategy 2: adaptive margin
		{
			if (init_rect_down_.x < 35)
				boundary |= Rect(0, margin, margin, img_down.rows - 2 * margin);
			if (init_rect_down_.y < 35)
				boundary |= Rect(margin, 0, img_down.cols - 2 * margin, margin);
			if (init_rect_down_.x + init_rect_down_.width > 320 - 35)
				boundary |= Rect(img_down.cols - margin, margin, margin, img_down.rows - 2 * margin);
			if (init_rect_down_.y + init_rect_down_.height > 320 - 35)
				boundary |= Rect(margin, img_down.rows - margin, img_down.cols - 2 * margin, margin);
		}
		if (frameNum_ == 1)
			//set a small "roi" for the first frame.
			roi_ = rectScale(init_rect_down_, 2)&boundary;
		else
			roi_ = boundary;


		//w strategy
		//w can not be too small, otherwise be easily affected by light spots.
		width_min_ = init_rect_down_.width;
		width_max_ = width_min_ * 3.0 / 2;
	}
}



void PupilDetectorHaar::coarseDetection(const Mat& img_down, Rect& pupil_rect_coarse,
	Rect& outer_rect_coarse, double& max_response_coarse, double& mu_inner, double& mu_outer)
{
	initialSearchRange(img_down);

	//2.1 integral image
	cv::integral(img_down, integral_img_); // size: (M+1)*(N+1)


	//save the detail of Haar results.
#ifdef HAAR_TEST
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
	max_response_coarse = -255;
	for (int width = width_min_; width <= width_max_; width += whstep_)
	{
		//height is not used in here actually. It is used for furture extension.
		int height_min = width;
		for (int height = height_min; height <= width; height += whstep_)
		{
			Rect pupil_rect, outer_rect;
			double mu_inner_t, mu_outer_t;
			auto max_response = getResponseMap(integral_img_, width, height, 
				ratio_outer_, useSquareHaar_, kf_, roi_, xystep_, 
				pupil_rect, outer_rect, mu_inner_t, mu_outer_t);

			if (max_response_coarse < max_response)
			{
				pupil_rect_coarse = pupil_rect;
				//outer_rect_coarse_ = outer_rect;
				//max_response_coarse = max_response;
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
		vector<Rect> inner_rectlist;
		rectSuppression(rectlistAll, responselistAll, inner_rectlist, responselist);
		//Point2f initCenter((params.init_rect.x + params.init_rect.width) / 2,
		//	(params.init_rect.y + params.init_rect.height) / 2);

		//based on the distance to the initial center.
		Point2f initCenter((init_rect_down_.x + init_rect_down_.width) / 2,
			(init_rect_down_.y + init_rect_down_.height) / 2);
		double dis = 10000;
		for (int i = 0; i < inner_rectlist.size(); i++)
		{
#ifdef HAAR_TEST
			Rect outer_rect = rectScale(inner_rectlist[i], ratio_outer_, true, useSquareHaar_);
			draw(img_BGR, inner_rectlist[i], outer_rect, responselist[i]);
#endif

			Point2f iCenter(inner_rectlist[i].x + (inner_rectlist[i].width) / 2,
				inner_rectlist[i].y + (inner_rectlist[i].height) / 2);
			double dis_t = norm(initCenter - iCenter);
			if (dis_t < dis)
			{
				pupil_rect_coarse = inner_rectlist[i];
				dis = dis_t;
			}

			//scale inner rect to the resolution of original image for output.
			inner_rectlist[i] = rectScale(inner_rectlist[i], ratio_downsample_, false);
		}
		inner_rectlist_ = inner_rectlist;
	}
	
	outer_rect_coarse = rectScale(pupil_rect_coarse, ratio_outer_, true, useSquareHaar_)&imgboundary_;
	max_response_coarse = getResponseValue(integral_img_, pupil_rect_coarse, outer_rect_coarse, kf_,
		mu_inner, mu_outer);
}



void PupilDetectorHaar::fineDetection(const Mat& img_down, const Rect& pupil_rect_coarse, Rect& pupil_rect_fine)
{
	//detected rect expand a bit
	double expand_ratio = 1.42;//ratio=2 is too large
	Rect expand_rect = rectScale(pupil_rect_coarse, expand_ratio)&imgboundary_;
	Mat img = img_down(expand_rect);
	

	//3.1 Thresholding: mu_inner_ is always larger than the intensity of pupil pixels.
	Mat img_bw;
	{
		double thresh = mu_inner_;
		threshold(img, img_bw, thresh, 255, cv::THRESH_BINARY_INV);
		//threshold(img, img_bw, thresh, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
	}
	

	//3.2 Region dilation
	{
		Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
		cv::dilate(img_bw, img_bw, kernel, Point(-1, -1), 1);
	}

	//3.3 Region selecting, pupil region
	Mat labels, stats, centroids;
	cv::connectedComponentsWithStats(img_bw, labels, stats, centroids);
	Mat area = stats.col(cv::CC_STAT_AREA); //at least two elements
	area.at<int>(0, 0) = 0;//the first element is the whole image, not consider
	Mat stats_t;
	for (int i = 1; i < area.rows; i++)
		if (area.at<int>(i) > 0.04*img_bw.cols*img_bw.rows) //0.01=1/10*1/10
			stats_t.push_back(stats.row(i));

	//if stats_t.row ==0, there must be some error.
	//if (stats_t.rows == 0)
	//	throw("wrong bw image!");

	if (stats_t.rows == 1)
	{
		pupil_rect_fine = Rect(stats_t.at<int>(0, 0) + expand_rect.x,
			stats_t.at<int>(0, 1) + expand_rect.y,
			stats_t.at<int>(0, 2), stats_t.at<int>(0, 3));
		return;
	}


	//strategy1：选择面积最大的，可能被其它黑斑影响，比如iris或者睫毛
	//{
	//	Point maxLoc; //x坐标必然是0，y坐标表示index
	//	double maxval;
	//	mymax(area, maxLoc, maxval);
	//	//img_bw = labels == maxLoc.y;
	//	int index = maxLoc.y;
	//	pupil_rect2_ = Rect(stats.at<int>(index, 0) + base_rect.x,
	//		stats.at<int>(index, 1) + base_rect.y,
	//		stats.at<int>(index, 2), stats.at<int>(index, 3));
	//}

	//strategy2：选择包含中心的区域
	//int index = labels.at<int>(img_bw.cols/2, img_bw.rows/2);
	//if (index)
	//{
	//	pupil_rect2_ = Rect(stats.at<int>(index, 0) + base_rect.x,
	//		stats.at<int>(index, 1) + base_rect.y,
	//		stats.at<int>(index, 2), stats.at<int>(index, 3));
	//	if (pupil_rect2_.width < pupil_rect_.width/3)
	//		pupil_rect2_ = pupil_rect_; //阈值得到区域太小时
	//}
	//else //无区域的时候
	//	pupil_rect2_ = pupil_rect_;

	//strategy3：最大两个区域合并构成(如果s1/s2<10)，
	//不合适，对于之后的椭圆提取不利，不知道如何抉择

	//strategy4：优先中心包含的，没有则选择最大的
	//{
	//	int index = labels.at<int>(img_bw.cols / 2, img_bw.rows / 2);
	//	if (!index || area.at<int>(index) < 0.04*img_bw.cols*img_bw.rows)
	//	{
	//		Point maxLoc; //x坐标必然是0，y坐标表示index
	//		double maxval;
	//		mymax(area, maxLoc, maxval);
	//		index = maxLoc.y;
	//	}
	//	pupil_rect2_ = Rect(stats.at<int>(index, 0) + base_rect.x,
	//		stats.at<int>(index, 1) + base_rect.y,
	//		stats.at<int>(index, 2), stats.at<int>(index, 3));
	//}

	//strategy 5：the region through the image center, otherwise the darkest region.
	{
		int index = labels.at<int>(img_bw.cols / 2, img_bw.rows / 2);
		if (index && area.at<int>(index) > 0.04*img_bw.cols*img_bw.rows)
			pupil_rect_fine = Rect(stats.at<int>(index, 0) + expand_rect.x,
				stats.at<int>(index, 1) + expand_rect.y,
				stats.at<int>(index, 2), stats.at<int>(index, 3));
		else
		{
			Rect rect1 = Rect(stats_t.at<int>(0, 0), stats_t.at<int>(0, 1),
				stats_t.at<int>(0, 2), stats_t.at<int>(0, 3));
			Rect rect2 = Rect(stats_t.at<int>(1, 0), stats_t.at<int>(1, 1),
				stats_t.at<int>(1, 2), stats_t.at<int>(1, 3));
			Point centroid1(rect1.x + rect1.width / 2, rect1.y + rect1.height / 2);
			Point centroid2(rect2.x + rect2.width / 2, rect2.y + rect2.height / 2);
			int intensity1 = img.at<uchar>(centroid1);
			int intensity2 = img.at<uchar>(centroid2);
			if (abs(intensity1 - intensity2) < 5)
				pupil_rect_fine = rect1 | rect2;
			else
			{
				if (intensity1 > intensity2)
					pupil_rect_fine = rect2;
				else
					pupil_rect_fine = rect1;
			}
			pupil_rect_fine = pupil_rect_fine + expand_rect.tl();
		}
	}
}







double PupilDetectorHaar::getResponseMap(const Mat & integral_img, \
	int width, int height, double ratio, bool useSquareHaar, double kf, Rect roi, int xystep,
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
	for (int x = xmin; x <= xmax; x += xystep)
		for (int y = ymin; y <= ymax; y += xystep)
		{
			Rect pupil_rect0(x, y, width, height);
			Rect outer_rect0 = rectScale(pupil_rect0, ratio, true, useSquareHaar);

			double mu_inner0, mu_outer0;
			auto f = getResponseValue(integral_img, pupil_rect0, outer_rect0, kf, mu_inner0, mu_outer0);
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



