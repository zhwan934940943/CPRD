#pragma once

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
	cv::integral(img_down, integral_img_); // size: (M+1)*(N+1)



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
			auto max_response = getResponseMap(integral_img_, ratio,
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

		max_response_coarse_ = getResponseValue(integral_img_, pupil_rect_coarse_, outer_rect_coarse_, mu_inner_, mu_outer_);
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



void LPWTest_Haar_iris()
{
	LPW_init_iris();

	//method init
	method_name = "Haar";


	//for (int i = 0; i != caselist.size(); i++)//caselist.size()
	int i = 50; //单个case测试
	//for (int i = 62; i != 64; i++)//caselist.size()
	{
		//这个最好放置在里面，否则需要每次初始化params.mu_inner，这样才不会使用上一次的数据
		HaarParams params;
		params.initRectFlag = true;
		params.squareHaarFlag = false;
		//kernel策略1：ratio多选1
		params.outer_ratio = 1.42; //1.42, 2, 3, 4, 5, 6, 7
		//kernel策略2：mu_inner权重
		params.kf = 2.0; //1,1.1,1.2,1.3
		params.wh_step = 2;//2,3,4
		params.xy_step = 4;
		cout << "LPWTest_Haar" << "\n" << "initRectFlag=" << params.initRectFlag << "	"
			<< "squareHaarFlag=" << params.squareHaarFlag << "\n"
			<< "r=" << params.outer_ratio << "	" << "kf=" << params.kf << "	"
			<< "wh_step=" << params.wh_step << "	" << "xy_step=" << params.xy_step << "\n";

		if (params.initRectFlag)
		{
			params.init_rect = init_rectlist[i];

			//经测试，初始scale 2,3,4都可得到初始rect的响应
			params.roi = rectScale(init_rectlist[i], 2);//仅第一帧的ROI
			//可以考虑下面的方法
			//params.roi = (x, y +h/2-w/2, w, w)//init_rectlist[i];
		}


		string casename = caselist[i];
		cout << casename << endl;

		//groundtruth file. e.g., 1-1.txt
		ifstream fin_groundtruth(dataset_dir + casename + ".txt");
		{
			if (!fin_groundtruth.is_open())
				throw("cannot open file" + casename);
		}


		//保存路径
		string errorfile_name = "r" + to_string((params.outer_ratio)) + " "
			+ dataset_name + " " + casename + ".txt";
		ofstream fout(results_dir + method_name + "/" + errorfile_name);
		{
			if (!fout.is_open())
				throw("cannot open file" + errorfile_name);
		}


		double x, y;
		VideoCapture cap(dataset_dir + casename + ".avi");

		PupilDetectorHaar lasthaar;
		bool firstFrameFlag = true;
		while (fin_groundtruth >> x)
		{
			fin_groundtruth >> y;

			Mat frame;
			cap >> frame;
			{
				if (frame.empty())
					throw("image import error!");
			}

			Mat img_gray;
			img2Gray(frame, img_gray);
			//filterImg(img_gray, img_gray);


			PupilDetectorHaar haar;
			haar.detectIris(img_gray, params);

			if (firstFrameFlag)
			{
				params.mu_inner = haar.mu_inner_;
				params.mu_outer = haar.mu_outer_;
				firstFrameFlag = false;
			}


			//blink detection
			//第一帧同样适用，因为初始max_response_=-255
			//bool abnormal_flag = false;
			/*if (haar.max_response_ < lasthaar.max_response_ - 10)
			{
				haar = lasthaar;
				abnormal_flag = true;
			}
			else
				lasthaar = haar;*/

				//xy策略：ROI
				//params.roi = rectScale(haar.pupil_rect_, 4);
			params.roi = Rect(0, 0, img_gray.cols, img_gray.rows);

			Mat img_haar;
			cv::cvtColor(img_gray, img_haar, CV_GRAY2BGR);
			haar.drawCoarse(img_haar);
			rectangle(img_haar, haar.pupil_rect_fine_, BLUE, 1, 8);

			bool flag_sucess_inner = haar.pupil_rect_coarse_.contains(Point2f(x, y));
			bool flag_sucess_outer = haar.outer_rect_coarse_.contains(Point2f(x, y));

			//flag: 是否备选的rect内含有pupil
			bool flag_sucess_candidates = false;
			vector<Rect> rectlist = haar.inner_rectlist_;
			for (int i = 0; i < rectlist.size(); i++)
			{
				if (rectlist[i].contains(Point2f(x, y)))
				{
					flag_sucess_candidates = true;
					break;
				}
			}

			Point2f center(haar.pupil_rect_coarse_.x + haar.pupil_rect_coarse_.width*1.0f / 2,
				haar.pupil_rect_coarse_.y + haar.pupil_rect_coarse_.height*1.0f / 2);
			double error = norm(center - Point2f(x, y));

			bool flag_sucess_inner2 = haar.pupil_rect_fine_.contains(Point2f(x, y));
			Point2f center2(haar.pupil_rect_fine_.x + haar.pupil_rect_fine_.width*1.0f / 2,
				haar.pupil_rect_fine_.y + haar.pupil_rect_fine_.height*1.0f / 2);
			double error2 = norm(center2 - Point2f(x, y));


			//---------------------以上为Haar检测及其结果----------------------------

			//Rect boundary(0, 0, img_gray.cols, img_gray.rows);
			//double validRatio = 1.2; //策略：1.42
			//Rect validRect = rectScale(haar.pupil_rect2_, validRatio)&boundary;
			//Mat img_pupil = img_gray(validRect);
			//GaussianBlur(img_pupil, img_pupil, Size(5, 5), 0, 0);


			Point2f center3;

			//cv::RotatedRect ellipse_rect;
			//if (haar.mu_outer_ - haar.mu_inner_ < 10) //策略：0,10,20
			//	center3 = center2;
			//else
			//{
			//	//edges提取
			//	//自适应阈值canny method
			//	PupilExtractionMethod detector;
			//	/*Mat edges;
			//	Rect inlinerRect = haar.pupil_rect2_ - haar.pupil_rect2_.tl();
			//	detector.detectPupilContour(img_pupil, edges, inlinerRect);*/

			//	double tau1 = 1 - 20.0 / img_pupil.cols;//策略：10，20
			//	Mat edges = canny_pure(img_pupil, false, false, 64 * 2, tau1, 0.5);

			//	Mat edges_filter;
			//	{
			//		int tau;
			//		//if (haar.mu_outer_ - haar.mu_inner_ > 30)
			//		//	tau = params.mu_outer;
			//		//else
			//		tau = haar.mu_inner_ + 100;
			//		//光强抑制	
			//		//PupilDetectorHaar::filterLight(img_t, img_t, tau);

			//		//1 利用光强过强抑制部分edges
			//		Mat img_bw;
			//		threshold(img_pupil, img_bw, tau, 255, cv::THRESH_BINARY);
			//		Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
			//		dilate(img_bw, img_bw, kernel, Point(-1, -1), 1);
			//		edges_filter = edges & ~img_bw;

			//		//2 利用curves size过滤较小的片段
			//		edgesFilter(edges_filter);
			//	}
			//	imshow("edges", edges);
			//	imshow("edgesF", edges_filter);

			//	//int K;//iterations
			//	//{
			//	//	double p = 0.99;	// success rate 0.99
			//	//	double e = 0.7;		// outlier ratio, 0.7效果很好，但是时间长
			//	//	K = cvRound(log(1 - p) / log(1 - pow(1 - e, 5)));
			//	//}
			//	//RotatedRect ellipse_rect;
			//	//detector.fitPupilEllipse(edges_filter, ellipse_rect, K);


			//	//利用RANSAC
			//	detector.fitPupilEllipseSwirski(img_pupil, edges_filter, ellipse_rect);
			//	ellipse_rect.center = ellipse_rect.center + Point2f(validRect.tl());
			//	if (haar.pupil_rect2_.contains(ellipse_rect.center) && (ellipse_rect.size.width > 0))
			//	{
			//		center3 = ellipse_rect.center;
			//		drawMarker(img_haar, center3, Scalar(0, 0, 255));
			//		ellipse(img_haar, ellipse_rect, RED);
			//	}
			//	else
			//		center3 = center2;


			//	//	//利用PuRe进一步提取
			//	//	//PuRe detectorPuRe;
			//	//	//Pupil pupil = detectorPuRe.run(img_pupil);
			//	//	//pupil.center = pupil.center + Point2f(validRect.tl());
			//}//end if


			double error3 = norm(center3 - Point2f(x, y));



			//save results
			{
				fout << flag_sucess_inner << "	" << flag_sucess_outer << "	"
					<< rectlist.size() << "	" << flag_sucess_candidates << "	"
					<< error << "	" << haar.max_response_coarse_ << "	"
					<< haar.mu_inner_ << "	" << haar.mu_outer_ << "	"
					//以下是detectToFine的结果
					<< flag_sucess_inner2 << "	" << error2 << "	"
					//以下为ellipse fitting结果
					<< error3 << endl;
			}
			//imshow("pupil region", img_pupil);
			imshow("Results", img_haar);
			waitKey(5);
		}//end while
		fin_groundtruth.close();
		fout.close();
	}//end for

}



void LPWTest_Haar()
{
	LPW_init();

	//method init
	method_name = "Haar";


	//for (int i = 0; i != caselist.size(); i++)//caselist.size()
	int i = 56; //test a case
	//for (int i = 62; i != 64; i++)//caselist.size()
	{
		//这个最好放置在里面，否则需要每次初始化params.mu_inner，这样才不会使用上一次的数据
		HaarParams params;
		params.initRectFlag = true;
		params.squareHaarFlag = false;
		//kernel策略1：ratio多选1
		params.outer_ratio = 1.42; //1.42, 2, 3, 4, 5, 6, 7
		//kernel策略2：mu_inner权重
		params.kf = 1.4; //1,1.1,1.2,1.3
		params.wh_step = 2;//2,3,4
		params.xy_step = 4;
		cout << "LPWTest_Haar" << "\n" << "initRectFlag=" << params.initRectFlag << "	"
			<< "squareHaarFlag=" << params.squareHaarFlag << "\n"
			<< "r=" << params.outer_ratio << "	" << "kf=" << params.kf << "	"
			<< "wh_step=" << params.wh_step << "	" << "xy_step=" << params.xy_step << "\n";

		if (params.initRectFlag)
		{
			params.init_rect = init_rectlist[i];

			//经测试，初始scale 2,3,4都可得到初始rect的响应
			params.roi = rectScale(init_rectlist[i], 2);//仅第一帧的ROI
			//可以考虑下面的方法
			//params.roi = (x, y +h/2-w/2, w, w)//init_rectlist[i];
		}


		string casename = caselist[i];
		cout << casename << endl;

		//groundtruth file. e.g., 1-1.txt
		ifstream fin_groundtruth(dataset_dir + casename + ".txt");
		{
			if (!fin_groundtruth.is_open())
				throw("cannot open file" + casename);
		}


		//保存路径
		string errorfile_name = "r" + to_string((params.outer_ratio)) + " "
			+ dataset_name + " " + casename + ".txt";
		ofstream fout(results_dir + method_name + "/" + errorfile_name);
		{
			if (!fout.is_open())
				throw("cannot open file" + errorfile_name);
		}


		double x, y;
		VideoCapture cap(dataset_dir + casename + ".avi");

		PupilDetectorHaar lasthaar;
		bool firstFrameFlag = true;
		bool secondFrameFlag = false;
		int frameCount = 0;

		cv::VideoWriter coarse_vid, fine_vid;
		//coarse_vid.open(to_string(i)+"coarse.avi", CV_FOURCC('M', 'J', 'P', 'G'), 95.0, cv::Size(640, 480), true);
		//fine_vid.open(to_string(i) + "fine.avi", CV_FOURCC('M', 'J', 'P', 'G'), 95.0, cv::Size(640, 480), true);

		while (fin_groundtruth >> x)
		{
			frameCount += 1;
			if (frameCount == 2)
				secondFrameFlag = true;
			else
				secondFrameFlag = false;

			fin_groundtruth >> y;

			Mat frame;
			cap >> frame;
			{
				if (frame.empty())
					throw("image import error!");
			}

			Mat img_gray;
			img2Gray(frame, img_gray);
			//filterImg(img_gray, img_gray);


			PupilDetectorHaar haar;
			haar.detect(img_gray);

			if (firstFrameFlag)
			{
				params.mu_inner = haar.mu_inner_;
				params.mu_outer = haar.mu_outer_;
				firstFrameFlag = false;
			}


			//blink detection
			//第一帧同样适用，因为初始max_response_=-255
			//bool abnormal_flag = false;
			/*if (haar.max_response_ < lasthaar.max_response_ - 10)
			{
				haar = lasthaar;
				abnormal_flag = true;
			}
			else
				lasthaar = haar;*/

				//xy策略：ROI
				//params.roi = rectScale(haar.pupil_rect_, 4);
			params.roi = Rect(0, 0, img_gray.cols, img_gray.rows);

			Mat img_haar;
			cv::cvtColor(img_gray, img_haar, CV_GRAY2BGR);
			haar.drawCoarse(img_haar);
			//coarse_vid << img_haar;
			//if (secondFrameFlag)
			//	imwrite(to_string(i)+"Coarse.png", img_haar);
			rectangle(img_haar, haar.pupil_rect_fine_, BLUE, 1, 8);
			//if (secondFrameFlag)
			//{
			//	imwrite(to_string(i) + "Fine.png", img_haar);
			//	break;
			//}
			//fine_vid << img_haar;

			bool flag_sucess_inner = haar.pupil_rect_coarse_.contains(Point2f(x, y));
			bool flag_sucess_outer = haar.outer_rect_coarse_.contains(Point2f(x, y));

			//flag: 是否备选的rect内含有pupil
			bool flag_sucess_candidates = false;
			vector<Rect> rectlist = haar.inner_rectlist_;
			for (int i = 0; i < rectlist.size(); i++)
			{
				if (rectlist[i].contains(Point2f(x, y)))
				{
					flag_sucess_candidates = true;
					break;
				}
			}

			Point2f center(haar.pupil_rect_coarse_.x + haar.pupil_rect_coarse_.width*1.0f / 2,
				haar.pupil_rect_coarse_.y + haar.pupil_rect_coarse_.height*1.0f / 2);
			double error = norm(center - Point2f(x, y));

			bool flag_sucess_inner2 = haar.pupil_rect_fine_.contains(Point2f(x, y));
			Point2f center2(haar.pupil_rect_fine_.x + haar.pupil_rect_fine_.width*1.0f / 2,
				haar.pupil_rect_fine_.y + haar.pupil_rect_fine_.height*1.0f / 2);
			double error2 = norm(center2 - Point2f(x, y));


			//---------------------以上为Haar检测及其结果----------------------------

			Rect boundary(0, 0, img_gray.cols, img_gray.rows);
			double validRatio = 1.2; //策略：1.42
			Rect validRect = rectScale(haar.pupil_rect_fine_, validRatio)&boundary;
			Mat img_pupil = img_gray(validRect);
			GaussianBlur(img_pupil, img_pupil, Size(5, 5), 0, 0);


			Point2f center3;

			cv::RotatedRect ellipse_rect;
			if (haar.mu_outer_ - haar.mu_inner_ < 10) //策略：0,10,20
				center3 = center2;
			else
			{
				//edges提取
				//自适应阈值canny method
				PupilExtractionMethod detector;
				/*Mat edges;
				Rect inlinerRect = haar.pupil_rect2_ - haar.pupil_rect2_.tl();
				detector.detectPupilContour(img_pupil, edges, inlinerRect);*/

				double tau1 = 1 - 20.0 / img_pupil.cols;//策略：10，20
				Mat edges = canny_pure(img_pupil, false, false, 64 * 2, tau1, 0.5);

				Mat edges_filter;
				{
					int tau;
					//if (haar.mu_outer_ - haar.mu_inner_ > 30)
					//	tau = params.mu_outer;
					//else
					tau = haar.mu_inner_ + 100;
					//光强抑制	
					//PupilDetectorHaar::filterLight(img_t, img_t, tau);

					//1 利用光强过强抑制部分edges
					Mat img_bw;
					threshold(img_pupil, img_bw, tau, 255, cv::THRESH_BINARY);
					Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
					dilate(img_bw, img_bw, kernel, Point(-1, -1), 1);
					edges_filter = edges & ~img_bw;

					//2 利用curves size过滤较小的片段
					edgesFilter(edges_filter);
				}
				imshow("edges", edges);
				imshow("edgesF", edges_filter);

				//int K;//iterations
				//{
				//	double p = 0.99;	// success rate 0.99
				//	double e = 0.7;		// outlier ratio, 0.7效果很好，但是时间长
				//	K = cvRound(log(1 - p) / log(1 - pow(1 - e, 5)));
				//}
				//RotatedRect ellipse_rect;
				//detector.fitPupilEllipse(edges_filter, ellipse_rect, K);


				//利用RANSAC
				detector.fitPupilEllipseSwirski(img_pupil, edges_filter, ellipse_rect);
				ellipse_rect.center = ellipse_rect.center + Point2f(validRect.tl());
				if (haar.pupil_rect_fine_.contains(ellipse_rect.center) && (ellipse_rect.size.width > 0))
				{
					center3 = ellipse_rect.center;
					drawMarker(img_haar, center3, Scalar(0, 0, 255));
					ellipse(img_haar, ellipse_rect, RED);
				}
				else
					center3 = center2;


				//	//利用PuRe进一步提取
				//	//PuRe detectorPuRe;
				//	//Pupil pupil = detectorPuRe.run(img_pupil);
				//	//pupil.center = pupil.center + Point2f(validRect.tl());
			}//end if


			double error3 = norm(center3 - Point2f(x, y));



			//save results
			{
				fout << flag_sucess_inner << "	" << flag_sucess_outer << "	"
					<< rectlist.size() << "	" << flag_sucess_candidates << "	"
					<< error << "	" << haar.max_response_coarse_ << "	"
					<< haar.mu_inner_ << "	" << haar.mu_outer_ << "	"
					//以下是detectToFine的结果
					<< flag_sucess_inner2 << "	" << error2 << "	"
					//以下为ellipse fitting结果
					<< error3 << endl;
			}
			//imshow("pupil region", img_pupil);
			imshow("Results", img_haar);
			waitKey(5);
		}//end while
		fin_groundtruth.close();
		fout.close();
		coarse_vid.release();
		fine_vid.release();
	}//end for

}


Rect boundary(0, 0, img_gray.cols, img_gray.rows);
double validRatio = 1.2; //策略：1.42
Rect validRect = rectScale(pupil_rect_fine_, validRatio)&boundary;
Mat img_pupil = img_gray(validRect);
GaussianBlur(img_pupil, img_pupil, Size(5, 5), 0, 0);


Point2f center_fitting;

cv::RotatedRect ellipse_rect;
//if (haar.mu_outer_ - haar.mu_inner_ < 10) //策略：0,10,20
//	center_fitting = center_fine;
//else
{
	//edges提取
	//自适应阈值canny method
	PupilExtractionMethod detector;
	/*Mat edges;
	Rect inlinerRect = haar.pupil_rect2_ - haar.pupil_rect2_.tl();
	detector.detectPupilContour(img_pupil, edges, inlinerRect);*/

	double tau1 = 1 - 20.0 / img_pupil.cols;//策略：10，20
	Mat edges = canny_pure(img_pupil, false, false, 64 * 2, tau1, 0.5);

	Mat edges_filter;
	{
		int tau;
		//if (haar.mu_outer_ - haar.mu_inner_ > 30)
		//	tau = params.mu_outer;
		//else
		tau = haar.mu_inner_ + 100;
		//光强抑制	
		//PupilDetectorHaar::filterLight(img_t, img_t, tau);

		//1 利用光强过强抑制部分edges
		Mat img_bw;
		threshold(img_pupil, img_bw, tau, 255, cv::THRESH_BINARY);
		Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5));
		dilate(img_bw, img_bw, kernel, Point(-1, -1), 1);
		edges_filter = edges & ~img_bw;

		//2 利用curves size过滤较小的片段
		edgesFilter(edges_filter);
	}
	imshow("edges", edges);
	imshow("edgesF", edges_filter);

	//int K;//iterations
	//{
	//	double p = 0.99;	// success rate 0.99
	//	double e = 0.7;		// outlier ratio, 0.7效果很好，但是时间长
	//	K = cvRound(log(1 - p) / log(1 - pow(1 - e, 5)));
	//}
	//RotatedRect ellipse_rect;
	//detector.fitPupilEllipse(edges_filter, ellipse_rect, K);


	//利用RANSAC
	detector.fitPupilEllipseSwirski(img_pupil, edges_filter, ellipse_rect);
	ellipse_rect.center = ellipse_rect.center + Point2f(validRect.tl());
	if (haar.pupil_rect_fine_.contains(ellipse_rect.center) && (ellipse_rect.size.width > 0))
	{
		center_fitting = ellipse_rect.center;
		drawMarker(img_coarse, center_fitting, Scalar(0, 0, 255));
		ellipse(img_coarse, ellipse_rect, RED);
	}
	else
		center_fitting = center_fine;


	//	//利用PuRe进一步提取
	//	//PuRe detectorPuRe;
	//	//Pupil pupil = detectorPuRe.run(img_pupil);
	//	//pupil.center = pupil.center + Point2f(validRect.tl());
}//end if