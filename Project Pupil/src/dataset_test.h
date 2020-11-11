#pragma once

#ifndef Dataset_Test_H_
#define Dataset_Test_H_

#include<chrono>
#include <opencv2/opencv.hpp>
#include "PuRe/PuRe.h"
#include "m_PupilDetectorHaar.h"

//#define RESULT_EXPORT
//#define VIDEO_EXPORT

using namespace std::chrono;


class DatasetTest
{
public:
	DatasetTest() {};


	void dataset_init(const string& dataset_name)
	{
		string allDatasets_dir = "D:/OneDrive - Platinum/mycodelib/pupil detection/";
		dataset_dir_ = allDatasets_dir + dataset_name + "/";
		cout << "dataset dir = " << dataset_dir_ << "\n";

		//1 input
		//caselist.txt has prefixes for each case.
		readStringList_txt(dataset_dir_ + "caselist.txt", caselist_);

		ifstream fin(dataset_dir_ + "init_rect_pupil.txt");
		double x, y, width, height;
		while (fin >> x)
		{
			fin >> y >> width >> height;
			init_rectlist_.push_back(Rect(x, y, width, height));
		}

		//2 results dir
		results_dir_ = allDatasets_dir + "results/";
		cout << "result dir = " << results_dir_ << "\n";
	}






	//------------------------------- My Haar method -------------------------------
	//The ground truth is the pupil center (x,y)
	void LPWTest_Haar(double ratio = 1.4)
	{
		string dataset_name = "LPW";
		dataset_init(dataset_name);

		string method_name = "Haar";
		double ratio_outer = ratio;//1.4, 2, 3, 4, 5, 6, 7
		double kf = 1; //1,1.1,1.2,1.3,1.4,...
		bool useSquareHaar = 0;
		bool useInitRect = 0;
		int whstep = 2;//2,3,4,...
		int xystep = 4;//2,3,4,...

		std::stringstream ss;
		ss << setprecision(3) << "r" << ratio_outer << " k" << kf << "/";
		string result_path = results_dir_ + method_name + "/" + ss.str();

		ofstream logfile(result_path + "logfile.txt");
		logfile << "LPWTest_Haar\n";
		logfile << "useInitRect = " << useInitRect << "	"
			<< "useSquareHaar = " << useSquareHaar << "\n"
			<< "r = " << ratio_outer << "	" << "kf = " << kf << "	"
			<< "wh_step = " << whstep << "	" << "xy_step = " << xystep << "\n";

		for (int i = 0; i != caselist_.size(); i++)//caselist.size()
		//int i = 50; //test a case
		//for (int i = 62; i != 64; i++)//caselist.size()
		{
			//------------------------- 1 case init -------------------------
			string casename = caselist_[i];

			//groundtruth file. e.g., 1-1.txt
			ifstream fin_groundtruth(dataset_dir_ + casename + ".txt");
			{
				if (!fin_groundtruth.is_open())
					throw("cannot open file" + casename);
			}


			//------------------------- 2 Haar init -------------------------
			PupilDetectorHaar haar;
			haar.ratio_outer_ = ratio_outer;//1.42, 2, 3, 4, 5, 6, 7
			haar.kf_ = kf; //1,1.1,1.2,1.3,1.4,...
			haar.useSquareHaar_ = useSquareHaar;
			haar.useInitRect_ = useInitRect;
			haar.whstep_ = whstep;//2,3,4,...
			haar.xystep_ = xystep;//2,3,4,...

			if (haar.useInitRect_)
				haar.init_rect_ = init_rectlist_[i];

			cout << "casename = " << casename << endl
				<< "useInitRect = " << haar.useInitRect_ << "	"
				<< "useSquareHaar = " << haar.useSquareHaar_ << "\n"
				<< "r = " << haar.ratio_outer_ << "	" << "kf = " << haar.kf_ << "	"
				<< "wh_step = " << haar.whstep_ << "	" << "xy_step = " << haar.xystep_ << "\n";

			//3 results dir
			//string errorfile_name = "r" + to_string((haar.ratio_outer_)) + " "
				//+ dataset_name + " " + casename + ".txt";
			string errorfile_name = dataset_name + " " + casename + ".txt";
			ofstream fout(result_path + errorfile_name);
			{
				if (!fout.is_open())
					throw("cannot open file" + errorfile_name);
			}

			string runtime_record = "runtime " + dataset_name + " " + casename + ".txt";
			ofstream f_runtime(result_path + runtime_record);


#ifdef VIDEO_EXPORT
			cv::VideoWriter coarse_vid, fine_vid;
			coarse_vid.open(to_string(i) + "coarse.avi", CV_FOURCC('M', 'J', 'P', 'G'), 95.0, cv::Size(640, 480), true);
			fine_vid.open(to_string(i) + "fine.avi", CV_FOURCC('M', 'J', 'P', 'G'), 95.0, cv::Size(640, 480), true);
#endif

			VideoCapture cap(dataset_dir_ + casename + ".avi", cv::CAP_OPENCV_MJPEG);
			double ground_x, ground_y; //groundtruth
			while (fin_groundtruth >> ground_x)
			{
				fin_groundtruth >> ground_y;

				auto t0 = high_resolution_clock::now();

				Mat frame;
				//cap.grab();
				//cap.retrieve(frame);
				cap >> frame;
				auto t1 = high_resolution_clock::now();

				checkImg(frame);
				Mat img_gray;
				img2Gray(frame, img_gray);
				haar.detect(img_gray);
				auto t2 = high_resolution_clock::now();

				cv::RotatedRect ellipse_rect;
				Point2f center_fitting;
				bool flag = haar.extractEllipse(img_gray, haar.pupil_rect_fine_,
					ellipse_rect, center_fitting);
				auto t3 = high_resolution_clock::now();


				//show
				Mat img_coarse, img_fine;
				cv::cvtColor(img_gray, img_coarse, CV_GRAY2BGR);
				//img_coarse = frame;

				int thickness = 2;
				haar.drawCoarse(img_coarse);
				img_fine = img_coarse.clone();
				rectangle(img_fine, haar.pupil_rect_fine_, BLUE, thickness, 8);

				ellipse(img_fine, ellipse_rect, GREEN, thickness);
				drawMarker(img_fine, center_fitting, GREEN, cv::MARKER_CROSS, 20, thickness);

				imshow("Results", img_fine);
				auto t4 = high_resolution_clock::now();
				waitKey(1);


				//calculate & save results
				bool flag_sucess_inner = haar.pupil_rect_coarse_.contains(Point2f(ground_x, ground_y));
				bool flag_sucess_outer = haar.outer_rect_coarse_.contains(Point2f(ground_x, ground_y));
				//check whether rectlist with different w contains the ground truth.
				bool flag_sucess_candidates = false;
				vector<Rect> rectlist = haar.inner_rectlist_;
				for (int i = 0; i < rectlist.size(); i++)
				{
					if (rectlist[i].contains(Point2f(ground_x, ground_y)))
					{
						flag_sucess_candidates = true;
						break;
					}
				}
				double error_coarse = norm(haar.center_coarse_ - Point2f(ground_x, ground_y));

				bool flag_sucess_fine = haar.pupil_rect_fine_.contains(Point2f(ground_x, ground_y));
				double error_fine = norm(haar.center_fine_ - Point2f(ground_x, ground_y));

				double error_fitting = norm(center_fitting - Point2f(ground_x, ground_y));
				{
					fout << flag_sucess_inner << "	" << flag_sucess_outer << "	"
						<< rectlist.size() << "	" << flag_sucess_candidates << "	"
						<< error_coarse << "	" << haar.max_response_coarse_ << "	"
						<< haar.mu_inner_ << "	" << haar.mu_outer_ << "	"
						//below: fine detection
						<< flag_sucess_fine << "	" << error_fine << "	"
						//below: ellipse fitting
						<< error_fitting << endl;
				}

				auto t5 = high_resolution_clock::now();

				duration<double, milli> elapsed_ms1 = t1 - t0;
				duration<double, milli> elapsed_ms2 = t2 - t1;
				duration<double, milli> elapsed_ms3 = t3 - t2;
				duration<double, milli> elapsed_ms4 = t4 - t3;
				duration<double, milli> elapsed_ms5 = t5 - t4;
				f_runtime << elapsed_ms1.count() << "	" << elapsed_ms2.count() << "	"
					<< elapsed_ms3.count() << "	" << elapsed_ms4.count() << "	" << elapsed_ms5.count() << endl;



#ifdef RESULT_EXPORT
				int saveFrameNum = 2; //save a specific frame results for better analysis
				if (haar.frameNum_ == saveFrameNum)
				{
					cv::imwrite(to_string(i) + "Coarse.png", img_coarse);
					cv::imwrite(to_string(i) + "Fine.png", img_fine);
					cv::imwrite(to_string(i) + "CoarseRaw.png", haar.img_coarse_);
					cv::imwrite(to_string(i) + "RectSuppression.png", haar.img_rect_suppression_);
					break;
				}
#endif
#ifdef VIDEO_EXPORT
				coarse_vid << img_coarse;
				fine_vid << img_fine;
#endif

			}//end while
			fin_groundtruth.close();
			fout.close();
			f_runtime.close();

			logfile << haar.mu_inner_ << "	" << haar.mu_outer_ << "	" << haar.kf_ << endl;
#ifdef VIDEO_EXPORT
			coarse_vid.release();
			fine_vid.release();
#endif
		}//end for
		logfile.close();
	}


	//The ground truth is the pupil ellipse (x,y,l,s,theta)
	void SwirskiTest_Haar()
	{
		cout << "SwirskiTest_Haar\n"; //-----------------------
		string method_name = "Haar";
		string dataset_name = "Swirski datasets";
		dataset_init(dataset_name);

		for (int i = 0; i != caselist_.size(); i++)//caselist.size()
		//int i = 1; //test one case
		{
			//------------------------- 1 case init -------------------------
			string casename = caselist_[i];
			cout << "casename = " << casename << endl;

			//groundtruth file. e.g., 1-1.txt
			ifstream fin_groundtruth(dataset_dir_ + casename + ".txt");
			{
				if (!fin_groundtruth.is_open())
					throw("cannot open file" + casename);
			}


			//------------------------- 2 Haar init -------------------------
			PupilDetectorHaar haar;
			haar.kf_ = 1.4; //1,1.1,1.2,1.3
			haar.ratio_outer_ = 1.42;//1.42, 2, 3, 4, 5, 6, 7
			haar.useSquareHaar_ = false;
			haar.useInitRect_ = true;
			haar.xystep_ = 4;//2,3,4
			haar.whstep_ = 4;

			if (haar.useInitRect_)
				haar.init_rect_ = init_rectlist_[i];


			cout << "useInitRect = " << haar.useInitRect_ << "	"
				<< "useSquareHaar = " << haar.useSquareHaar_ << "\n"
				<< "r = " << haar.ratio_outer_ << "	" << "kf = " << haar.kf_ << "	"
				<< "wh_step = " << haar.whstep_ << "	" << "xy_step = " << haar.xystep_ << "\n";


			//3 results dir
			string errorfile_name = "r" + to_string((haar.ratio_outer_)) + " "
				+ dataset_name + " " + casename + ".txt";
			ofstream fout(results_dir_ + method_name + "/" + errorfile_name);
			{
				if (!fout.is_open())
					throw("cannot open file" + errorfile_name);
			}


			int img_index;
			char a2; //For storing vertical lines in ground-truth document
			double ground_x, ground_y, long_axis, short_axis, angle;
			while (fin_groundtruth >> img_index) //-----------------------
			{
				fin_groundtruth >> a2 >> ground_x >> ground_y >>
					long_axis >> short_axis >> angle;
				Mat frame = imread(dataset_dir_ + casename + "/" + to_string(img_index) + "-eye.png");
				checkImg(frame);

				Mat img_gray;
				img2Gray(frame, img_gray);
				//filterImg(img_gray, img_gray);

				haar.detect(img_gray);

				cv::RotatedRect ellipse_rect;
				Point2f center_fitting;
				bool flag = haar.extractEllipse(img_gray, haar.pupil_rect_fine_,
					ellipse_rect, center_fitting);


				//show
				Mat img_coarse, img_fine;
				cv::cvtColor(img_gray, img_coarse, CV_GRAY2BGR);

				int thickness = 2;
				haar.drawCoarse(img_coarse);
				img_fine = img_coarse.clone();
				rectangle(img_fine, haar.pupil_rect_fine_, BLUE, thickness, 8);

				ellipse(img_fine, ellipse_rect, RED, thickness);
				drawMarker(img_fine, center_fitting, RED, cv::MARKER_CROSS, 20, thickness);

				imshow("Results", img_fine);
				waitKey(5);


				//calculate & save results
				bool flag_sucess_inner = haar.pupil_rect_coarse_.contains(Point2f(ground_x, ground_y));
				bool flag_sucess_outer = haar.outer_rect_coarse_.contains(Point2f(ground_x, ground_y));
				//check whether rectlist with different w contains the ground truth.
				bool flag_sucess_candidates = false;
				vector<Rect> rectlist = haar.inner_rectlist_;
				for (int i = 0; i < rectlist.size(); i++)
				{
					if (rectlist[i].contains(Point2f(ground_x, ground_y)))
					{
						flag_sucess_candidates = true;
						break;
					}
				}
				double error_coarse = norm(haar.center_coarse_ - Point2f(ground_x, ground_y));

				bool flag_sucess_fine = haar.pupil_rect_fine_.contains(Point2f(ground_x, ground_y));
				double error_fine = norm(haar.center_fine_ - Point2f(ground_x, ground_y));

				double error_fitting = norm(center_fitting - Point2f(ground_x, ground_y));

				RotatedRect el = RotatedRect(Point(ground_x, ground_y), Size(long_axis * 2, short_axis * 2), angle * 180 / PI); // ------------------
				Rect2f rect = el.boundingRect2f();
				float ratio_width = haar.pupil_rect_coarse_.width*1.0f / rect.width;
				float ratio_width2 = haar.pupil_rect_fine_.width*1.0f / rect.width;
				{
					fout << flag_sucess_inner << "	" << flag_sucess_outer << "	"
						<< rectlist.size() << "	" << flag_sucess_candidates << "	"
						<< error_coarse << "	" << haar.max_response_coarse_ << "	"
						<< haar.mu_inner_ << "	" << haar.mu_outer_ << "	"
						//below: fine detection
						<< flag_sucess_fine << "	" << error_fine << "	"
						//below: ellipse fitting
						<< error_fitting << "	"
						<< ratio_width << "	" << ratio_width2 << endl;
				}
			}//end while
			fin_groundtruth.close();
			fout.close();
		}//end for
	}

	void samplesTest_Haar()
	{
		string dirname = "samples/";
		string filename = "imagelist.txt";
		vector<string> imagelist;
		readStringList_txt(dirname + filename, imagelist);

		for (int r = 2; r <= 7; r++)
		{
			//for (int i = 0; i < imagelist.size(); ++i)//imagelist.size()
			int i = 14;
			{
				cout << imagelist[i] << endl;
				Mat img = imread(dirname + imagelist[i]);


				PupilDetectorHaar haar;
				haar.ratio_outer_ = r;//1.4, 2, 3, 4, 5, 6, 7
				haar.kf_ = 1; //1,1.1,1.2,1.3,1.4,...
				haar.useSquareHaar_ = 1;
				haar.useInitRect_ = 0;
				haar.xystep_ = 1;//2,3,4,...
				haar.whstep_ = 1;//2,3,4,...

				Mat img_gray;
				img2Gray(img, img_gray);
				haar.detect(img_gray);

				cv::RotatedRect ellipse_rect;
				Point2f center_fitting;
				bool flag = haar.extractEllipse(img_gray, haar.pupil_rect_fine_,
					ellipse_rect, center_fitting);

				//show
				Mat img_coarse, img_fine;
				cv::cvtColor(img_gray, img_coarse, CV_GRAY2BGR);

				int thickness = 2;
				haar.drawCoarse(img_coarse);
				img_fine = img_coarse.clone();
				rectangle(img_fine, haar.pupil_rect_fine_, BLUE, thickness, 8);

				ellipse(img_fine, ellipse_rect, GREEN, thickness);
				drawMarker(img_fine, center_fitting, GREEN, cv::MARKER_CROSS, 20, thickness);

				imshow("Results", img_fine);
				waitKey(500);

				string result_dir = dirname + "results/";
				imwrite(result_dir + imagelist[i] + " r" + to_string(int(haar.ratio_outer_)) + " Coarse.png", img_coarse);
				imwrite(result_dir + imagelist[i] + " r" + to_string(int(haar.ratio_outer_)) + " Fine.png", img_fine);
			}
		}
	}


	//Pupilnet dataset
	void PupilnetDataset_init()
	{

		string dataset_name = "pupilnet datasets";

		//caselist保存每个case的文件前缀名
		//readStringList2(dataset_dir + "caselist.txt", caselist);
		caselist_ = { "data set new I","data set new II","data set new III",
			"data set new IV","data set new V" };



		//init rect的路径与导入
		ifstream fin(dataset_dir_ + "init_rect_pupil.txt");
		double x, y, width, height;
		while (fin >> x)
		{
			fin >> y >> width >> height;
			init_rectlist_.push_back(Rect(x, y, width, height));
		}
	}

	void PupilnetDatasetTest_Haar()
	{
		PupilnetDataset_init();
		string dataset_name;

		//method init
		string method_name = "Haar";

		HaarParams params;
		params.initRectFlag = true;
		params.squareHaarFlag = true;
		//kernel策略1：ratio多选1
		params.outer_ratio = 1.42; //1.42, 2, 3, 4, 5, 6, 7
		//kernel策略2：mu_inner权重
		params.kf = 1.4; //1,1.1,1.2,1.3
		params.wh_step = 1;//2,3,4
		params.xy_step = 4;
		cout << "PupilnetDatasetTest_Haar" << "\n" << "initRectFlag=" << params.initRectFlag << "	"
			<< "squareHaarFlag=" << params.squareHaarFlag << "\n"
			<< "r=" << params.outer_ratio << "	" << "kf=" << params.kf << "	"
			<< "wh_step=" << params.wh_step << "	" << "xy_step=" << params.xy_step << "\n";

		for (int i = 0; i != caselist_.size(); i++)//caselist.size()
		//int i = 54; //单个case测试
		{
			if (params.initRectFlag)
			{
				params.init_rect = init_rectlist_[i];

				//经测试，初始scale 2,3,4都可得到初始rect的响应
				params.roi = rectScale(init_rectlist_[i], 2);//仅第一帧的ROI
				//可以考虑下面的方法
				//params.roi = (x, y +h/2-w/2, w, w)//init_rectlist[i];
			}


			string casename = caselist_[i];
			cout << casename << endl;

			//groundtruth file. e.g., 1-1.txt
			ifstream fin_groundtruth(dataset_dir_ + casename + ".txt");
			{
				if (!fin_groundtruth.is_open())
					throw("cannot open file" + casename);
			}


			//保存路径
			string errorfile_name = "r" + to_string((params.outer_ratio)) + " "
				+ dataset_name + " " + casename + ".txt";
			ofstream fout(results_dir_ + method_name + "/" + errorfile_name);
			{
				if (!fout.is_open())
					throw("cannot open file" + errorfile_name);
			}


			int a2; //用于存储文档中的第一个数0
			int img_index;
			double x, y;
			bool firstFrameFlag = true;
			while (fin_groundtruth >> a2)
			{
				fin_groundtruth >> img_index >> x >> y;

				string num_s = to_string(img_index);
				string num_0;
				for (int i0 = 0; i0 < 10 - num_s.length(); i0++)
					num_0.append("0");
				num_s = num_0 + num_s;


				Mat frame = imread(dataset_dir_ + casename + "/" + num_s + ".png");
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


				params.roi = Rect(0, 0, img_gray.cols, img_gray.rows);

				Mat img_haar;
				cvtColor(img_gray, img_haar, CV_GRAY2BGR);
				haar.drawCoarse(img_haar);
				rectangle(img_haar, haar.pupil_rect_fine_, BLUE, 1, 8);

				Mat img_pupil = img_gray(haar.pupil_rect_fine_);


				//save results
				{
					//groudthruth data要变换一次
					x = x / 2;
					y = img_gray.rows - y / 2;

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




					Point2f center3;
					//利用PuRe进一步提取
					{
						Rect boundary(0, 0, img_gray.cols, img_gray.rows);
						Rect roiRect = rectScale(haar.pupil_rect_fine_, 1.42)&boundary;
						Mat img_t = img_gray(roiRect);
						int tau;
						//if (haar.mu_outer_ - haar.mu_inner_ > 30)
						//	tau = params.mu_outer;
						//else
						tau = haar.mu_inner_ + 30;
						PupilDetectorHaar::filterLight(img_t, img_t, tau);

						//if (haar.mu_outer_ - haar.mu_inner_ < 30)
						//	center3 = center2;
						//else
						{
							PuRe detector;
							Pupil pupil = detector.run(img_t);
							pupil.center = pupil.center + Point2f(roiRect.tl());
							if (haar.pupil_rect_fine_.contains(pupil.center))
							{
								drawMarker(img_haar, pupil.center, Scalar(0, 0, 255));
								if (pupil.size.width > 0)
									ellipse(img_haar, pupil, Scalar(0, 0, 255));

								center3 = pupil.center;
							}
							else
								center3 = center2;
						}
					}
					double error3 = norm(center3 - Point2f(x, y));



					fout << flag_sucess_inner << "	" << flag_sucess_outer << "	"
						<< rectlist.size() << "	" << flag_sucess_candidates << "	"
						<< error << "	" << haar.max_response_coarse_ << "	"
						<< haar.mu_inner_ << "	" << haar.mu_outer_ << "	"
						//以下是detectToFine的结果
						<< flag_sucess_inner2 << "	" << error2 << "	"
						//以下为ellipse fitting结果
						<< error3 << "	" << endl;
				}
				//imshow("pupil region", img_pupil);
				imshow("Results", img_haar);
				waitKey(5);
			}//end while
			fin_groundtruth.close();
			fout.close();
		}//end for

	}



	//------------------------------- PuRe -------------------------------
	void LPWTest_PuRe()
	{
		//method init
		string method_name = "PuRe";
		string dataset_name;

		//for (int i = 33; i != caselist.size(); i++)//caselist.size()
		int i = 1; //单个case测试
		{
			string casename = caselist_[i];
			cout << casename << endl;

			//groundtruth file. e.g., 1-1.txt
			ifstream fin_groundtruth(dataset_dir_ + casename + ".txt");
			{
				if (!fin_groundtruth.is_open())
					throw("cannot open file" + casename);
			}


			//保存路径
			string errorfile_name = dataset_name + " " + casename + ".txt";
			ofstream fout(results_dir_ + method_name + "/" + errorfile_name);
			{
				if (!fout.is_open())
					throw("cannot open file" + errorfile_name);
			}


			double x, y;
			VideoCapture cap(dataset_dir_ + casename + ".avi");

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

				PuRe detector;
				Pupil pupil = detector.run(img_gray);
				drawMarker(frame, pupil.center, Scalar(0, 0, 255));
				if (pupil.size.width > 0)
					ellipse(frame, pupil, Scalar(0, 0, 255));

				double error = norm(pupil.center - Point2f(x, y));

				//save results
				fout << error << endl;

				imshow("Results", frame);
				waitKey(5);
			}//end while
			fin_groundtruth.close();
			fout.close();
		}//end for
	}


	void SwirskiTest_PuRe()
	{
		string datasets = "C:/KernelData/0 code lib/pupil datasets/";
		string swiriski = "Swiriski datasets";
		vector<string> casenames = { "p1-left","p1-right","p2-left","p2-right" };


		for (int i = 0; i < casenames.size(); i++)
		{
			string casename = casenames[i];
			string dirname = datasets + swiriski + "/" + casename + "/";
			string filename = "pupil-ellipses.txt";
			ifstream fin(dirname + filename);
			if (!fin.is_open())
				throw("cannot open file" + casename);

			string errorname = swiriski + " " + casename + ".txt";
			ofstream fout(datasets + "error/PuRe/" + errorname);
			if (!fout.is_open())
				throw("cannot open file" + errorname);

			int img_index;
			char a2;
			double x, y, long_axis, short_axis, angle;
			while (fin >> img_index)
			{
				cout << endl << img_index << endl;
				fin >> a2 >> x >> y >>
					long_axis >> short_axis >> angle;
				Mat frame = imread(dirname + "frames/" + to_string(img_index) + "-eye.png");
				if (frame.empty())
					throw("image import error!");

				PuRe detector;
				Mat frame2;
				cvtColor(frame, frame2, CV_BGR2GRAY);
				Pupil pupil = detector.run(frame2);
				drawMarker(frame, pupil.center, Scalar(0, 0, 255));
				ellipse(frame, pupil, Scalar(0, 0, 255));
				//PupilExtractionMethod detector;
				//measureTime([&]() {detector.detect(frame); }, "detector\t");
				double error = norm(pupil.center - Point2f(x, y));
				fout << error << endl;
				namedWindow(casename);
				moveWindow(casename, 0, 0);
				imshow(casename, frame);
				waitKey(30);
			}
			fin.close();
			fout.close();
		}
	}




	string dataset_dir_;		//dir of current datasets
	string results_dir_;

	vector<string> caselist_;
	vector<Rect> init_rectlist_;
private:

};





#endif