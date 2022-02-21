// handGesRec.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <windows.h>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "IsoGesRec/handGesRec.h"
#include <chrono>

using namespace std;

int main()
{
	/* -----------  init model inference session  ----------- */
	const char* pamlDetModelPath = "./models/palm_detection.dat";
	const char* handGesRecModelPath = "./models/decoder_tcnn_05-31-01-13.dat";
	const char* handStaModelPath =  "./models/staNet_07-15.dat"; // "./models/encoder_sta_v104.dat";
	const char* handFeatModel = "./models/encoder_tcnn_05-31-01-13.dat";
	const char* anchorFilePath = "./models/anchors.bin";
	float thresholdDy = 0.6;   // dynamic gesture threshold
	float thresholdSta = 0.8;  // static gesture threshold
	int output_gesture = 0;
	int fps = 30; // set inference fps
	HyperParam hyper_param = { 0, 5, 4, 2, 5, 0.05 , 0.9};
	if (hyper_param.vote_num < hyper_param.hit_num) 
	{
		cout << "[error]:hyper_param.hit_num is not allowed to surpass hyper_param.vote_num!" << endl;
		return 0;
	};
	void* p_session = handLandmarks_Init(pamlDetModelPath, handStaModelPath, handGesRecModelPath, handFeatModel, anchorFilePath, hyper_param);
	cout << "load model done." << endl;

	cv::Mat rawFrame, showFrame, image_gestures_copy;
	/* -----------  opencv video capture  ----------- */
	cv::VideoCapture cap;
	cap.open(0);
	//cap.open("E:/demos/gest_test_demo.mp4");
	int rawHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	int rawWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int video_shape[2] = { rawHeight, rawWidth };
	/* -----------  gesture images  ----------- */
	//string gesture_classes[] = { "doing_other_things", "swiping_down", "swiping_left", "swiping_right", "swiping_up", "thumb_down", "thumb_up", "zoom_in", "zoom_out", "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart", "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart",  "thumb_up",  "thumb_down" };
	//remove overlap
	string gesture_classes[] = { "doing_other_things", "swiping_down", "swiping_left", "swiping_right", "swiping_up", 
		"zoom_in", "zoom_out", "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart",  
		"thumb_up",  "thumb_down"}; 
	int predict_gesture;
	cout << "start gesture recognition!" << endl;
	int last_gest = 0;
	while (cap.read(rawFrame))
	{
		auto stop1 = chrono::high_resolution_clock::now();
		rawFrame.copyTo(showFrame);
		//cv::flip(showFrame, showFrame, +1);
		float output_handRegion[7] = { 0, 0, 0, 0, 0, 0, 0 };
		predict_gesture = handLandmarks_inference(p_session, rawFrame.data, video_shape, output_handRegion, false);

		// fps setup
		auto stop2 = chrono::high_resolution_clock::now();
		auto infer_time1 = chrono::duration_cast<chrono::milliseconds>(stop2 - stop1).count();
		//cout << "infer time cost:" << infer_time1 << endl;
		if ((1000 / fps - infer_time1) > 0) { } //  Sleep(1000 / fps - infer_time1);
		// get predict gesture and predict probability
		float outputProb = output_handRegion[6];
		output_gesture = predict_gesture;
		int new_height = rawFrame.rows; 
		int new_width = rawFrame.cols;
		cv::resize(rawFrame, rawFrame, cv::Size(new_width, new_height));
		cv::resize(showFrame, showFrame, cv::Size(new_width, new_height));

		// visualize the bounding box
		if (outputProb > 0.0)
		{
			if (rawFrame.cols > rawFrame.rows)
			{
				cv::rectangle(showFrame, cv::Rect(int(output_handRegion[0] * rawFrame.cols), int(output_handRegion[1] * rawFrame.cols - (rawFrame.cols - rawFrame.rows) / 2), int((output_handRegion[2] - output_handRegion[0])*rawFrame.cols),
					int((output_handRegion[3] - output_handRegion[1])* rawFrame.cols)), cv::Scalar(0, 0, 255), 1, 1, 0);
				/*if (output_gesture > 0)
					cv::putText(showFrame, gesture_classes[last_gest], cv::Point2d(int((rawFrame.cols - rawFrame.rows) / 2 + output_handRegion[0] * rawFrame.rows), int(output_handRegion[1] * rawFrame.rows)), 0, 1.0f, cv::Scalar(0, 0, 255));*/
			}
			else
			{
				cv::rectangle(showFrame, cv::Rect(int(output_handRegion[0] * rawFrame.rows - (rawFrame.rows - rawFrame.cols) / 2), int(output_handRegion[1] * rawFrame.rows), int((output_handRegion[2] - output_handRegion[0]) * rawFrame.rows),
					int((output_handRegion[3] - output_handRegion[1]) * rawFrame.rows)), cv::Scalar(0, 0, 255), 1, 1, 0);
				/*if (output_gesture > 0)
					cv::putText(showFrame, gesture_classes[last_gest], cv::Point2d(int(output_handRegion[0] * rawFrame.cols), int((rawFrame.rows - rawFrame.cols) / 2 + output_handRegion[1] * rawFrame.cols)), 0, 1.0f, cv::Scalar(0, 0, 255));*/

			}
		}

		std::string text;
		if (output_gesture > 0)
		{
			text = "Detect gesture: " + gesture_classes[output_gesture];
		}
		else
		{
			text = "Detect gesture: ";
		}

		int baseline, thickness = 2, font_face = cv::FONT_HERSHEY_COMPLEX;
		double font_scale = 1;
		cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
		cv::Point origin;
		origin.x = showFrame.cols / 100;
		origin.y = showFrame.rows / 100 + text_size.height;
		cv::putText(showFrame, text, origin, font_face, font_scale, cv::Scalar(70, 23, 11), thickness, 8, 0);

		//}
		//cout << "predict gesture prob: " << outputProb << endl;
		//cout << "predict gesture:" << gesture_classes[output_gesture] << endl;
		cv::imshow("handDetection", showFrame);
		if (cv::waitKey(5) == 27) // 'esc'
		{
			cv::destroyAllWindows();
			return -1;
		};
	}
}


