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
	const char* handKptsModelPath = "./models/hand_landmarks_v2.dat";
	const char* anchorFilePath = "./models/anchors.bin";
	int fps = 30; // set inference fps
	void* p_session = handLandmarks_Init(pamlDetModelPath, handKptsModelPath, anchorFilePath);
	cout << "load model done." << endl;

	cv::Mat rawFrame, showFrame, image_gestures_copy;
	/* -----------  opencv video capture  ----------- */
	cv::VideoCapture cap;
	cap.open(0);
	//cap.open("E:/demos/gest_test_demo.mp4");
	int rawHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	int rawWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int video_shape[2] = { rawHeight, rawWidth };
	cout << "start gesture recognition!" << endl;
	int last_gest = 0;
	while (cap.read(rawFrame))
	{
		cv::Mat frame_rgba;
		cv::cvtColor(rawFrame, frame_rgba, cv::COLOR_BGR2BGRA);
		auto stop1 = chrono::high_resolution_clock::now();
		rawFrame.copyTo(showFrame);
		//cv::flip(showFrame, showFrame, +1);
		//DetHands output;
		float* output = new float[10];
		int result = handLandmarks_inference(p_session, frame_rgba.data, video_shape, output, true);

		// fps setup
		auto stop2 = chrono::high_resolution_clock::now();
		auto infer_time1 = chrono::duration_cast<chrono::milliseconds>(stop2 - stop1).count();
		cout << "推理时间（ms）:" << infer_time1 << endl;
		cout << "左手检测置信度:" << output[4] << endl;
		cout << "右手检测置信度:" << output[9] << endl;
		float threshold = 0.95;
		if(output[4] > threshold)
		{
			cv::rectangle(showFrame, cv::Point(output[0] * rawWidth, output[1] * rawHeight),
				cv::Point(output[2] * rawWidth, output[3] * rawHeight), cv::Scalar(0, 255,0),1,1,0);
		}
		if (output[9] > threshold)
		{
			cv::rectangle(showFrame, cv::Point(output[5] * rawWidth, output[6] * rawHeight),
				cv::Point(output[7] * rawWidth, output[8] * rawHeight), cv::Scalar(0, 255, 0), 1, 1, 0);
		}

		cv::imshow("handDetection", showFrame);
		if (cv::waitKey(5) == 27) // 'esc'
		{
			cv::destroyAllWindows();
			return -1;
		};
		delete output;
	}
}


