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
	const char* anchorFilePath = "./models/anchors.bin";
	int fps = 30; // set inference fps
	void* p_session = handLandmarks_Init(pamlDetModelPath, anchorFilePath);
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
		auto stop1 = chrono::high_resolution_clock::now();
		rawFrame.copyTo(showFrame);
		//cv::flip(showFrame, showFrame, +1);
		DetHands output;
		int result = handLandmarks_inference(p_session, rawFrame.data, video_shape, output, false);

		// fps setup
		auto stop2 = chrono::high_resolution_clock::now();
		auto infer_time1 = chrono::duration_cast<chrono::milliseconds>(stop2 - stop1).count();
		cout << "推理时间（ms）:" << infer_time1 << endl;
		cout << "右手检测置信度:" << output.r_hand.score << endl;
		float threshold = 0.95;
		if(output.r_hand.score > threshold)
		{
			cv::rectangle(showFrame, cv::Point(output.r_hand.xmin * rawWidth, output.r_hand.ymin* rawHeight),
				cv::Point(output.r_hand.xmax * rawWidth, output.r_hand.ymax* rawHeight), cv::Scalar(0, 255,0),1,1,0);
		}

		cv::imshow("handDetection", showFrame);
		if (cv::waitKey(5) == 27) // 'esc'
		{
			cv::destroyAllWindows();
			return -1;
		};
	}
}


