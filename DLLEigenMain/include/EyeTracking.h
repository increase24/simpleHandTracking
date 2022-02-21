#pragma once
#ifndef EYE_TRACKING_H
#define EYE_TRACKING_H

#include "EyeTrackingExport.h"
// #include "EyeTrackerUtils.h"
// OpenCV dependencies
#include <opencv2/core/core.hpp>
#include <queue>
// #include <opencv2/objdetect.hpp>
//
//#ifdef _DLL_EXPORTS
//#define DLL_API _declspec(dllexport)
//#else
//#define DLL_API _declspec(dllimport)
//#endif

namespace LandmarkDetector
{
	class EyeTrackerExport {

	public:

		//// A default constructor
		//EyeTracker();

		//// Constructor from a model file
		//EyeTracker(std::string fname);

		//// Empty Destructor	as the memory of every object will be managed by the corresponding libraries (no pointers)
		//~EyeTracker() {}

		// Reading the model in
		virtual bool Read(std::string name) = 0;

		virtual bool Inference(const cv::Mat &image, cv::Mat_<float> &lastest_detected_landmarks) = 0;

		virtual bool GetLoadStatus() = 0;

		// virtual void GetLandmarks(std::deque<cv::Mat_<float>> &input_landmarks) = 0;

		// virtual std::deque<cv::Mat_<float>> GetLandmarks() = 0;

		virtual float GetDetectionCertainty() = 0;

		virtual cv::Mat_<float> GetFaceLandmark() = 0;
		virtual cv::Mat_<float> GetLeftEyeLandmark() = 0;
		virtual cv::Mat_<float> GetRightEyeLandmark() = 0;



	};
}
#endif // EYE_TRACKING_H
