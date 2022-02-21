#ifndef EYE_TRACKER_H
#define EYE_TRACKER_H

#include "EyeTrackingExport.h"
// #include "EyeTrackerUtils.h"
// OpenCV dependencies
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>


namespace LandmarkDetector
{
	class EyeTracker {

	public:
		CLNF_EYE_TRACK eye_model;
		EyeModelParameters model_params;
		std::string model_location;

		// See if the model was read in correctly
		bool loaded_successfully;

		// A default constructor
		EyeTracker();

		// Constructor from a model file
		EyeTracker(std::string fname);

		// Empty Destructor	as the memory of every object will be managed by the corresponding libraries (no pointers)
		~EyeTracker() {}

		// output members
		cv::Mat_<float> detected_landmarks;
		cv::Mat_<float> left_eye_landmarks;
		cv::Mat_<float> right_eye_landmarks;
		float			detection_certainty;

		// Reading the model in
		bool Read(std::string name);

		bool Inference(const cv::Mat &image, cv::Mat_<float> lastest_detected_landmarks);

		bool DecodeEyeLandmarks(const CLNF_EYE_TRACK& clnf_model);

	};
}
#endif // EYE_TRACKER_H
