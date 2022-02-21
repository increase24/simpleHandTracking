
#ifndef EYE_TRACKER_UTILS_H
#define EYE_TRACKER_UTILS_H

// OpenCV includes
#include <opencv2/core/core.hpp>

#include "EyeTrackingExport.h"

namespace LandmarkDetector
{
	//===========================================================================	
	// Defining a set of useful utility functions to be used within CLNF

	//===========================================================================
	// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
	//===========================================================================
	// This is a modified version of openCV code that allows for precomputed dfts of templates and for precomputed dfts of an image
	// _img is the input img, _img_dft it's dft (optional), _integral_img the images integral image (optional), squared integral image (optional), 
	// templ is the template we are convolving with, templ_dfts it's dfts at varying windows sizes (optional),  _result - the output, method the type of convolution
	void matchTemplate_m(const cv::Mat_<float>& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, 
		const cv::Mat_<float>&  templ, std::map<int, cv::Mat_<double> >& templ_dfts, cv::Mat_<float>& result, int method);

	// Useful utility for grabing a bounding box around a set of 2D landmarks (as a 1D 2n x 1 vector of xs followed by doubles or as an n x 2 vector)
	void ExtractBoundingBox(const cv::Mat_<float>& landmarks, float &min_x, float &max_x, float &min_y, float &max_y);
	std::vector<cv::Point2f> CalculateAllParts(const cv::Mat_<float>& shape2D);
	std::vector<cv::Point2f> CalculateAllParts(const CLNF_EYE_TRACK& clnf_model);
	std::vector<cv::Point2f> CalculateAllEyeParts(const CLNF_EYE_TRACK& clnf_model);
	std::vector<cv::Point3f> Calculate3DEyeParts(const CLNF_EYE_TRACK& clnf_model, float fx, float fy, float cx, float cy);

	std::vector<cv::Point2f> CalculateVisibleLandmarks(const cv::Mat_<float>& shape2D, const cv::Mat_<int>& visibilities);

	//============================================================================
	// Matrix reading functionality
	//============================================================================

	// Reading a matrix written in a binary format
	void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat);

	// Reading in a matrix from a stream
	void ReadMat(std::ifstream& stream, cv::Mat& output_matrix);

	// Skipping comments (lines starting with # symbol)
	void SkipComments(std::ifstream& stream);


}
#endif // EYE_TRACKER_UTILS_H
