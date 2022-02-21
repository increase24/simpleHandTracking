
#ifndef CEN_PATCH_EXPERT_H
#define CEN_PATCH_EXPERT_H

// system includes
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>

namespace LandmarkDetector
{
	//===========================================================================
	/**
	The classes describing the CEN patch experts
	*/

	class CEN_patch_expert {
	public:

		// Width and height of the patch expert support area
		int width_support;
		int height_support;

		// Neural weights
		std::vector<cv::Mat_<float>> biases;

		// Neural weights
		std::vector<cv::Mat_<float>> weights;

		std::vector<int> activation_function;
		
		// Confidence of the current patch expert (used for NU_RLMS optimisation)
		double  confidence;

		CEN_patch_expert() { ; }

		// A copy constructor
		CEN_patch_expert(const CEN_patch_expert& other);

		// Reading in the patch expert
		void Read(std::ifstream &stream);

		// The actual response computation from intensity image
		void Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response);

		void ResponseInternal(cv::Mat_<float>& response);

		// For frontal faces can apply mirrored and non-mirrored experts at the same time
		void ResponseSparse(const cv::Mat_<float> &area_of_interest_left, const cv::Mat_<float> &area_of_interest_right, cv::Mat_<float> &response_left, cv::Mat_<float> &response_right, cv::Mat_<float>& mapMatrix, cv::Mat_<float>& im2col_prealloc_left, cv::Mat_<float>& im2col_prealloc_right);

	};

	void interpolationMatrix(cv::Mat_<float>& mapMatrix, int response_height, int response_width, int input_width, int input_height);

}
#endif // CEN_PATCH_EXPERT_H
