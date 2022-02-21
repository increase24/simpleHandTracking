
#ifndef PATCH_EXPERTS_H
#define PATCH_EXPERTS_H

// OpenCV includes
#include <opencv2/core/core.hpp>


#include "SVR_patch_expert.h"
#include "CCNF_patch_expert.h"
#include "CEN_patch_expert.h"
#include "PDM.h"

namespace LandmarkDetector
{
//===========================================================================
/** 
    Combined class for all of the patch experts
*/
class Patch_experts
{

public:

	// The collection of SVR patch experts (for intensity/grayscale images), the experts are laid out scale->view->landmark
	std::vector<std::vector<std::vector<Multi_SVR_patch_expert> > >	svr_expert_intensity;
	 
	// The collection of LNF (CCNF) patch experts (for intensity images), the experts are laid out scale->view->landmark
	std::vector<std::vector<std::vector<CCNF_patch_expert> > >			ccnf_expert_intensity;

	// The node connectivity for CCNF experts, at different window sizes and corresponding to separate edge features
	std::vector<std::vector<cv::Mat_<float> > >					sigma_components;

	// The collection of CEN patch experts (for intensity images), the experts are laid out scale->view->landmark
	std::vector<std::vector<std::vector<CEN_patch_expert> > >			cen_expert_intensity;

	//Useful to pre-allocate data for im2col so that it is not allocated for every iteration and every patch
	std::vector< std::map<int, cv::Mat_<float> > > preallocated_im2col;

	// The available scales for intensity patch experts
	std::vector<double>							patch_scaling;

	// The available views for the patch experts at every scale (in radians)
	std::vector<std::vector<cv::Vec3d> >               centers;

	// Landmark visibilities for each scale and view
	std::vector<std::vector<cv::Mat_<int> > >          visibilities;

	cv::Mat_<int>							mirror_inds;
	cv::Mat_<int>							mirror_views;

	// Early termination calibration values, useful for CE-CLM model to speed up the multi-hypothesis setup
	std::vector<double> early_term_weights;
	std::vector<double> early_term_biases;
	std::vector<double> early_term_cutoffs;


	// A default constructor
	Patch_experts(){;}

	// A copy constructor
	Patch_experts(const Patch_experts& other);

	// Returns the patch expert responses given a grayscale image.
	// Additionally returns the transform from the image coordinates to the response coordinates (and vice versa).
	// The computation also requires the current landmark locations to compute response around, the PDM corresponding to the desired model, and the parameters describing its instance
	// Also need to provide the size of the area of interest and the desired scale of analysis
	void Response(std::vector<cv::Mat_<float> >& patch_expert_responses, cv::Matx22f& sim_ref_to_img, cv::Matx22f& sim_img_to_ref, const cv::Mat_<float>& grayscale_image,
							 const PDM& pdm, const cv::Vec6f& params_global, const cv::Mat_<float>& params_local, int window_size, int scale);

	// Getting the best view associated with the current orientation
	int GetViewIdx(const cv::Vec6f& params_global, int scale) const;

	// The number of views at a particular scale
	inline int nViews(size_t scale = 0) const { return (int)centers[scale].size(); };

	// Reading in all of the patch experts
	bool Read(std::vector<std::string> intensity_svr_expert_locations, std::vector<std::string> intensity_ccnf_expert_locations,
		std::vector<std::string> intensity_cen_expert_locations, std::string early_term_loc = "");
   

private:
	bool Read_SVR_patch_experts(std::string expert_location, std::vector<cv::Vec3d>& centers, std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<Multi_SVR_patch_expert> >& patches, double& scale);
	bool Read_CCNF_patch_experts(std::string patchesFileLocation, std::vector<cv::Vec3d>& centers, std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<CCNF_patch_expert> >& patches, double& patchScaling);
	bool Read_CEN_patch_experts(std::string expert_location, std::vector<cv::Vec3d>& centers, std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<CEN_patch_expert> >& patches, double& scale);

	// Helper for collecting visibilities
	std::vector<int> Collect_visible_landmarks(std::vector<std::vector<cv::Mat_<int> > > visibilities, int scale, int view_id, int n);
};
 
}
#endif // PATCH_EXPERTS_H
