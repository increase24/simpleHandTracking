#pragma once
#ifndef EYE_TRACKING_EXPORT_H
#define EYE_TRACKING_EXPORT_H

// OpenCV dependencies
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>


#include "PDM.h"
#include "Patch_experts.h"
#include "LandmarkDetectionValidator.h"
#include "EyeTrackingParameters.h"

namespace LandmarkDetector
{

	// A main class containing all the modules required for landmark detection
	// Face shape model
	// Patch experts
	// Optimization techniques
	class CLNF_EYE_TRACK {

	public:

		//===========================================================================
		// Member variables that contain the model description

		// The linear 3D Point Distribution Model
		PDM					pdm;
		// The set of patch experts
		Patch_experts		patch_experts;

		// The local and global parameters describing the current model instance (current landmark detections)

		// Local parameters describing the non-rigid shape
		cv::Mat_<float>    params_local;

		// Global parameters describing the rigid shape [scale, euler_x, euler_y, euler_z, tx, ty]
		cv::Vec6f           params_global;

		// A collection of hierarchical CLNF models that can be used for refinement
		std::vector<CLNF_EYE_TRACK>								hierarchical_models;
		std::vector<std::string>						hierarchical_model_names;
		std::vector<std::vector<std::pair<int, int>>>	hierarchical_mapping;
		std::vector<EyeModelParameters>				hierarchical_params;

		//==================== Helpers for face detection and landmark detection validation =========================================

		// TODO these should be static, and loading should be made easier

		// Validate if the detected landmarks are correct using an SVR regressor
		DetectionValidator	landmark_validator;

		// Indicating if landmark detection succeeded (based on SVR validator)
		bool				detection_success;

		// Indicating if the tracking has been initialised (for video based tracking)
		bool				tracking_initialised;

		// The actual output of the regressor (-1 is perfect detection 1 is worst detection)
		float				detection_certainty;

		// Indicator if eye model is there for eye detection
		bool				eye_model;

		// the triangulation per each view (for drawing purposes only)
		std::vector<cv::Mat_<int> >	triangulations;

		//===========================================================================
		// Member variables that retain the state of the tracking (reflecting the state of the lastly tracked (detected) image

		// Lastly detect 2D model shape [x1,x2,...xn,y1,...yn]
		cv::Mat_<float>			detected_landmarks;

		// The landmark detection likelihoods (combined and per patch expert)
		float					model_likelihood;
		cv::Mat_<float>			landmark_likelihoods;

		// Keeping track of how many frames the tracker has failed in so far when tracking in videos
		// This is useful for knowing when to initialise and reinitialise tracking
		int failures_in_a_row;

		// A template of a face that last succeeded with tracking (useful for large motions in video)
		cv::Mat_<uchar> face_template;

		// Useful when resetting or initialising the model closer to a specific location (when multiple faces are present)
		cv::Point_<double> preference_det;

		// Tracking which view was used last
		int view_used;

		// See if the model was read in correctly
		bool loaded_successfully;

		// A default constructor
		CLNF_EYE_TRACK();

		// Constructor from a model file
		CLNF_EYE_TRACK(std::string fname);

		// Copy constructor (makes a deep copy of the detector)
		CLNF_EYE_TRACK(const CLNF_EYE_TRACK& other);

		// Assignment operator for lvalues (makes a deep copy of the detector)
		CLNF_EYE_TRACK & operator= (const CLNF_EYE_TRACK& other);

		// Empty Destructor	as the memory of every object will be managed by the corresponding libraries (no pointers)
		~CLNF_EYE_TRACK() {}

		// Move constructor
		CLNF_EYE_TRACK(const CLNF_EYE_TRACK&& other);

		// Assignment operator for rvalues
		CLNF_EYE_TRACK & operator= (const CLNF_EYE_TRACK&& other);

		// Does the actual work - landmark detection
		// bool DetectLandmarks(const cv::Mat_<uchar> &image, EyeModelParameters& params, cv::Mat_<float> input_landmarks);
		bool DetectLandmarks(const cv::Mat_<uchar> &image, EyeModelParameters& params);
		bool DetectParts(const cv::Mat_<uchar> &image, EyeModelParameters& params);

		// Gets the shape of the current detected landmarks in camera space (given camera calibration)
		// Can only be called after a call to DetectLandmarksInVideo or DetectLandmarksInImage
		cv::Mat_<float> GetShape(float fx, float fy, float cx, float cy) const;

		// A utility bounding box function
		cv::Rect_<float> GetBoundingBox() const;

		// Get the currently non-self occluded landmarks
		cv::Mat_<int> GetVisibilities() const;

		// Reset the model (useful if we want to completelly reinitialise, or we want to track another video)
		void Reset();

		// Reset the model, choosing the face nearest (x,y) where x and y are between 0 and 1.
		void Reset(double x, double y);

		// Reading the model in
		void Read(std::string name);

	private:

		// Helper reading function
		bool Read_CLNF(std::string clnf_location);

		// the speedup of RLMS using precalculated KDE responses (described in Saragih 2011 RLMS paper)
		std::map<int, cv::Mat_<float> >		kde_resp_precalc;

		// The model fitting: patch response computation and optimisation steps
		bool Fit(const cv::Mat_<float>& intensity_image, const std::vector<int>& window_sizes, const EyeModelParameters& parameters);

		// Mean shift computation that uses precalculated kernel density estimators (the one actually used)
		void NonVectorisedMeanShift_precalc_kde(cv::Mat_<float>& out_mean_shifts, const std::vector<cv::Mat_<float> >& patch_expert_responses,
			const cv::Mat_<float> &dxs, const cv::Mat_<float> &dys, int resp_size, float a, int scale, int view_id,
			std::map<int, cv::Mat_<float> >& mean_shifts);

		// The actual model optimisation (update step), returns the model likelihood
		float NU_RLMS(cv::Vec6f& final_global, cv::Mat_<float>& final_local, const std::vector<cv::Mat_<float> >& patch_expert_responses,
			const cv::Vec6f& initial_global, const cv::Mat_<float>& initial_local,
			const cv::Mat_<float>& base_shape, const cv::Matx22f& sim_img_to_ref,
			const cv::Matx22f& sim_ref_to_img, int resp_size, int view_idx, bool rigid, int scale,
			cv::Mat_<float>& landmark_lhoods, const EyeModelParameters& parameters, bool compute_lhood);

		// Generating the weight matrix for the Weighted least squares
		void GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const EyeModelParameters& parameters);

	};
	
}
#endif // EYE_TRACKING_EXPORT_H
