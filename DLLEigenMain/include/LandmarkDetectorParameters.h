
//  Parameters of the CE-CLM, CLNF, and CLM trackers
#ifndef LANDMARK_DETECTOR_PARAM_H
#define LANDMARK_DETECTOR_PARAM_H

#include <vector>

namespace LandmarkDetector
{

struct FaceModelParameters
{

	// A number of RLMS or NU-RLMS iterations
	int num_optimisation_iteration;
	
	// Should pose be limited to 180 degrees frontal
	bool limit_pose;
	
	// Should face validation be done
	bool validate_detections;

	// Landmark detection validator boundary for correct detection, the regressor output 1 (perfect alignment) 0 (bad alignment), 
	float validation_boundary;

	// Used when tracking is going well
	std::vector<int> window_sizes_small;

	// Used when initialising or tracking fails
	std::vector<int> window_sizes_init;
	
	// Used for the current frame
	std::vector<int> window_sizes_current;
	
	// How big is the tracking template that helps with large motions
	float face_template_scale;	
	bool use_face_template;

	// Where to load the model from
	std::string model_location;
	
	// this is used for the smooting of response maps (KDE sigma)
	float sigma;

	float reg_factor;	// weight put to regularisation
	float weight_factor; // factor for weighted least squares

	// should multiple views be considered during reinit
	bool multi_view;
	
	// Based on model location, this affects the parameter settings
	enum LandmarkDetector { CLM_DETECTOR, CLNF_DETECTOR, CECLM_DETECTOR };
	LandmarkDetector curr_landmark_detector;

	// How often should face detection be used to attempt reinitialisation, every n frames (set to negative not to reinit)
	int reinit_video_every;

	// Determining which face detector to use for (re)initialisation, HAAR is quicker but provides more false positives and is not goot for in-the-wild conditions
	// Also HAAR detector can detect smaller faces while HOG SVM is only capable of detecting faces at least 70px across
	// MTCNN detector is much more accurate that the other two, and is even suitable for profile faces, but it is somewhat slower
	enum FaceDetector{HAAR_DETECTOR, HOG_SVM_DETECTOR, MTCNN_DETECTOR};

	std::string haar_face_detector_location;
	std::string mtcnn_face_detector_location;
	FaceDetector curr_face_detector;

	// Should the model be refined hierarchically (if available)
	bool refine_hierarchical;

	// Should the parameters be refined for different scales
	bool refine_parameters;

	FaceModelParameters();

	FaceModelParameters(std::vector<std::string> &arguments);

	private:
		void init();
		void check_model_path(const std::string& root = "/");

};

}

#endif // LANDMARK_DETECTOR_PARAM_H
