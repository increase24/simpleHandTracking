//  Parameters of the CE-CLM, CLNF, and CLM trackers
#ifndef EYE_TRRACKING_H
#define EYE_TRRACKING_H

#include <vector>

namespace LandmarkDetector
{

struct EyeModelParameters
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

	// Should the model be refined hierarchically (if available)
	bool refine_hierarchical;

	// Should the parameters be refined for different scales
	bool refine_parameters;

	EyeModelParameters();

	EyeModelParameters(std::vector<std::string> &arguments);

	private:
		void init();
		void check_model_path(const std::string& root = "/");

};

}

#endif // EYE_TRRACKING_PARAM_H
