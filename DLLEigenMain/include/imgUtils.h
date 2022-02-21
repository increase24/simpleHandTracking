#pragma once
#include<opencv2/core/core.hpp>
#include<vector>

using namespace cv;

namespace ImgUtils
{
	Point2f affine_transform(Point2f pt, Mat &t);
	Mat get_affine_transform(Point2f center, Point2f scale, float rot, int* output_size,
		Point2f shift, bool inv);
	Point2f get_dir(Point2f src_point, float rot_rad);
	Mat crop(Mat &img, Point2f center, Point2f scale, int *output_size, float rot);
	Mat crop_by_trans(Mat &img, Mat &trans, int *output_size);
	std::vector<Point2f> pts2cs(Mat_<float> &pts);
	Mat ConvertMat2CHW(Mat &input_img);

	Point2f get_3rd_point(Point2f a, Point2f b);
	Point2f Point2fMul(Point2f pt_x, Point2f pt_y);

	bool isRotationMatrix(Mat &R);
	Vec3f rotationMatrixToEulerAngles(Mat &R);
	Mat eulerAnglesToRotationMatrix(Vec3f &theta);

	//class KalmanFilterForPoints {
	//public:
	//	KalmanFilterForPoints(int state_num, int measure_num, float cov_process, float cov_measure);

	//	Mat update(Mat measurement);

	//private:
	//	KalmanFilter *KF;
	//};

	//class LowPassFilter {

	//private:
	//	double y, a, s;
	//	bool initialized;

	//	void setAlpha(double alpha);

	//public:

	//	LowPassFilter(double alpha, double initval = 0.0);

	//	double filter(double value);

	//	double filterWithAlpha(double value, double alpha);

	//	bool hasLastRawValue(void);

	//	double lastRawValue(void);

	//};

	//class OneEuroFilter {

	//private:
	//	double freq;
	//	double mincutoff;
	//	double beta_;
	//	double dcutoff;
	//	LowPassFilter *x;
	//	LowPassFilter *dx;
	//	TimeStamp lasttime;

	//	double alpha(double cutoff);

	//	void setFrequency(double f);

	//	void setMinCutoff(double mc);

	//	void setBeta(double b);

	//	void setDerivateCutoff(double dc);

	//public:

	//	OneEuroFilter(double freq,
	//		double mincutoff = 1.0, double beta_ = 0.0, double dcutoff = 1.0);

	//	double filter(double value, TimeStamp timestamp = UndefinedTime);

	//	~OneEuroFilter(void);

	//};


	class PoseEstimator {
	public:

		PoseEstimator(int* img_size, std::string model_file_name);

		/*Solve pose from all the 68 image points
		Return (rotation_vector, translation_vector) as pose.*/
		std::vector<Mat_<float>> solve_pose_by_68_points(Mat_<float> &image_points);

		void draw_axes(Mat &img, Mat_<float> &R, Mat_<float> &t);
	

	private:
		/* attributes */
		int size[2];

		// 3D model points.
		Mat_<float> model_points_68;

		// Camera internals
		float focal_length;
		float camera_center[2];
		Mat_<float> camera_matrix;
		
		// Assuming no lens distortion
		Mat_<float> dist_coeffs;

		// Rotation vector and translation vector
		Mat_<float> r_vec;
		Mat_<float> t_vec;


		/* methods */
		Mat_<float> _get_full_model_points(std::string filename);

	};
}