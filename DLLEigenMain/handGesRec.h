#pragma once
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <functional>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
//#include <torch/script.h>
#include <onnxruntime_cxx_api.h>
// #ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>

using namespace std;
//using namespace Eigen;

# define M_PI 3.1415926
const int modelWidth = 256, modelHeight = 256, modelWidth_GesRec = 128, modelHeight_GesRec = 128, seqLen = 9, embedDim = 255;
const int modelWidth_GesRec_static = 224, modelHeight_GesRec_static = 224;
const int numAnchors = 2944 , outDim = 18, batchSizePalm = 1, batchSizeHand = 1, numGestures = 17; // 896
const int numKeypointsPalm = 7, numKeypointsHand = 21, numJointConnect = 20, palmDetFreq = 20;
const float scoreClipThrs = 100.0, minScoreThrs = 0.66, minSuppressionThrs = 0.3; //handThrs = 0.85;
float handThrs = 0.85;
const float palm_shift_y = 0.5, palm_shift_x = 0, palm_box_scale = 2.6,
hand_shift_y = 0, hand_shift_x = 0, hand_box_scale = 2.1;
const int gesPreInterv = 0, gesPreTime = 0;
const int handUpId = 9, handDownId = 0, palmUpId = 2, palmDownId = 0;
string gesture_classes[] = { "number_1", "number_2", "number_3", "another_number_3", "number_4", "number_5", "number_6", "thumb_up", "ok", "heart" };
string GES_CLASSES[] = { "doing_other_things", "swiping_down", "swiping_left", "swiping_right", "swiping_up", "thumb_down", "thumb_up", "zoom_in", "zoom_out", "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart", "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart",  "thumb_up",  "thumb_down" };
string GES_DY[] = { "doing_other_things", "thumb_down", "thumb_up", "zoom_in", "zoom_out", "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart"};
string GES_STA[] = { "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart",  "thumb_up",  "thumb_down" , "Fist"}; //, "Zoom" };
string GES_FINAL[] = { "doing_other_things", "swiping_down", "swiping_left", "swiping_right", "swiping_up", "zoom_in", "zoom_out", "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart",  "thumb_up",  "thumb_down" ,"Fist" };
//                               0                  1              2               3              4             5            6         7           8            9           10         11           12      13     14           15            16          
typedef struct anchorType {
	float x;
	float y;
	float w;
	float h;
} anchorType;

typedef struct BoxInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	cv::Mat kpts;
	float score;
	int label;
} BoxInfo;

typedef struct HeadInfo
{
	std::string cls_layer;
	std::string dis_layer;
	int stride;
} HeadInfo;

typedef struct HyperParam {
	int mode; //识别模式，0：静态手势+动态手势；1：静态手势；2：动态手势；
	int vote_num; //静态手势的投票器窗长，比如为5，表示对过去5帧的识别结果进行投票
	int hit_num; //如vote_num为5，hit_num为4时，表示过去5帧内有4帧识别为同一手势，则输出这一手势，否则不输出静态手势识别结果
	int swip_cnt_thres; // 滑动累计帧数
	int miss_swip_cnt_thres; // 滑动过程中允许的未识别成滑动的帧数
	float swip_dist_thres; //滑动的距离（长宽的短边的倍数，如0.1）
	float hand_detect_thres; //手掌检测的置信度,范围为0-1之间的小数，默认为0.85,检测不到手掌时请调小该值，将其他物体误认为手掌时请调大该值
} HyperParam;

struct detRect
{
	cv::Mat img;
	cv::Mat_<float> affineMat;
	cv::Point2f rawCenter;
	cv::Point2f rotCenter;
	// added for hand crop information recording
	cv::Mat image_fullRegion;
	cv::Mat image_handRegion; //after expanding
	float palm_bbox[4];
	float hand_bbox[4];
	float rotate_angle;
	detRect(cv::Mat &src, cv::Mat_<float> &a_mat, cv::Point2f &point1, cv::Point2f &point2, cv::Mat &src_fullRegion, cv::Mat &src_handRegion, cv::Rect &_palm_bbox, cv::Rect &_hand_bbox, float _angle) :
		img(src), affineMat(a_mat), rawCenter(point1), rotCenter(point2), image_fullRegion(src_fullRegion), image_handRegion(src_handRegion), rotate_angle(_angle)
	{
		palm_bbox[0] = _palm_bbox.x;
		palm_bbox[1] = _palm_bbox.y;
		palm_bbox[2] = _palm_bbox.width;
		palm_bbox[3] = _palm_bbox.height;
		hand_bbox[0] = _hand_bbox.x;
		hand_bbox[1] = _hand_bbox.y;
		hand_bbox[2] = _hand_bbox.width;
		hand_bbox[3] = _hand_bbox.height;
	}
};

cv::Mat_<float> computePointAffine(cv::Mat_<float> &pointsMat, cv::Mat_<float> &affineMat, bool inverse)
{
	// cout<<pointsMat.size<<endl;
	if (!inverse)
	{
		cv::Mat_<float> ones = cv::Mat::ones(pointsMat.cols, 1, CV_32F);
		pointsMat.push_back(ones);
		return affineMat * pointsMat;
	}
	else
	{
		pointsMat.row(0) -= affineMat.at<float>(0, 2);
		pointsMat.row(1) -= affineMat.at<float>(1, 2);
		cv::Mat_<float> affineMatInv = affineMat(cv::Rect(0, 0, 2, 2)).inv();
		return affineMatInv * pointsMat;
	}
}

struct detMeta
{
	int fid;     // Frame id, pay attention to MAX_INT;
	int detType; // 0 is palm det, 1 is landmark det
	int xmin, ymin, xmax, ymax;
	float shift_x, shift_y, box_scale;
	cv::Point2f handUp, handDown;
	detMeta(int x_min, int y_min, int x_max, int y_max,
		cv::Point2f &Up, cv::Point2f &Down, int type = 0, int id = 0) :
		fid(id), xmin(x_min), ymin(y_min), xmax(x_max), ymax(y_max), detType(type), handUp(Up), handDown(Down)
	{
		if (type == 0) //detection, 7 hand keypoints
		{
			shift_x = palm_shift_x;
			shift_y = palm_shift_y;
			box_scale = palm_box_scale;
		}
		else  //tracking, 21 hand keypoints
		{
			shift_x = hand_shift_x;
			shift_y = hand_shift_y;
			box_scale = hand_box_scale;
		}
	}
	//cropFrame -> &img
	detRect getTransformedRect(cv::Mat &img, bool square_long = true)
	{
		auto xscale = xmax - xmin, yscale = ymax - ymin;
		cv::Rect paml_bbox = cv::Rect(xmin, ymin, xscale, yscale);
		/* ---- Compute rotatation ---- */
		auto angleRad = atan2(handDown.x - handUp.x, handDown.y - handUp.y);
		auto angleDeg = angleRad * 180 / M_PI; // the angle between palm direction and vertical direction
		// Movement
		// shift_y > 0 : move 0(palmDownId) --> 2(palmUpId); shift_x > 0 : move right hand side of 0->2
		//(x_center, y_center): the center point after shifting but before rotating
		auto x_center = xmin + xscale * (0.5 - shift_y * sin(angleRad) + shift_x * cos(angleRad));
		auto y_center = ymin + yscale * (0.5 - shift_y * cos(angleRad) - shift_x * sin(angleRad));
		cv::Point2f rawCenter(x_center, y_center);
		cv::Mat img_copy, img_handRegion;
		img.copyTo(img_copy);
		float box_scale_handRegion = 3.0;
		auto x_leftTop = max(int(x_center - xscale * box_scale_handRegion / 2.0), 0), y_leftTop = max(int(y_center - yscale * box_scale_handRegion / 2.0), 0);
		//cv::rectangle(img_copy, cv::Rect(x_leftTop, y_leftTop, min(int(xscale*2.6), img_copy.cols - x_leftTop - 1), min(int(yscale*2.6), img_copy.rows - y_leftTop - 1)), { 0.0, 255.0, 0.0 });
		// expand by 2.6 rather than box_scale(2.1/2.6)
		cv::Rect roi_hand = cv::Rect(x_leftTop, y_leftTop, min(int(xscale* box_scale_handRegion), img_copy.cols - x_leftTop - 1), min(int(yscale* box_scale_handRegion), img_copy.rows - y_leftTop - 1));
		img_handRegion = img_copy(roi_hand);
		//cv::imshow("Hand Region", img_handRegion);
		//cv::waitKey(1);

		if (square_long)
			xscale = yscale = max(xscale, yscale); //padding

		auto xrescale = xscale * box_scale, yrescale = yscale * box_scale;
		/* ---- Get cropped Hands ---- */
		// affineMat.size: 2x3
		cv::Mat_<float> affineMat = cv::getRotationMatrix2D(cv::Point2f(img.cols, img.rows) / 2, -angleDeg, 1); // center, angle, scale
		auto bbox = cv::RotatedRect(cv::Point2f(), img.size(), -angleDeg).boundingRect2f();
		affineMat.at<float>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
		affineMat.at<float>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

		cv::Mat rotFrame;
		cv::warpAffine(img, rotFrame, affineMat, bbox.size());

		cv::Mat_<float> pointMat(2, 1);
		pointMat << x_center, y_center;
		cv::Mat_<float> rotPtMat = computePointAffine(pointMat, affineMat, false); // center point after affine transformation
		cv::Point2f rotCenter(rotPtMat(0), rotPtMat(1));
		// Out of range cases
		float xrescale_2 = xrescale / 2, yrescale_2 = yrescale / 2;
		float xDwHalf = min(rotCenter.x, xrescale_2), yDwHalf = min(rotCenter.y, yrescale_2);
		float xUpHalf = rotCenter.x + xrescale_2 > rotFrame.cols ? rotFrame.cols - rotCenter.x : xrescale_2;
		float yUpHalf = rotCenter.y + yrescale_2 > rotFrame.rows ? rotFrame.rows - rotCenter.y : yrescale_2;
		auto cropHand = rotFrame(cv::Rect(rotCenter.x - xDwHalf, rotCenter.y - yDwHalf, xDwHalf + xUpHalf, yDwHalf + yUpHalf));
		//cv::imshow("ROI", cropHand);
		cv::copyMakeBorder(cropHand, cropHand, yrescale_2 - yDwHalf, yrescale_2 - yUpHalf,
			xrescale_2 - xDwHalf, xrescale_2 - xUpHalf, cv::BORDER_CONSTANT);
		return detRect(cropHand, affineMat, rawCenter, rotCenter, img_copy, img_handRegion, paml_bbox, roi_hand, angleDeg);
	}
};

struct handLandmarks
{
	vector<Ort::Session*> sessions;
	cv::Mat_<float> anchors;
	cv::Mat image_gestures;
	HyperParam hyper_params;
};

char* modelDecryption(ifstream &inFile, string &pwd, int encrypt_num, int model_size);
Ort::Session* sessionInit(string modelPath, int decrypt_num, string pwd);

extern "C" _declspec(dllexport) void* __stdcall handLandmarks_Init(const char* p_palmDetModel, const char* p_staGesRecModel, const char* p_handGesRecModel,
	const char* p_handFeatModel, const char* p_anchorFile, HyperParam hyper_params); 
//handLandmarks_Init(const char* p_palmDetModel, const char* p_handGesRecModel,
//	const char* p_handFeatModel, const char* p_anchorFile);
extern "C" _declspec(dllexport) int __stdcall handLandmarks_inference(void* p_self, void* image, int* image_shape, float* hand_region, bool debug_print);
void decodeBoxes(const cv::Mat& rawBoxes, const cv::Mat& rawScores, const cv::Mat & anchors, std::vector<BoxInfo> & boxes);
void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH);