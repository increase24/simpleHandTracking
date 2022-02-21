#pragma once

typedef struct HyperParam {
	int mode; //识别模式，0：静态手势+动态手势; 1：静态手势；2：动态手势；
	// 静态手势包括 "doing_other_things", "swiping_down", "swiping_left", "swiping_right", "swiping_up", "zoom_in", "zoom_out"
	// 动态手势包括 "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart",  "thumb_up",  "thumb_down"
	int vote_num; //静态手势的投票器窗长，比如为5，表示对过去5帧的识别结果进行投票
	int hit_num; //如vote_num为5，hit_num为4时，表示过去5帧内有4帧识别为同一手势，则输出这一手势，否则不输出静态手势识别结果
	int swip_cnt_thres; // 滑动累计帧数
	int miss_swip_cnt_thres; // 滑动过程中允许的未识别成滑动的帧数
	float swip_dist_thres; //滑动的距离（长宽的短边的倍数，如0.1）
	float hand_detect_thres; //手掌检测的置信度,范围为0-1之间的小数，默认为0.85,检测不到手掌时请调小该值，将其他物体误认为手掌时请调大该值
} HyperParam;

extern "C" _declspec(dllexport) void* __stdcall handLandmarks_Init(const char* p_palmDetModel, const char* p_staGesRecModel, const char* p_handGesRecModel,
	const char* p_handFeatModel, const char* p_anchorFile, HyperParam hyper_params);
extern "C" _declspec(dllexport) int __stdcall handLandmarks_inference(void* p_self, void* image, int* image_shape, float* output, bool debug_print); 

