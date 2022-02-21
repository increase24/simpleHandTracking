#pragma once
struct detection_box
{
	float xmin = 0;
	float xmax = 0;
	float ymin = 0;
	float ymax = 0;
	float score = 0;
};

struct DetHands
{
	detection_box l_hand;
	detection_box r_hand;
};

extern "C" _declspec(dllexport) void* __stdcall handLandmarks_Init(const char* p_palmDetModel, const char* p_anchorFile);
extern "C" _declspec(dllexport) int __stdcall handLandmarks_inference(void* p_self, void* image, int* image_shape, DetHands &hands, bool debug_print);

