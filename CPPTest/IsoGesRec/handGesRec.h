#pragma once

typedef struct HyperParam {
	int mode; //ʶ��ģʽ��0����̬����+��̬����; 1����̬���ƣ�2����̬���ƣ�
	// ��̬���ư��� "doing_other_things", "swiping_down", "swiping_left", "swiping_right", "swiping_up", "zoom_in", "zoom_out"
	// ��̬���ư��� "number_1", "number_2", "number_3", "number_4", "number_5", "number_6", "ok", "heart",  "thumb_up",  "thumb_down"
	int vote_num; //��̬���Ƶ�ͶƱ������������Ϊ5����ʾ�Թ�ȥ5֡��ʶ��������ͶƱ
	int hit_num; //��vote_numΪ5��hit_numΪ4ʱ����ʾ��ȥ5֡����4֡ʶ��Ϊͬһ���ƣ��������һ���ƣ����������̬����ʶ����
	int swip_cnt_thres; // �����ۼ�֡��
	int miss_swip_cnt_thres; // ���������������δʶ��ɻ�����֡��
	float swip_dist_thres; //�����ľ��루����Ķ̱ߵı�������0.1��
	float hand_detect_thres; //���Ƽ������Ŷ�,��ΧΪ0-1֮���С����Ĭ��Ϊ0.85,��ⲻ������ʱ���С��ֵ����������������Ϊ����ʱ������ֵ
} HyperParam;

extern "C" _declspec(dllexport) void* __stdcall handLandmarks_Init(const char* p_palmDetModel, const char* p_staGesRecModel, const char* p_handGesRecModel,
	const char* p_handFeatModel, const char* p_anchorFile, HyperParam hyper_params);
extern "C" _declspec(dllexport) int __stdcall handLandmarks_inference(void* p_self, void* image, int* image_shape, float* output, bool debug_print); 

