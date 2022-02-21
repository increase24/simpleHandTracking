#include "pch.h"
#include "handGesRec.h"
//#include <jni.h>
#include <string>



void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
	std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	std::vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i) {
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}
	for (int i = 0; i < int(input_boxes.size()); ++i) {
		for (int j = i + 1; j < int(input_boxes.size());) {
			float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
			float w = (std::max)(float(0), xx2 - xx1 + 1);
			float h = (std::max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);
			if (ovr >= NMS_THRESH) {
				input_boxes.erase(input_boxes.begin() + j);
				vArea.erase(vArea.begin() + j);
			}
			else {
				j++;
			}
		}
	}
}

inline float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x)); //fast_exp(-x);
}

inline float clamp(float x, float thresh = 100.f) {
	if (x < -thresh) {
		return -thresh;
	}
	else if (x > thresh) {
		return thresh;
	}
	else {
		return x;
	}

}

void decodeBoxes(const cv::Mat& rawBoxes, const cv::Mat& rawScores, const cv::Mat & anchors, std::vector<BoxInfo> & boxes) {
	//float modelWidth = 256.0, modelHeight = 256.0;
	for (int idx = 0; idx < rawScores.rows; idx++) {
		float score = rawScores.at<float>(idx, 0);
		score = clamp(score);
		score = sigmoid(score);
		auto x_center = rawBoxes.at<float>(idx, 0) / modelWidth * anchors.at<float>(idx, 2) + anchors.at<float>(idx, 0);
		auto y_center = rawBoxes.at<float>(idx, 1) / modelHeight * anchors.at<float>(idx, 3) + anchors.at<float>(idx, 1);
		auto w = rawBoxes.at<float>(idx, 2) / modelWidth * anchors.at<float>(idx, 2);
		auto h = rawBoxes.at<float>(idx, 3) / modelHeight * anchors.at<float>(idx, 3);
		BoxInfo box;
		box.x1 = x_center - w / 2.f;
		box.y1 = y_center - h / 2.f;
		box.x2 = x_center + w / 2.f;
		box.y2 = y_center + h / 2.f;
		// box.kpts = rawBoxes(cv::Rect(idx, 4, idx + 1, outDim - 4));
		box.kpts = rawBoxes(cv::Range(idx, idx + 1), cv::Range(4, outDim));
		box.score = score;
		box.label = 0;
		boxes.push_back(box);
		//if (score > minSuppressionThrs) {
		//
		//}
	}
}

char* modelDecryption(ifstream &inFile, string &pwd, int encrypt_num, int model_size)
{
	int pwd_size = pwd.size();
	int i = 0, j = 0;
	char ch;
	char *decryp_model = new char[model_size];
	while (inFile.get(ch))
	{
		if (i < encrypt_num)
		{
			decryp_model[i] = ch ^ pwd[j >= pwd_size ? j = 0 : j++];
		}
		else
		{
			decryp_model[i] = ch;
		}
		i++;
	}
	return decryp_model;
}

Ort::Session* sessionInit(string modelPath, int decrypt_num, string pwd)
{
	Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	std::wstring widstr = std::wstring(modelPath.begin(), modelPath.end());
	const wchar_t *ModelPath = widstr.c_str();

	ifstream stream(ModelPath, std::ios::binary | std::ios::ate);
	streamsize size = stream.tellg();
	stream.seekg(0, std::ios::beg);
	char* model = modelDecryption(stream, pwd, decrypt_num, static_cast<int>(size));
	stream.close();
	Ort::Session* session = new Ort::Session(*env, model, static_cast<size_t>(size), session_options);
	delete[]model;

	return session;
}

cv::Mat mat2chw(cv::Mat inFrame)
{
	auto size = inFrame.size();
	cv::Size newsize(size.width, size.height * 3);
	cv::Mat frameCHW(newsize, CV_32FC1);
	for (int i = 0; i < inFrame.channels(); ++i)
	{
		cv::extractChannel(
			inFrame,
			cv::Mat(
				size.height,
				size.width,
				CV_32FC1,
				&(frameCHW.at<float>(size.height*size.width*i))),
			i);
	}
	return frameCHW;
}

void* __stdcall handLandmarks_Init(const char* p_palmDetModel, const char* p_staGesRecModel, const char* p_handGesRecModel,
	const char* p_handFeatModel, const char* p_anchorFile, HyperParam hyper_params)
{
	//cv::Mat padding = load_gestureImg(gesture_classes);
	// file vars
	string anchorFile = p_anchorFile;
	string palmModel = p_palmDetModel;
	string staGesModel = p_staGesRecModel;
	string gesFeatModel = p_handFeatModel;
	string gesModel = p_handGesRecModel;

	Ort::Session* sessionPalm = sessionInit(palmModel, 4444, "Adasd@#4s24!3da");
	Ort::Session* sessionSta = sessionInit(staGesModel, 6666, "Z0hRAJmCkGBHpkJ2*");
	Ort::Session* sessionFeat = sessionInit(gesFeatModel, 7777, "z0hrajmckgbhpkj2*");
	Ort::Session* sessionGes = sessionInit(gesModel, 6666, "Z0hRAJmCkGBHpkJ2*");

	// opencv vars
	//int rawHeight, rawWidth, cropWidthLowBnd, cropWidth, cropHeightLowBnd, cropHeight;
	//cv::Mat frame, rawFrame, showFrame, cropFrame, inFrame, tmpFrame;
	//cv::Mat nchors;
	//cv::VideoCapture cap;
	//deque<detRect> cropHands;
	//deque<detMeta> handMetaForward;

	/* ---- load anchor binary file ---- */
	cv::Mat_<float> anchors_cvMat = cv::Mat::zeros(numAnchors, 4, CV_32FC1);
	fstream fin_cvMat(anchorFile, ios::in | ios::binary);
	fin_cvMat.read((char *)anchors_cvMat.data, anchors_cvMat.cols* anchors_cvMat.rows * sizeof(float));
	//fstream fin(anchorFile, ios::in | ios::binary);
	//fin.read((char *)anchors.data_ptr(), anchors.numel() * sizeof(float));

	/* ---- init ONNX rt ---- */

	//Ort::AllocatorWithDefaultOptions allocator;
	std::vector<int64_t> palm_input_node_dims = { batchSizePalm, 3, modelHeight, modelWidth };
	size_t palm_input_tensor_size = batchSizePalm * 3 * modelHeight * modelWidth;
	std::vector<int64_t> hand_input_node_dims = { batchSizeHand, 3, modelHeight, modelWidth };
	size_t hand_input_tensor_size = batchSizeHand * 3 * modelHeight * modelWidth;
	std::vector<float> input_tensor_values(palm_input_tensor_size);
	std::vector<const char *> input_node_names = { "input" };
	std::vector<const char *> output_node_names = { "output1", "output2" };


	handLandmarks* handLdks = new handLandmarks();
	handLdks->sessions.push_back(sessionPalm);
	handLdks->sessions.push_back(sessionSta);
	handLdks->sessions.push_back(sessionFeat);
	handLdks->sessions.push_back(sessionGes);
	handLdks->anchors = anchors_cvMat;
	handLdks->hyper_params = hyper_params;

	return handLdks;
}


int __stdcall handLandmarks_inference(void* p_self, void* image, int* image_shape, float* hand_region, bool debug_print)
{
	static handLandmarks* handLdks = (handLandmarks*)(p_self);
	static Ort::Session* sess_palmDet = handLdks->sessions[0];
	static Ort::Session* sess_handGesStatic = handLdks->sessions[1];
	static Ort::Session* sess_handFeat = handLdks->sessions[2];
	static Ort::Session* sess_handGesRec = handLdks->sessions[3];
	static HyperParam hyper_params = handLdks->hyper_params;
	handThrs = hyper_params.hand_detect_thres;

	unsigned char* _input = (unsigned char*)(image);
	int img_h = image_shape[0];
	int img_w = image_shape[1];
	// convert unsigned char* to cv::Mat
	cv::Mat rawFrame(img_h, img_w, CV_8UC3, _input);

	//static configuration
	static Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	//static int modelWidth = 256, modelHeight = 256, modelWidth_GesRec = 224, modelHeight_GesRec = 224;
	static size_t palm_input_tensor_size = batchSizePalm * 3 * modelHeight * modelWidth;
	static std::vector<int64_t> palm_input_node_dims = { batchSizePalm, 3, modelHeight, modelWidth };
	static std::vector<const char *> input_node_names = { "input" };
	static std::vector<const char *> output_node_names = { "output1", "output2" };
	static std::vector<const char *> input_node_name_Feat = { "input" };
	static std::vector<const char *> input_node_name_GesRec = { "input" };
	static std::vector<const char *> output_node_name_GesRec = { "output" };
	static std::vector<const char *> output_node_name_Feat = { "output" };
	static std::vector<const char *> input_node_name_sta = { "input" };
	static std::vector<const char *> output_node_name_sta = { "output" };
	static cv::Mat input_seq_img(batchSizeHand, seqLen * embedDim, CV_32FC1);
	static cv::Mat input_seq_img_tmp(batchSizeHand, seqLen * embedDim, CV_32FC1);
	static deque<cv::Point2f> swip_mov_dist;
	input_seq_img_tmp = cv::Scalar(0);
	static int frameOffset = embedDim;  //3 * modelHeight_GesRec * modelWidth_GesRec;
	static int counter_frame = 0;
	static int counter_frame_sta = 0;
	int last_idx_max = 0;
	static int miss_hand_cnt = 0;
	static int miss_swip_cnt = 0;
	static int swip_cnt = 0;
	float output_prob = 0.0f;
	// voter for static gesture recognition
	static int* vote_sta = new int[hyper_params.vote_num];

	// queue of palm bbox and hand region
	static deque<detRect> cropHands, cropHandsOri;
	static deque<detMeta> handMetaForward, handMetaForwardOri;
	static ofstream record_log;
	cv::Mat frame, cropFrame, paddingFrame, centerCropFrame, inFrame, tmpFrame, showFrame, showFrameOri, resizeFrame, gesInpFrame;
	//torch::Tensor inputRawTensor, inputTensor, rawBoxesP, rawScoresP;
	int rawHeight, rawWidth, cropHeightLowBnd, cropWidthLowBnd, cropHeight, cropWidth, paddingHeight, paddingWidth;
	float handCenterX, handCenterY, bbox_len;
	frame = rawFrame;
	rawHeight = frame.rows;
	rawWidth = frame.cols;

	/* crop long edge -> padding short edge */
	//// cropping long edge
	//if (rawHeight > rawWidth)
	//{
	//	cropHeightLowBnd = (rawHeight - rawWidth) / 2;
	//	cropWidthLowBnd = 0;
	//	cropHeight = cropWidth = rawWidth;
	//}
	//else
	//{
	//	cropWidthLowBnd = (rawWidth - rawHeight) / 2;
	//	cropHeightLowBnd = 0;
	//	cropHeight = cropWidth = rawHeight;
	//}
	//float swip_dist_thresh = cropHeight * hyper_params.swip_dist_thres;
	//int showHeight = cropHeight, showWidth = cropWidth;
	//cv::Rect ROI(cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
	//cropFrame = frame(ROI);
	//cropFrame.copyTo(showFrame);
	
	// padding short edge
	if (rawHeight > rawWidth)
	{
		int padding_w = (rawHeight - rawWidth) / 2;
		int padding_h = 0;
		cv::copyMakeBorder(frame, cropFrame, padding_h, padding_h, padding_w, padding_w, cv::BORDER_CONSTANT, 0);
		cropHeight = cropWidth = rawHeight;
	}
	else
	{
		int padding_w = 0;
		int padding_h = (rawWidth - rawHeight) / 2;
		cv::copyMakeBorder(frame, cropFrame, padding_h, padding_h, padding_w, padding_w, cv::BORDER_CONSTANT, 0);
		cropHeight = cropWidth = rawWidth;
	}

	float swip_dist_thresh = cropHeight * hyper_params.swip_dist_thres;
	int showHeightOri = cropHeight, showWidthOri = cropWidth;
	cropFrame.copyTo(showFrameOri);
	

	/* --------------------------------------- perform palm detection ------------------------------------- */
	if (handMetaForward.empty())
	{
		cv::resize(cropFrame, inFrame, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
		int showHeight = inFrame.rows, showWidth = inFrame.cols;
		inFrame.copyTo(showFrame);
		inFrame.copyTo(resizeFrame);

		cv::cvtColor(inFrame, inFrame, cv::COLOR_BGR2RGB);
		inFrame.convertTo(inFrame, CV_32F);
		inFrame = inFrame / 127.5 - 1.0;

		cv::Mat frameCHW = mat2chw(inFrame);
		auto stop1 = chrono::high_resolution_clock::now();
		Ort::Value inputTensor_ort = Ort::Value::CreateTensor<float>(memory_info, (float_t *)frameCHW.data, palm_input_tensor_size, palm_input_node_dims.data(), 4);
		auto output_tensors = sess_palmDet->Run(Ort::RunOptions(nullptr), input_node_names.data(), &inputTensor_ort, 1, output_node_names.data(), 2);
		auto stop2 = chrono::high_resolution_clock::now();
		auto infer_time1 = chrono::duration_cast<chrono::milliseconds>(stop2 - stop1).count();
		if (debug_print)
		{
			cout << "hand detection time cost:" << infer_time1 << "ms" << endl;
		}
		float* rawBoxesPPtr = output_tensors[0].GetTensorMutableData<float>(); // bounding box
		float* rawScoresPPtr = output_tensors[1].GetTensorMutableData<float>(); // confidence

		/* ---- decode the boxes ----*/
		std::vector<BoxInfo> detectionBoxes;
		BoxInfo detectionBox;
		BoxInfo outputBox;
		cv::Mat rawBoxes(numAnchors, outDim, CV_32FC1, rawBoxesPPtr);
		cv::Mat rawScores(numAnchors, 1, CV_32FC1, rawScoresPPtr);

		decodeBoxes(rawBoxes, rawScores, handLdks->anchors, detectionBoxes);
		//printf("palmDetection num of detectionBoxes : %d\n", detectionBoxes.size());
		nms(detectionBoxes, minSuppressionThrs);
		//printf("palmDetection num of detectionBoxes after NMS : %d\n", detectionBoxes.size());

		for (int i = 0; i < batchSizePalm; i++)
		{
			// select the bbox with the highest confidence
			int opt_idx = 0;
			float max_score = 0.0f;
			for (int idx = 0; idx < detectionBoxes.size(); idx++) {
				float score = detectionBoxes[idx].score;
				if (score > max_score && score > handThrs) {
					max_score = score;
					opt_idx = idx;
				}
			}
			
			detectionBox = detectionBoxes[opt_idx];
			outputBox = detectionBoxes[opt_idx];

			auto ymin = detectionBox.y1 * showHeight;
			auto xmin = detectionBox.x1 * showWidth;
			auto ymax = detectionBox.y2 * showHeight;
			auto xmax = detectionBox.x2 * showWidth;

			auto yminOri = detectionBox.y1 * showHeightOri;
			auto xminOri = detectionBox.x1 * showWidthOri;
			auto ymaxOri = detectionBox.y2 * showHeightOri;
			auto xmaxOri = detectionBox.x2 * showWidthOri;

			auto kpts = detectionBox.kpts;
			handCenterX = (xminOri + xmaxOri) / 2;
			handCenterY = (yminOri + ymaxOri) / 2;

			hand_region[0] = detectionBox.x1;
			hand_region[1] = detectionBox.y1;
			hand_region[2] = detectionBox.x2;
			hand_region[3] = detectionBox.y2;

			cv::Point2f handUp = cv::Point2f(kpts.at<float>(0, palmUpId * 2), kpts.at<float>(0, palmUpId * 2 + 1)),
				handDown = cv::Point2f(kpts.at<float>(0, palmDownId * 2), kpts.at<float>(0, palmDownId * 2 + 1));

			if (max_score > handThrs)
			{
				if (debug_print) {
					cout << "max_score: " << max_score << endl;
				}
				handMetaForward.push_back(detMeta(xmin, ymin, xmax, ymax, handUp, handDown, 0));
				handMetaForwardOri.push_back(detMeta(xminOri, yminOri, xmaxOri, ymaxOri, handUp, handDown, 0));
				//cv::rectangle(showFrame, cv::Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)), cv::Scalar(0, 0, 255), 1, 1, 0);
			}
		}
	}

	while (!handMetaForward.empty())
	{
		cropHands.push_back(handMetaForward.front().getTransformedRect(resizeFrame));
		cropHandsOri.push_back(handMetaForwardOri.front().getTransformedRect(showFrameOri));
		//cv::circle(showFrame, cropHands.front().rawCenter, 2, { 255, 0, 0 }, 2);
		handMetaForward.pop_front();
		handMetaForwardOri.pop_front();
	}
	//
	/* ----------------- Hand Keypoint Detection NN Inference ---------------------- */
	cv::Mat cropImage_Affine;
	float v_max = 0.0;
	int idx_max = 0;
	float v_max_static = 0.0;
	int idx_max_static = 0;
	int res_final = 0;
	int batchSizeHand = cropHands.size();
	if (batchSizeHand)
	{
		auto cropHand = cropHands.front();
		hand_region[4] = cropHand.rawCenter.x / (float)cropHand.image_fullRegion.cols;
		hand_region[5] = cropHand.rawCenter.y / (float)cropHand.image_fullRegion.rows;

		/* ---- Draw Hand landmarks ---- */
		/* ---- Hand Gesture Recognition ---- */
		size_t memOffset = numKeypointsHand * 3;
		float gesDyMaxProb, gesStaMaxProb;
		int gesDyMaxPos, gesStaMaxPos;
		bool outResult = FALSE;
		for (int i = 0; i < batchSizeHand; i++) //batchSizeHand
		{
			if (i < 1)
			{
				//auto cropHandTensor = torch::empty({ 1, modelWidth_GesRec, modelHeight_GesRec, 3 });
				/* ---------  padding and resize  --------- */
				auto h_tmpCropFrameOri = cropHandsOri.front().image_handRegion.rows;
				auto w_tmpCropFrameOri = cropHandsOri.front().image_handRegion.cols;
				auto long_sideOri = max(h_tmpCropFrameOri, w_tmpCropFrameOri);
				cv::Mat tmpCropFrameOri = cv::Mat::zeros(cv::Size(long_sideOri, long_sideOri), CV_8UC3);
				if (h_tmpCropFrameOri > w_tmpCropFrameOri) // height > width
				{
					cv::Rect region(int(0.5*(long_sideOri - w_tmpCropFrameOri)), 0, w_tmpCropFrameOri, h_tmpCropFrameOri);
					cropHandsOri.front().image_handRegion.copyTo(tmpCropFrameOri(region));
				}
				else
				{
					cv::Rect region(0, int(0.5*(long_sideOri - h_tmpCropFrameOri)), w_tmpCropFrameOri, h_tmpCropFrameOri);
					cropHandsOri.front().image_handRegion.copyTo(tmpCropFrameOri(region));
				}
				// cv::imshow("tmpCropFrameOri", tmpCropFrameOri);
				cv::cvtColor(tmpCropFrameOri, tmpCropFrameOri, cv::COLOR_BGR2RGB);
				tmpCropFrameOri.convertTo(tmpCropFrameOri, CV_32F);
				cv::resize(tmpCropFrameOri, tmpCropFrameOri, cv::Size(modelWidth_GesRec, modelHeight_GesRec), 0, 0, cv::INTER_LINEAR);
				tmpCropFrameOri /= 255.0;
				cv::Mat frameCHWOri = mat2chw(tmpCropFrameOri);

				auto h_tmpCropFrame = cropHands.front().image_handRegion.rows;
				auto w_tmpCropFrame = cropHands.front().image_handRegion.cols;
				auto long_side = max(h_tmpCropFrame, w_tmpCropFrame);
				cv::Mat tmpCropFrame = cv::Mat::zeros(cv::Size(long_side, long_side), CV_8UC3);
				if (h_tmpCropFrame > w_tmpCropFrame) // height > width
				{
					cv::Rect region(int(0.5*(long_side - w_tmpCropFrame)), 0, w_tmpCropFrame, h_tmpCropFrame);
					cropHands.front().image_handRegion.copyTo(tmpCropFrame(region));
				}
				else
				{
					cv::Rect region(0, int(0.5*(long_side - h_tmpCropFrame)), w_tmpCropFrame, h_tmpCropFrame);
					cropHands.front().image_handRegion.copyTo(tmpCropFrame(region));
				}

				// cv::imshow("tmpCropFrame", tmpCropFrame);
				cv::cvtColor(tmpCropFrame, tmpCropFrame, cv::COLOR_BGR2RGB);
				tmpCropFrame.convertTo(tmpCropFrame, CV_32F);
				cv::resize(tmpCropFrame, tmpCropFrame, cv::Size(modelWidth_GesRec, modelHeight_GesRec), 0, 0, cv::INTER_LINEAR);
				tmpCropFrame /= 255.0;

				cv::Mat frameCHW = mat2chw(tmpCropFrame);

				counter_frame = 0;
				//tmpCropHand = tmpCropHand.permute({ 0, 3, 1, 2 }).to(torch::kCPU, false, true);
				std::vector<int64_t> cropHand_input_node_dims = { 1, 3, modelHeight_GesRec, modelWidth_GesRec };
				size_t cropHand_input_tensor_size = 1 * 3 * modelHeight_GesRec * modelWidth_GesRec;
				Ort::Value input_tensorOri = Ort::Value::CreateTensor<float>(memory_info,
					(float_t *)frameCHWOri.data, cropHand_input_tensor_size, cropHand_input_node_dims.data(), 4);
				Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
					(float_t *)frameCHW.data, cropHand_input_tensor_size, cropHand_input_node_dims.data(), 4);
				auto output_feat_tensors = sess_handFeat->Run(Ort::RunOptions{ nullptr },
					input_node_name_Feat.data(), &input_tensor, 1, output_node_name_Feat.data(), 1);
				auto output_sta_tensors = sess_handGesStatic->Run(Ort::RunOptions{ nullptr },
					input_node_name_sta.data(), &input_tensorOri, 1, output_node_name_sta.data(), 1);

				float *GesOutputFeat = output_feat_tensors[0].GetTensorMutableData<float>();
				//auto GesDyOutputFeatTensor = torch::from_blob(GesOutputFeat, { 1, 256 });
				//float *GesStaOutput = output_feat_tensors[0].GetTensorMutableData<float>();
				float *GesStaOutput = output_sta_tensors[0].GetTensorMutableData<float>();
				cv::Mat GesStaOutputTensor(1, 11, CV_32FC1, GesStaOutput);
				//auto GesStaOutputTensor = torch::from_blob(GesStaOutput, { 1, 13 });

				memcpy(input_seq_img_tmp.data, input_seq_img.data, sizeof(float) * seqLen * frameOffset);
				memcpy((float*)input_seq_img.data + (seqLen - 1) * frameOffset, output_feat_tensors[0].GetTensorMutableData<float>(), //tmpCropHand.data_ptr(),
					sizeof(float) * frameOffset);
				memcpy(input_seq_img.data, (float*)input_seq_img_tmp.data + frameOffset, sizeof(float) * (seqLen - 1) * frameOffset);

				std::vector<int64_t> input_feat_node_dims = { 1, seqLen, 1, embedDim };
				size_t input_tensor_feat_size = 1 * embedDim * seqLen;
				Ort::Value input_tensor_feat = Ort::Value::CreateTensor<float>(memory_info,
					(float_t *)input_seq_img.data, input_tensor_feat_size, input_feat_node_dims.data(), 4);

				auto output_ges_tensors = sess_handGesRec->Run(Ort::RunOptions{ nullptr },
					input_node_name_GesRec.data(), &input_tensor_feat, 1, output_node_name_GesRec.data(), 1);

				float *GesOutput = output_ges_tensors[0].GetTensorMutableData<float>();
				cv::Mat GesDyOutputTensor(1, 12, CV_32FC1, GesOutput);

				double minVal = 0, maxVal = 0;
				int minIdx[2] = {};
				int maxIdx[2] = {};
				cv::minMaxIdx(GesDyOutputTensor, &minVal, &maxVal, minIdx, maxIdx);

				gesDyMaxProb = static_cast<float>(maxVal);
				gesDyMaxPos = maxIdx[1];
				//if (gesDyMaxPos == 7 || gesDyMaxPos == 8)
				//{
				//cout << "gesture dynamic output: " << GES_DY[gesDyMaxPos] << endl;
				//cout << "gesDyMaxPos: " << gesDyMaxPos << endl;
				//cout << "gesDyMaxProb: " << gesDyMaxProb << endl;
				//}

				minVal = 0, maxVal = 0;
				minIdx[2] = {};
				maxIdx[2] = {};
				cv::minMaxIdx(GesStaOutputTensor, &minVal, &maxVal, minIdx, maxIdx);

				gesStaMaxProb = static_cast<float>(maxVal);
				gesStaMaxPos = maxIdx[1];

				//cout << "gesture static output: " << GES_STA[gesStaMaxPos] << endl;
				//cout << "gesStaMaxPos: " << gesStaMaxPos << endl;
				//cout << "gesStaMaxProb: " << gesStaMaxProb << endl;

				//last_idx_max = gesDyMaxPos;
				//output_prob = gesStaMaxProb;

				//                                              Thumb down                         Thumb up
				if ((gesStaMaxPos == gesDyMaxPos - 5)||(gesStaMaxPos==9&&gesDyMaxPos==1)||(gesStaMaxPos==8&&gesDyMaxPos==2))
				{
					output_prob = gesStaMaxProb;
					last_idx_max = gesStaMaxPos + 7;
				}
				else if (gesStaMaxProb > 0.9&&gesStaMaxPos != 8&& gesStaMaxPos != 9) // && gesStaMaxPos !=9 && gesStaMaxProb > gesDyMaxProb
				{
					output_prob = gesStaMaxProb;
					last_idx_max = gesStaMaxPos + 7;
					//cout << "gesture static output: " << GES_STA[gesStaMaxPos] << endl;
					//cout << "gesStaMaxPos: " << gesStaMaxPos << endl;
				}
				else
				{
					output_prob = 0.f;
					last_idx_max = 0;
				}

				if (gesDyMaxProb > 0.75) // && (GES_DY[gesDyMaxPos] == "zoom_in" || GES_DY[gesDyMaxPos] == "zoom_out" || GES_DY[gesDyMaxPos] == "doing_other_things")
				{

					// cout << "gesture dynamic output: " << GES_DY[gesDyMaxPos] << endl;
					if (GES_DY[gesDyMaxPos] == "zoom_in")
					{
						last_idx_max = 5;
						output_prob = gesDyMaxProb;
					}
					else if (GES_DY[gesDyMaxPos] == "zoom_out")
					{
						last_idx_max = 6;
						output_prob = gesDyMaxProb;
					}
					else if (GES_DY[gesDyMaxPos] == "doing_other_things")
					{
						last_idx_max = 0;
						output_prob = gesDyMaxProb;
					}
					
				}

				//if (gesDyMaxProb > 0.85 && GES_DY[gesDyMaxPos] == "number_5")
				//{
				//	output_prob = gesDyMaxProb;
				//	last_idx_max = 11;
				//}

				/*if (gesDyMaxProb > 0.79 && GES_DY[gesDyMaxPos] == "number_1")
				{
					output_prob = gesDyMaxProb;
					last_idx_max = 7;
				}*/

				if (gesStaMaxProb > 0.95 && GES_STA[gesStaMaxPos] == "number_1" 
					&& gesDyMaxProb > 0.95 && GES_DY[gesDyMaxPos] == "number_1")
				{
					output_prob = gesStaMaxProb;
					last_idx_max = 7;
				}
				if (debug_print)
				{
					cout << "gesture static output: " << GES_STA[gesStaMaxPos] << endl;
					cout << "gesStaMaxProb: " << gesStaMaxProb << endl;

					cout << "gesture dynamic output: " << GES_DY[gesDyMaxPos] << endl;
					cout << "gesDyMaxProb: " << gesDyMaxProb << endl;
				}
				
				//cout << "last_idx_max: " << last_idx_max << endl;

				/*if (gesStaMaxProb > 0.95 && gesStaMaxProb > gesDyMaxProb)
				{
					last_idx_max = gesStaMaxPos + 17;
					output_prob = gesStaMaxProb;
				}
				else if (gesDyMaxProb > 0.7 && gesDyMaxProb >= gesStaMaxProb)
				{
					last_idx_max = gesDyMaxPos;
					output_prob = gesDyMaxProb;
				}
				else
				{
					last_idx_max = 0;
					output_prob = 0;
				}*/

				/* swiping judge */
				// if (last_idx_max == 11 || last_idx_max == 5 || last_idx_max == 6)  // gesture number 5
				if(last_idx_max == 7 || (GES_STA[gesStaMaxPos] == "number_6" && GES_DY[gesDyMaxPos] == "doing_other_things") || (GES_STA[gesStaMaxPos] == "thumb_up" && GES_DY[gesDyMaxPos] == "doing_other_things") )
				{
					swip_cnt += 1;
					cv::Point2f mov_coord(handCenterX, handCenterY);  // current bbox center coord
					if (!swip_mov_dist.empty())
					{
						cv::Point2f last_swip_coord = swip_mov_dist.back();  // last bbox center coord
						float x_mov = mov_coord.x - last_swip_coord.x;
						float y_mov = mov_coord.y - last_swip_coord.y;
					}
					swip_mov_dist.push_back(mov_coord);
				}
				else
				{
					miss_swip_cnt += 1;
				}
				// start swip judge
				if (swip_cnt > hyper_params.swip_cnt_thres)
				{
					cv::Point2f first_swip_dist = swip_mov_dist.front();
					cv::Point2f last_swip_dist = swip_mov_dist.back();
					float x_mov = last_swip_dist.x - first_swip_dist.x;
					float y_mov = last_swip_dist.y - first_swip_dist.y;
					int final_dir;
					float final_dist;
					if (abs(x_mov) > abs(y_mov))
					{
						if (x_mov > 0)
						{
							final_dir = 0;   // left
						}
						else
						{
							final_dir = 1;    // right
						}
						final_dist = abs(x_mov);
					}
					else
					{
						if (y_mov > 0)
						{
							final_dir = 2;   // down
						}
						else
						{
							final_dir = 3;   // up 
						}
						final_dist = abs(y_mov);
					}
					// moving towards a direction further than thresh
					if (final_dist > swip_dist_thresh)
					{
						if (final_dir == 0)
						{
							last_idx_max = 2;
						}
						else if (final_dir == 1)
						{
							last_idx_max = 3;
						}
						else if (final_dir == 2)
						{
							last_idx_max = 1;
						}
						else if (final_dir == 3)
						{
							last_idx_max = 4;
						}
						output_prob = gesStaMaxProb;
						// reset the params
						swip_mov_dist.clear();
						miss_swip_cnt = 0;
						swip_cnt = 0;
					}
					else {
						miss_swip_cnt = hyper_params.miss_swip_cnt_thres + 1;  // moving too little, jump to reset
					}
					// too many misses, reset the params
					if (miss_swip_cnt > hyper_params.miss_swip_cnt_thres) {
						swip_mov_dist.clear();
						miss_swip_cnt = 0;
						swip_cnt = 0;
					}
				}

			}
			cropHands.pop_front();
			cropHandsOri.pop_front();
		}
	}
	else
	{
		miss_hand_cnt++;
	}

	if (miss_hand_cnt >= hyper_params.miss_swip_cnt_thres)
	{
		last_idx_max = 0;
		miss_hand_cnt = 0;
		//output_prob = 0;
	}
	hand_region[6] = output_prob;

	
	//if (last_idx_max >= 17 && last_idx_max <= 24) 
	//{
	//	last_idx_max = last_idx_max - 8;
	//}
	//else if (last_idx_max == 25) // static Thumb_up idx -> dynamic Thumb_up idx 
	//{
	//	last_idx_max = 6;
	//}
	//else if (last_idx_max == 26) // static Thumb_Down idx -> dynamic Thumb_Down idx 
	//{
	//	last_idx_max = 5;
	//}
	// remove gesture number 5
	if (last_idx_max == 11)
	{
		last_idx_max = 0;
	}

	// voting if static gesture recognized
	if(last_idx_max >= 7 && last_idx_max <= 16)
	{
		if (hyper_params.vote_num == 1) {
			last_idx_max = last_idx_max;
		}
		else if (hyper_params.vote_num > 1) {
			int n_coincide = 0;
			for (int idx = 0; idx < hyper_params.vote_num - 1; idx++) {
				*(vote_sta+idx) = *(vote_sta + idx + 1);
				if (*(vote_sta + idx + 1) == last_idx_max) {
					n_coincide++;
				}
			}
			*(vote_sta + hyper_params.vote_num - 1) = last_idx_max;
			if (n_coincide >= hyper_params.hit_num - 1) {
				last_idx_max = last_idx_max;
			}
			else 
			{
				last_idx_max = 0;
			}
		}
	}

	if (last_idx_max == 17) // do not output "Fist"
	{
		last_idx_max = 0;
	}

	int predict_gesture = last_idx_max;
	if (hyper_params.mode == 0) {
		return predict_gesture;
	}
	else if (hyper_params.mode == 2) {  //dynamic gesture
		if (predict_gesture<=6) {
			return predict_gesture;
		}
		else {
			return 0;
		}
	}
	else if (hyper_params.mode == 1) {  //static gesture
		if (predict_gesture >= 7 && predict_gesture <= 16) {
			return predict_gesture;
		}
		else {
			return 0;
		}
	}
	
}