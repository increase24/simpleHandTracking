#include "pch.h"
#include "handGesRec.h"
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

void* __stdcall handLandmarks_Init(const char* p_palmDetModel, const char* p_handKptsModel, const char* p_anchorFile)
{
	//cv::Mat padding = load_gestureImg(gesture_classes);
	// file vars
	string anchorFile = p_anchorFile;
	string palmModel = p_palmDetModel;
	string handModel = p_handKptsModel;
	//string staGesModel = p_staGesRecModel;
	//string gesFeatModel = p_handFeatModel;
	//string gesModel = p_handGesRecModel;

	Ort::Session* sessionPalm = sessionInit(palmModel, 4444, "Adasd@#4s24!3da");
	Ort::Session* sessionHand = sessionInit(handModel, 7777, "CR2NX8MVxe0hz&0*");
	/*Ort::Session* sessionFeat = sessionInit(gesFeatModel, 7777, "z0hrajmckgbhpkj2*");
	Ort::Session* sessionGes = sessionInit(gesModel, 6666, "Z0hRAJmCkGBHpkJ2*");*/

	/* ---- load anchor binary file ---- */
	cv::Mat_<float> anchors_cvMat = cv::Mat::zeros(numAnchors, 4, CV_32FC1);
	fstream fin_cvMat(anchorFile, ios::in | ios::binary);
	fin_cvMat.read((char *)anchors_cvMat.data, anchors_cvMat.cols* anchors_cvMat.rows * sizeof(float));
	//fstream fin(anchorFile, ios::in | ios::binary);
	//fin.read((char *)anchors.data_ptr(), anchors.numel() * sizeof(float));

	fstream fout("D:\\debug.txt", ios::out | ios::binary);
	fout << "Successfully initialize the models!" << endl;

	/* ---- init ONNX rt ---- */
	////Ort::AllocatorWithDefaultOptions allocator;
	//std::vector<int64_t> palm_input_node_dims = { batchSizePalm, 3, modelHeight, modelWidth };
	//size_t palm_input_tensor_size = batchSizePalm * 3 * modelHeight * modelWidth;
	//std::vector<int64_t> hand_input_node_dims = { batchSizeHand, 3, modelHeight, modelWidth };
	//size_t hand_input_tensor_size = batchSizeHand * 3 * modelHeight * modelWidth;
	//std::vector<float> input_tensor_values(palm_input_tensor_size);
	//std::vector<const char *> input_node_names = { "input" };
	//std::vector<const char *> output_node_names = { "output1", "output2" };


	handLandmarks* handLdks = new handLandmarks();
	handLdks->sessions.push_back(sessionPalm);
	handLdks->sessions.push_back(sessionHand);
	handLdks->anchors = anchors_cvMat;

	return handLdks;
}

int __stdcall handLandmarks_inference(void* p_self, void* image, int* image_shape, float* hand_regions, bool debug_print)
{
	static handLandmarks* handLdks = (handLandmarks*)(p_self);
	static Ort::Session* sess_palmDet = handLdks->sessions[0];
	static Ort::Session* sess_handKpts = handLdks->sessions[1];
	//static Ort::Session* sess_handFeat = handLdks->sessions[2];
	//static Ort::Session* sess_handGesRec = handLdks->sessions[3];

	unsigned char* _input = (unsigned char*)(image);
	int img_h = image_shape[0];
	int img_w = image_shape[1];
	// convert unsigned char* to cv::Mat
	cv::Mat rawFrame_rgba(img_h, img_w, CV_8UC4, _input);
	cv::Mat rawFrame;
	cv::cvtColor(rawFrame_rgba, rawFrame, cv::COLOR_BGRA2BGR);
	if (debug_print)
	{
		fstream fout("D:\\debug.txt", ios::app);
		fout << "Successfully decode input image!" << endl;
	}
	//static configuration
	static Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	//static int modelWidth = 256, modelHeight = 256, modelWidth_GesRec = 224, modelHeight_GesRec = 224;
	static size_t palm_input_tensor_size = batchSizePalm * 3 * modelHeight * modelWidth;
	static size_t hand_input_tensor_size = 1 * 3 * modelHeight_handKpts * modelWidth_handKpts;
	static std::vector<int64_t> palm_input_node_dims = { batchSizePalm, 3, modelHeight, modelWidth };
	static std::vector<int64_t> hand_input_node_dims = { 1, modelHeight_handKpts, modelWidth_handKpts, 3};
	static std::vector<const char *> input_node_names = { "input" };
	static std::vector<const char *> output_node_names = { "output1", "output2" };
	static std::vector<const char *> input_node_name_handKpts = { "input_1" };
	static std::vector<const char *> output_node_name_handKpts = { "Identity", "Identity_1", "Identity_2" };

	static deque<cv::Point2f> swip_mov_dist;
	static int counter_frame = 0;
	static int counter_frame_sta = 0;
	int last_idx_max = 0;
	static int miss_hand_cnt = 0;
	static int miss_swip_cnt = 0;
	static int swip_cnt = 0;
	float output_prob = 0.0f;
	// voter for static gesture recognition
	// static int* vote_sta = new int[hyper_params.vote_num];

	// queue of palm bbox and hand region
	static deque<detRect> cropHands, cropHandsOri;
	static deque<detMeta> handMetaForwardOri;
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

	// float swip_dist_thresh = cropHeight * hyper_params.swip_dist_thres;
	int showHeightOri = cropHeight, showWidthOri = cropWidth;
	cropFrame.copyTo(showFrameOri);
	
	//cv::imwrite("D:\\show_frame.jpg", showFrameOri);
	if (debug_print)
	{
		fstream fout("D:\\debug.txt", ios::app);
		fout << "Successfully crop image!" << endl;
	}

	BoxInfo bbox[2];
	float score_bbox[2];
	/* --------------------------------------- perform palm detection ------------------------------------- */
	if (handMetaForwardOri.empty())
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
			fstream fout("D:\\debug.txt", ios::app);
			fout << "hand detection time cost:" << infer_time1 << "ms" << endl;
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
			// select the bboxs with the highest and the second highest confidence
			int opt_idx = 0, opt_idx_2 = 0;
			float max_score = 0.0f, max_score_2 = 0.0f;
			for (int idx = 0; idx < detectionBoxes.size(); idx++) {
				float score = detectionBoxes[idx].score;
				if (score > max_score && score > handThrs) {
					max_score = score;
					opt_idx = idx;
				}
				else if (score > max_score_2 && score > handThrs) {
					max_score_2 = score;
					opt_idx_2 = idx;
				}
			}
			int opt_idxs[2] = { opt_idx ,opt_idx_2 };
			float max_scores[2] = { max_score , max_score_2 };

			for (int i_bbox = 0; i_bbox<2;i_bbox++)
			{
				int _idx = opt_idxs[i_bbox];
				float _score = max_scores[i_bbox];

				detectionBox = detectionBoxes[i_bbox];
				outputBox = detectionBoxes[i_bbox];

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

				/*hand_regions[0] = detectionBox.x1;
				hand_regions[1] = detectionBox.y1;
				hand_regions[2] = detectionBox.x2;
				hand_regions[3] = detectionBox.y2;
				hand_regions[4] = max_score;*/
				bbox[i_bbox] = detectionBox;
				score_bbox[i_bbox] = _score;

				/*cv::rectangle(showFrameOri, cv::Point(detectionBox.x1 * showFrameOri.cols, detectionBox.y1  * showFrameOri.rows),
					cv::Point(detectionBox.x2 * showFrameOri.cols, detectionBox.y2 * showFrameOri.rows), cv::Scalar(0, 255, 0), 1, 1, 0);*/
			
				cv::Point2f handUp = cv::Point2f(kpts.at<float>(0, palmUpId * 2), kpts.at<float>(0, palmUpId * 2 + 1)),
					handDown = cv::Point2f(kpts.at<float>(0, palmDownId * 2), kpts.at<float>(0, palmDownId * 2 + 1));

				if (max_score > handThrs)
				{
					//handMetaForward.push_back(detMeta(xmin, ymin, xmax, ymax, handUp, handDown, 0));
					handMetaForwardOri.push_back(detMeta(xminOri, yminOri, xmaxOri, ymaxOri, handUp, handDown, 0));
					//cv::rectangle(showFrame, cv::Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)), cv::Scalar(0, 0, 255), 1, 1, 0);
				}
			}
		}
	}

	while (!handMetaForwardOri.empty())
	{
		//cropHands.push_back(handMetaForward.front().getTransformedRect(resizeFrame));
		cropHandsOri.push_back(handMetaForwardOri.front().getTransformedRect(showFrameOri));
		//cv::circle(showFrame, cropHands.front().rawCenter, 2, { 255, 0, 0 }, 2);
		//handMetaForward.pop_front();
		handMetaForwardOri.pop_front();
	}
	//
	/* ----------------- Hand Keypoint Detection NN Inference ---------------------- */
	int batchSizeHand = cropHandsOri.size();
	if (batchSizeHand)
	{
		for (int i_hand=0; i_hand< batchSizeHand; i_hand++)
		{
			auto cropHand = cropHandsOri.front();
			cv::Mat cropImage_Affine;
			resize(cropHand.img, cropImage_Affine, cv::Size(modelWidth_handKpts, modelHeight_handKpts), 0, 0, cv::INTER_LINEAR);
			cv::cvtColor(cropImage_Affine, cropImage_Affine, cv::COLOR_BGR2RGB);
			cropImage_Affine.convertTo(cropImage_Affine, CV_32F);
			cropImage_Affine = cropImage_Affine / 127.5 - 1.0;
			//cv::Mat cropFrameCHW = mat2chw(cropImage_Affine);
			cv::Mat cropFrameCHW = cropImage_Affine;
			Ort::Value input_ort = Ort::Value::CreateTensor<float>(memory_info, (float_t *)cropFrameCHW.data, hand_input_tensor_size, hand_input_node_dims.data(), 4);
			auto output_tensors = sess_handKpts->Run(Ort::RunOptions(nullptr), input_node_name_handKpts.data(), &input_ort, 1, output_node_name_handKpts.data(), 3);
			//float* rawBoxesPPtr = output_tensors[0].GetTensorMutableData<float>(); // bounding box
			float* score_hand = output_tensors[1].GetTensorMutableData<float>(); // confidence
			float* handness = output_tensors[2].GetTensorMutableData<float>(); // left hand->0; right hand->1
			//cout << "score_hand:" << score_hand[0] << endl;
			if (handness[0] > 0.9&&score_hand[0]>0.9) //right hand
			{
				hand_regions[5] = bbox[i_hand].x1;
				hand_regions[6] = bbox[i_hand].y1;
				hand_regions[7] = bbox[i_hand].x2;
				hand_regions[8] = bbox[i_hand].y2;
				hand_regions[9] = score_bbox[i_hand];
			}
			else if (handness[0] < 0.1 && score_hand[0]>0.9) //left hand
			{
				hand_regions[0] = bbox[i_hand].x1;
				hand_regions[1] = bbox[i_hand].y1;
				hand_regions[2] = bbox[i_hand].x2;
				hand_regions[3] = bbox[i_hand].y2;
				hand_regions[4] = score_bbox[i_hand];
			}
			//if (i_hand == 0)
			//{
			//	hand_regions[5] = bbox[i_hand].x1;
			//	hand_regions[6] = bbox[i_hand].y1;
			//	hand_regions[7] = bbox[i_hand].x2;
			//	hand_regions[8] = bbox[i_hand].y2;
			//	hand_regions[9] = score_bbox[i_hand];
			//}
			//else
			//{
			//	hand_regions[0] = bbox[i_hand].x1;
			//	hand_regions[1] = bbox[i_hand].y1;
			//	hand_regions[2] = bbox[i_hand].x2;
			//	hand_regions[3] = bbox[i_hand].y2;
			//	hand_regions[4] = score_bbox[i_hand];
			//}
			cropHandsOri.pop_front();
		}
	}
	return 0;
}