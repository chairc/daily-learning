#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
class yolox {
public:
	yolox(string modelpath, float confThreshold, float nmsThreshold);
	void detect(Mat& srcimg);

private:
	const int stride[3] = { 8, 16, 32 };
	const string classesFile = "coco.names";   ////����Ǵ��COCO���ݼ�������������������Լ����ݼ�ѵ���ģ���ô��Ҫ�޸�
	const int input_shape[2] = { 640, 640 };   //// height, width
	const float mean[3] = { 0.485, 0.456, 0.406 };
	const float std[3] = { 0.229, 0.224, 0.225 };
	float prob_threshold;
	float nms_threshold;
	vector<string> classes;
	int num_class;
	Net net;

	Mat resize_image(Mat srcimg, float* scale);
	void normalize(Mat& srcimg);
	int get_max_class(float* scores);
};
