#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

class QuickMethod {
private:

public:
	// ɫ�ʿռ�ת��
	void ColorSpace(Mat& image);
	void MatrixCreation(Mat& image);
	void MatrixCreation();
	void PixelVisit(Mat& image);
	void Operators(Mat& image);
	void TrackingBar(Mat& image);
};