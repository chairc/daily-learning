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
	void Key(Mat& image);
	void ColorStyle(Mat& image);
	void Bitwise(Mat& image);
	void Channels(Mat& image);
	void Inrange(Mat& image);
	void PixelStatistic(Mat& image);
};