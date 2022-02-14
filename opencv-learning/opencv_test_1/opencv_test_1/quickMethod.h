#pragma once
#include <opencv2/opencv.hpp>
#include <Windows.h>

using namespace cv;

class QuickMethod {
private:

public:
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
	void Drawing(Mat& image);
	void RandomDrawing();
	void PolylineDrawing();
	void MouseDrawing(Mat& image);
	void Norm(Mat& image);
	void Resize(Mat& image);
	void Flip(Mat& image);
	void Rotate(Mat& image);
	void Histogram(Mat& image);
	void Histogram2D(Mat& image);
};