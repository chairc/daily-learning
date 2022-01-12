#include "image.h"
#include "quickMethod.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void Image::OpenImage(int image_type) {
	try {
		Mat image_mat;
		if (image_type == 0) {
			// 彩色图像
			image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg");
		} else if (image_type == 1) {
			// 灰度图像
			image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg", IMREAD_GRAYSCALE);
		}
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		waitKey(0);
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Image::ImageColorSpace() {
	try {
		Mat image_mat;
		QuickMethod qd;
		image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		// 色彩空间转换
		qd.ColorSpace(image_mat);
		waitKey(0);
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Image::ImageMatrixCreation() {
	try {
		Mat image_mat;
		QuickMethod qd;
		image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		// 色彩空间转换
		// qd.MatrixCreation(image_mat);
		qd.MatrixCreation();
		waitKey(0);
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Image::ImagePixelVisit() {
	try {
		Mat image_mat;
		QuickMethod qd;
		image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		qd.PixelVisit(image_mat);
		waitKey(0);
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Image::ImageOperator() {
	try {
		Mat image_mat;
		QuickMethod qd;
		image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		qd.Operators(image_mat);
		waitKey(0);
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Image::ImageTrackingBar() {
	try {
		Mat image_mat;
		QuickMethod qd;
		image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		qd.TrackingBar(image_mat);
		waitKey(0);
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}


void Image::ImageKey() {
	try {
		Mat image_mat;
		QuickMethod qd;
		image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		qd.Key(image_mat);
		waitKey(0);
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Image::ImageColorStyle() {
	try {
		Mat image_mat;
		QuickMethod qd;
		image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		qd.ColorStyle(image_mat);
		waitKey(0);
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Image::ImageBitwise() {
	try {
		Mat image_mat;
		QuickMethod qd;
		image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		qd.Bitwise(image_mat);
		waitKey(0);
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

