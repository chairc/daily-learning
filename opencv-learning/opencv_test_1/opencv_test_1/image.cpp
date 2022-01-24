#include "image.h"
#include "quickMethod.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

static void ImageCommonMethod(enum MethodEnum me) {
	try {
		Mat image_mat;
		QuickMethod qd;
		//image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		image_mat = imread("C:/Users/lenovo/Desktop/1.jpg");
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("打开图片", WINDOW_FREERATIO);
		imshow("打开图片", image_mat);
		switch (me) {
			// 色彩空间转换
			case ImageColorSpace:
				qd.ColorSpace(image_mat);
				break;
			// 矩阵生成
			case ImageMatrixCreation:
				// qd.MatrixCreation(image_mat);
				qd.MatrixCreation();
				break;
			// 像素访问
			case ImagePixelVisit:
				qd.PixelVisit(image_mat);
				break;
			// 操作数
			case ImageOperator:
				qd.Operators(image_mat);
				break;
			// 滚动条亮度显示
			case ImageTrackingBar:
				qd.Operators(image_mat);
				break;
			// 键盘交互显示
			case ImageKey:
				qd.Key(image_mat);
				break;
			// 图片颜色样式显示
			case ImageColorStyle:
				qd.ColorStyle(image_mat);
				break;
			// 图片像素逻辑关系
			case ImageBitwise:
				qd.Bitwise(image_mat);
				break;
			// 通道分离与合并
			case ImageChannels:
				qd.Channels(image_mat);
				break;
			// 图像色彩空间转换
			case ImageInRange:
				qd.Inrange(image_mat);
				break;
			// 图像像素值统计
			case ImagePixelStatistic:
				qd.PixelStatistic(image_mat);
				break;

			default:
				break;
		}
		waitKey(0);
	} catch (const std::exception& e) {
		cout << e.what() << endl;
	}
}

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
	ImageCommonMethod(MethodEnum(0));
}

void Image::ImageMatrixCreation() {
	ImageCommonMethod(MethodEnum(1));
}

void Image::ImagePixelVisit() {
	ImageCommonMethod(MethodEnum(2));
}

void Image::ImageOperator() {
	ImageCommonMethod(MethodEnum(3));
}

void Image::ImageTrackingBar() {
	ImageCommonMethod(MethodEnum(4));
}


void Image::ImageKey() {
	ImageCommonMethod(MethodEnum(5));
}

void Image::ImageColorStyle() {
	ImageCommonMethod(MethodEnum(6));
}

void Image::ImageBitwise() {
	ImageCommonMethod(MethodEnum(7));
}

void Image::ImageChannels() {
	ImageCommonMethod(MethodEnum(8));
}

void Image::ImageInRange() {
	ImageCommonMethod(MethodEnum(9));
}

void Image::ImagePixelStatistic() {
	ImageCommonMethod(MethodEnum(10));
}

