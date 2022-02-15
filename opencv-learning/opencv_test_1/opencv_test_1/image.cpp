#include "image.h"
#include "quickMethod.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace std;

static void ImageCommonMethod(enum MethodEnum me,int image_num = 0) {
	try {
		Mat image_mat;
		QuickMethod qd;
		if (image_num == 0) {
			// 三色低像素图片15x20
			image_mat = imread("C:/Users/lenovo/Desktop/1.jpg");
		} else if (image_num == 1) {
			// 高像素图片
			image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		} else if (image_num == 2) {
			image_mat = imread("C:/Users/lenovo/Desktop/test1.jpg");
			// 灰度图
		} else if (image_num == 3) {
			image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg", IMREAD_GRAYSCALE);
		}
		
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		} else {
			namedWindow("打开图片", WINDOW_FREERATIO);
			imshow("打开图片", image_mat);
		}
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
			// 几何图像绘制
			case ImageDrawing:
				qd.Drawing(image_mat);
				break;
			// 图像随机绘制
			case ImageRandomDrawing:
				qd.RandomDrawing();
				break;
			// 多边形绘制
			case ImagePolylineDrawing:
				qd.PolylineDrawing();
				break;
			// 鼠标绘制
			case ImageMouseDrawing:
				qd.MouseDrawing(image_mat);
				break;
			// 归一化
			case ImageNorm:
				qd.Norm(image_mat);
				break;
			// 重设大小
			case ImageResize:
				qd.Resize(image_mat);
				break;
			// 图片翻转
			case ImageFlip:
				qd.Flip(image_mat);
				break;
			// 图片旋转
			case ImageRotate:
				qd.Rotate(image_mat);
				break;
			// 直方图
			case ImageHistogram:
				qd.Histogram(image_mat);
				break;
			// 2D直方图
			case ImageHistogram2D:
				qd.Histogram2D(image_mat);
				break;
			// 直方图均值化
			case ImageHistogramEqual:
				qd.HistogramEqual(image_mat);
				break;
			// 图像模糊
			case ImageBlur:
				qd.Blur(image_mat);
				break;
			// 高斯模糊
			case ImageGaussianBlur:
				qd.GaussianBlur(image_mat);
				break;
			// 照片人脸检测
			case ImageFaceDetection:
				qd.FaceDetection(image_mat, qd.LoadNet());
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
	ImageCommonMethod(MethodEnum(0), 0);
}

void Image::ImageMatrixCreation() {
	ImageCommonMethod(MethodEnum(1), 0);
}

void Image::ImagePixelVisit() {
	ImageCommonMethod(MethodEnum(2), 0);
}

void Image::ImageOperator() {
	ImageCommonMethod(MethodEnum(3), 0);
}

void Image::ImageTrackingBar() {
	ImageCommonMethod(MethodEnum(4), 0);
}


void Image::ImageKey() {
	ImageCommonMethod(MethodEnum(5), 0);
}

void Image::ImageColorStyle() {
	ImageCommonMethod(MethodEnum(6), 0);
}

void Image::ImageBitwise() {
	ImageCommonMethod(MethodEnum(7), 0);
}

void Image::ImageChannels() {
	ImageCommonMethod(MethodEnum(8), 0);
}

void Image::ImageInRange() {
	ImageCommonMethod(MethodEnum(9), 0);
}

void Image::ImagePixelStatistic() {
	ImageCommonMethod(MethodEnum(10), 0);
}

void Image::ImageDrawing() {
	ImageCommonMethod(MethodEnum(11), 1);
}

void Image::ImageRandomDrawing() {
	ImageCommonMethod(MethodEnum(12), -1);
}

void Image::ImagePolylineDrawing() {
	ImageCommonMethod(MethodEnum(13), -1);
}

void Image::ImageMouseDrawing() {
	ImageCommonMethod(MethodEnum(14), 1);
}

void Image::ImageNorm() {
	ImageCommonMethod(MethodEnum(15), 1);
}

void Image::ImageResize() {
	ImageCommonMethod(MethodEnum(16), 2);
}

void Image::ImageFlip() {
	ImageCommonMethod(MethodEnum(17), 1);
}

void Image::ImageRotate() {
	ImageCommonMethod(MethodEnum(18), 1);
}

void Image::ImageHistogram() {
	ImageCommonMethod(MethodEnum(19), 1);
}

void Image::ImageHistogram2D() {
	ImageCommonMethod(MethodEnum(20), 1);
}

void Image::ImageHistogramEqual() {
	ImageCommonMethod(MethodEnum(21), 3);
}

void Image::ImageBlur() {
	ImageCommonMethod(MethodEnum(22), 1);
}

void Image::ImageGaussianBlur() {
	ImageCommonMethod(MethodEnum(23), 1);
}

void Image::ImageFaceDetection() {
}

