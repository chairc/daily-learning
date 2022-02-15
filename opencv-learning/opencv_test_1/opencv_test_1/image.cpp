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
			// ��ɫ������ͼƬ15x20
			image_mat = imread("C:/Users/lenovo/Desktop/1.jpg");
		} else if (image_num == 1) {
			// ������ͼƬ
			image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg"); // B,G,R
		} else if (image_num == 2) {
			image_mat = imread("C:/Users/lenovo/Desktop/test1.jpg");
			// �Ҷ�ͼ
		} else if (image_num == 3) {
			image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg", IMREAD_GRAYSCALE);
		}
		
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		} else {
			namedWindow("��ͼƬ", WINDOW_FREERATIO);
			imshow("��ͼƬ", image_mat);
		}
		switch (me) {
			// ɫ�ʿռ�ת��
			case ImageColorSpace:
				qd.ColorSpace(image_mat);
				break;
			// ��������
			case ImageMatrixCreation:
				// qd.MatrixCreation(image_mat);
				qd.MatrixCreation();
				break;
			// ���ط���
			case ImagePixelVisit:
				qd.PixelVisit(image_mat);
				break;
			// ������
			case ImageOperator:
				qd.Operators(image_mat);
				break;
			// ������������ʾ
			case ImageTrackingBar:
				qd.Operators(image_mat);
				break;
			// ���̽�����ʾ
			case ImageKey:
				qd.Key(image_mat);
				break;
			// ͼƬ��ɫ��ʽ��ʾ
			case ImageColorStyle:
				qd.ColorStyle(image_mat);
				break;
			// ͼƬ�����߼���ϵ
			case ImageBitwise:
				qd.Bitwise(image_mat);
				break;
			// ͨ��������ϲ�
			case ImageChannels:
				qd.Channels(image_mat);
				break;
			// ͼ��ɫ�ʿռ�ת��
			case ImageInRange:
				qd.Inrange(image_mat);
				break;
			// ͼ������ֵͳ��
			case ImagePixelStatistic:
				qd.PixelStatistic(image_mat);
				break;
			// ����ͼ�����
			case ImageDrawing:
				qd.Drawing(image_mat);
				break;
			// ͼ���������
			case ImageRandomDrawing:
				qd.RandomDrawing();
				break;
			// ����λ���
			case ImagePolylineDrawing:
				qd.PolylineDrawing();
				break;
			// ������
			case ImageMouseDrawing:
				qd.MouseDrawing(image_mat);
				break;
			// ��һ��
			case ImageNorm:
				qd.Norm(image_mat);
				break;
			// �����С
			case ImageResize:
				qd.Resize(image_mat);
				break;
			// ͼƬ��ת
			case ImageFlip:
				qd.Flip(image_mat);
				break;
			// ͼƬ��ת
			case ImageRotate:
				qd.Rotate(image_mat);
				break;
			// ֱ��ͼ
			case ImageHistogram:
				qd.Histogram(image_mat);
				break;
			// 2Dֱ��ͼ
			case ImageHistogram2D:
				qd.Histogram2D(image_mat);
				break;
			// ֱ��ͼ��ֵ��
			case ImageHistogramEqual:
				qd.HistogramEqual(image_mat);
				break;
			// ͼ��ģ��
			case ImageBlur:
				qd.Blur(image_mat);
				break;
			// ��˹ģ��
			case ImageGaussianBlur:
				qd.GaussianBlur(image_mat);
				break;
			// ��Ƭ�������
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
			// ��ɫͼ��
			image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg");
		} else if (image_type == 1) {
			// �Ҷ�ͼ��
			image_mat = imread("C:/Users/lenovo/Desktop/head_index.jpg", IMREAD_GRAYSCALE);
		}
		if (image_mat.empty()) {
			cout << "image is empty..." << endl;
		}
		namedWindow("��ͼƬ", WINDOW_FREERATIO);
		imshow("��ͼƬ", image_mat);
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

