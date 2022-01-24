#include "quickMethod.h"
#include <iostream>
#include <vector>

using namespace std;

void QuickMethod::ColorSpace(Mat& image) {
	// ����ɫͼƬת��Ϊ�Ҷ�ͼ��HSVͼ
	Mat gray, hsv;
	// ȡֵ��Χ H��0 ~ 180 S��0 ~ 255 V��0 ~ 255 S��H������ɫ��V��������
	cvtColor(image, hsv, COLOR_BGR2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	namedWindow("HSV", WINDOW_FREERATIO);
	namedWindow("GRAY", WINDOW_FREERATIO);
	imshow("HSV", hsv);
	imshow("GRAY", gray);
	// ����ͼƬ
	imwrite("F:/program/opencv-learning/opencv_test_1_source_file/HSV.png", hsv);
	imwrite("F:/program/opencv-learning/opencv_test_1_source_file/GRAY.png", gray);
}

void QuickMethod::MatrixCreation(Mat& image) {
	Mat mat_clone, mat_copy;
	// ��¡
	mat_clone = image.clone();

	// ����
	image.copyTo(mat_copy);

	// �����հ�Matrix��CV_8UC1��ʾ8λ�޷���1ͨ��
	Mat mat_zero = Mat::zeros(Size(8, 8), CV_8UC1);
	cout << "mat_zero matrix: " << mat_zero << endl;
}

void QuickMethod::MatrixCreation() {


	// �����հ�Matrix��CV_8UC1��ʾ8λ�޷���3ͨ��
	Mat mat_custom = Mat::zeros(Size(400, 400), CV_8UC3);
	// ��3ͨ������ָ������ֵ
	mat_custom = Scalar(0, 255, 0);
	// cout << "mat_custom matrix: " << mat_custom << endl;
	imshow("�Զ��帳ֵͼ��", mat_custom);

	Mat mat_clone = mat_custom.clone();
	mat_clone = Scalar(255, 0, 0);
	imshow("mat_cloneͼ��", mat_clone);

	Mat mat_copy;
	mat_custom.copyTo(mat_copy);
	mat_copy = Scalar(170, 170, 170);
	imshow("mat_copyͼ��", mat_custom);

	Mat mat_custom2 = mat_custom;
	mat_custom2 = Scalar(0, 0, 255);
	imshow("mat_customͼ��", mat_custom);
	imshow("mat_custom2ͼ��", mat_custom2);
}

void QuickMethod::PixelVisit(Mat& image) {
	int width = image.cols;
	int height = image.rows;
	int channels = image.channels();
	cout << "width: " << width << " height: " << height << " channels: " << channels << endl;
	// ���鷽��
	/*for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (channels == 1) {	// �Ҷ�ͼ��
				int pv = image.at<uchar>(row, col);
				image.at<uchar>(row, col) = 255 - pv;
			} else if (channels == 3) {	// ��ɫͼ��
				Vec3b bgr = image.at<Vec3b>(row, col);
				image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
			}
		}
	}*/

	// ָ�뷽��
	for (int row = 0; row < height; row++) {
		uchar* current_row = image.ptr<uchar>(row);
		for (int col = 0; col < width; col++) {
			if (channels == 1) {	// �Ҷ�ͼ��
				int pv = *current_row;
				*current_row++ = 255 - pv;
			} else if (channels == 3) {	// ��ɫͼ��
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}
		}
	}
	namedWindow("���ض�д", WINDOW_FREERATIO);
	imshow("���ض�д", image);
}

void QuickMethod::Operators(Mat& image) {
	// �����Ӽ��˳���
	Mat dst_1, dst_2, dst_3, dst_4;
	dst_1 = image + Scalar(20, 20, 20);
	namedWindow("�ӷ�����", WINDOW_FREERATIO);
	imshow("�ӷ�����", dst_1);

	dst_2 = image - Scalar(50, 50, 50);
	namedWindow("��������", WINDOW_FREERATIO);
	imshow("��������", dst_2);

	dst_3 = image / Scalar(2, 2, 2);
	namedWindow("��������", WINDOW_FREERATIO);
	imshow("��������", dst_3);

	Mat dst_temp = Mat::zeros(image.size(), image.type());
	dst_temp = Scalar(2, 2, 2);
	multiply(image, dst_temp, dst_4);
	namedWindow("�˷�����", WINDOW_FREERATIO);
	imshow("�˷�����", dst_4);

	// ���鷽���Ӽ��˳�
	/*int width = image.cols;
	int height = image.rows;
	int channels = image.channels();
	Mat dst = Mat::zeros(image.size(), image.type());
	dst = Scalar(2, 2, 2);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			Vec3b bgr1 = image.at<Vec3b>(row, col);
			Vec3b bgr2 = dst.at<Vec3b>(row, col);
			// saturate_cast�ж��Ƿ���0~255֮��
			dst_1.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(bgr1[0] + bgr2[0]);
			dst_1.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(bgr1[1] + bgr2[1]);
			dst_1.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(bgr1[2] + bgr2[2]);
		}
	}*/

	// OpenCV�ļӼ��˳�API
	Mat dst_result = Mat::zeros(image.size(), image.type());
	Mat mat = Mat::zeros(image.size(), image.type());
	mat = Scalar(20, 20, 20);
	add(image, mat, dst_result);
	subtract(image, mat, dst_result);
	multiply(image, mat, dst_result);
	divide(image, mat, dst_result);
}

// �������ص�����
static void OnLightness(int lightness, void* user_data) {
	Mat image = *((Mat*)user_data);
	Mat dst_on_track = Mat::zeros(image.size(), image.type());
	Mat mat_on_track = Mat::zeros(image.size(), image.type());
	addWeighted(image, 1, mat_on_track, 0, lightness, dst_on_track);
	imshow("������Աȶȵ���", dst_on_track);
}

static void OnContrast(int contrast, void* user_data) {
	Mat image = *((Mat*)user_data);
	Mat dst_on_track = Mat::zeros(image.size(), image.type());
	Mat mat_on_track = Mat::zeros(image.size(), image.type());
	double temp = contrast / 200.0;
	addWeighted(image, temp, mat_on_track, 0.0, 0.0, dst_on_track);
	imshow("������Աȶȵ���", dst_on_track);
}

void QuickMethod::TrackingBar(Mat& image) {
	namedWindow("������Աȶȵ���", WINDOW_FREERATIO);
	int light_max_value = 100, lightness = 50, contrast_value = 100;
	createTrackbar("Value Bar", "������Աȶȵ���", &lightness, light_max_value, OnLightness, (void*)(&image));
	createTrackbar("Contrast Bar", "������Աȶȵ���", &contrast_value, 200, OnContrast, (void*)(&image));
	OnLightness(50, &image);
}

void QuickMethod::Key(Mat& image) {
	Mat dst = Mat::zeros(image.size(), image.type());
	namedWindow("������Ӧ", WINDOW_FREERATIO);
	// ����ѭ�����������̲���
	while (true) {
		int c = waitKey(100);
		// cout << "ascii: " << c << endl;
		// ��ESC�˳�
		if (c == 27) {
			break;
		}
		if (c == 49) {
			cout << "this key is: 1" << endl;
			cvtColor(image, dst, COLOR_BGR2GRAY);
		}
		if (c == 50) {
			cout << "this key is: 2" << endl;
			cvtColor(image, dst, COLOR_BGR2HSV);
		}
		if (c == 51) {
			cout << "this key is: 3" << endl;
			dst = Scalar(50, 50, 50);
			add(image, dst, dst);
		}
		imshow("������Ӧ", dst);
	}
}

void QuickMethod::ColorStyle(Mat& image) {
	int color_map[] = {
		COLORMAP_AUTUMN,
		COLORMAP_BONE,
		COLORMAP_CIVIDIS,
		COLORMAP_COOL,
		COLORMAP_DEEPGREEN,
		COLORMAP_HOT,
		COLORMAP_HSV,
		COLORMAP_INFERNO,
		COLORMAP_JET,
		COLORMAP_MAGMA,
		COLORMAP_OCEAN,
		COLORMAP_PARULA,
		COLORMAP_PINK,
		COLORMAP_PLASMA,
		COLORMAP_RAINBOW,
		COLORMAP_SPRING,
		COLORMAP_SUMMER,
		COLORMAP_TURBO,
		COLORMAP_TWILIGHT,
		COLORMAP_TWILIGHT_SHIFTED,
		COLORMAP_VIRIDIS,
		COLORMAP_WINTER
	};

	Mat dst;
	int i = 0;
	while (true) {
		int c = waitKey(1000);
		// ��ESC�˳�
		if (c == 27) {
			break;
		}
		applyColorMap(image, dst, color_map[i]);
		if (i == 21) {
			i = 0;
		} else {
			i++;
		}
		namedWindow("ͼƬ��ɫ��ʽѭ��", WINDOW_FREERATIO);
		imshow("ͼƬ��ɫ��ʽѭ��", dst);
	}
}

void QuickMethod::Bitwise(Mat& image) {
	Mat dst_1, dst_2, dst_3, dst_4;
	// ����������ͨ������
	Mat mat_1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_2 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_3 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_4 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_5 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_6 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_7 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_8 = Mat::zeros(Size(256, 256), CV_8UC3);
	// ��mat_1��������x,y�ʹ�С����ɫ��thickness�߿�-1��ʾ�����Σ�����0��ʾ���ƣ�
	rectangle(mat_1, Rect(100, 100, 50, 50), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(mat_2, Rect(130, 130, 50, 50), Scalar(0, 255, 255), -1, LINE_8, 0);
	rectangle(mat_3, Rect(100, 100, 50, 50), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(mat_4, Rect(130, 130, 50, 50), Scalar(0, 255, 255), -1, LINE_8, 0);
	rectangle(mat_5, Rect(100, 100, 50, 50), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(mat_6, Rect(130, 130, 50, 50), Scalar(0, 255, 255), -1, LINE_8, 0);
	// imshow("��ʾmat_1", mat_1);
	// imshow("��ʾmat_2", mat_2);

	// λ�����
	bitwise_and(mat_1, mat_2, dst_1);
	imshow("����λ������and��", dst_1);

	// λ�����
	bitwise_or(mat_3, mat_4, dst_2);
	imshow("����λ������or��", dst_2);

	// λ�ǲ���
	bitwise_not(image, dst_3);
	// ~imageȡ��
	// dst_3 = ~image;
	namedWindow("����λ������not��", WINDOW_FREERATIO);
	imshow("����λ������not��", dst_3);

	// λ������
	bitwise_xor(mat_5, mat_6, dst_4);
	imshow("����λ������xor��", dst_4);

}

void QuickMethod::Channels(Mat& image) {
	
	vector<Mat> mat_vector;
	split(image, mat_vector);
	namedWindow("ͨ����ʾ-��ɫ", WINDOW_FREERATIO);
	namedWindow("ͨ����ʾ-��ɫ", WINDOW_FREERATIO);
	namedWindow("ͨ����ʾ-��ɫ", WINDOW_FREERATIO);
	namedWindow("����ͨ����ʾ-��ɫ", WINDOW_FREERATIO);
	namedWindow("����ͨ����ʾ-��ɫ", WINDOW_FREERATIO);
	namedWindow("����ͨ����ʾ-��ɫ", WINDOW_FREERATIO);
	// ��ʾ��ͨ����ûһ��ͨ��
	imshow("ͨ����ʾ-��ɫ", mat_vector[0]);
	imshow("ͨ����ʾ-��ɫ", mat_vector[1]);
	imshow("ͨ����ʾ-��ɫ", mat_vector[2]);

	Mat dst_r, dst_g, dst_b, dst = Mat::zeros(image.size(), image.type());

	mat_vector[1] = 0;
	mat_vector[2] = 0;
	merge(mat_vector, dst_b);
	imshow("����ͨ����ʾ-��ɫ", dst_b);

	// �ϲ�֮����Ҫ���²��ԭʼͼ�񣬲�Ȼ������һ���ϲ���ͼƬ�����ջᵼ��ͨ��ȫΪ0
	split(image, mat_vector);
	mat_vector[0] = 0;
	mat_vector[2] = 0;
	merge(mat_vector, dst_g);
	imshow("����ͨ����ʾ-��ɫ", dst_g);

	split(image, mat_vector);
	mat_vector[0] = 0;
	mat_vector[1] = 0;
	merge(mat_vector, dst_r);
	imshow("����ͨ����ʾ-��ɫ", dst_r);

	// ���ͨ��
	int from_to[] = { 0,2,1,1,2,0 };
	// ������������������Ϊһ���������������ĸ���������������Ϊһ��������
	// ������������������������ͨ����Ӧ��������ͨ��������fromTo�е��м����������ͨ����ϵ��
	mixChannels(&image, 1, &dst, 1, from_to, 3);
	namedWindow("ͨ�����", WINDOW_FREERATIO);
	imshow("ͨ�����", dst);
}

void QuickMethod::Inrange(Mat& image) {
	Mat hsv,mask;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv, Scalar(35,43,46), Scalar(77,255,255), mask);
	namedWindow("ͼ��ɫ�ʿռ�ת��", WINDOW_FREERATIO);
	imshow("ͼ��ɫ�ʿռ�ת��", mask);

	Mat red_background = Mat::zeros(image.size(), image.type());
	red_background = Scalar(40, 40, 220);
	bitwise_not(mask, mask);
	image.copyTo(red_background, mask);
	namedWindow("��ͼ���", WINDOW_FREERATIO);
	imshow("��ͼ���", red_background);
}

void QuickMethod::PixelStatistic(Mat& image) {
	Mat mean, std_dev;
	vector<Mat> mv;
	split(image,mv);
	double min, max;
	Point min_loc, max_loc;
	for (int i = 0; i < mv.size(); i++) {
		minMaxLoc(mv[i], &min, &max, &min_loc, &max_loc, Mat());
		cout <<"channel: "<< i << " min value: " << min << ", max value: " << max << endl;
	}
	// cout << "min location: " << min_loc << ", max location: " << max_loc << endl;
	for (int i = 0; i < mv.size(); i++) {
		meanStdDev(mv[i], mean, std_dev);
		cout << "channel: " << i << endl << "mean value: " << endl << mean << endl << "std value: " << endl << std_dev << endl;
	}
	
}