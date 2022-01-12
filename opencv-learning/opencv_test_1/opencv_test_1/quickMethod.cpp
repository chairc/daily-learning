#include "quickMethod.h"
#include <iostream>

using namespace std;

void QuickMethod::ColorSpace(Mat& image) {
	// 将彩色图片转换为灰度图和HSV图
	Mat gray, hsv;
	// 取值范围 H：0 ~ 180 S：0 ~ 255 V：0 ~ 255 S和H调整颜色，V调整亮度
	cvtColor(image, hsv, COLOR_BGR2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	namedWindow("HSV", WINDOW_FREERATIO);
	namedWindow("GRAY", WINDOW_FREERATIO);
	imshow("HSV", hsv);
	imshow("GRAY", gray);
	// 保存图片
	imwrite("F:/program/opencv-learning/opencv_test_1_source_file/HSV.png", hsv);
	imwrite("F:/program/opencv-learning/opencv_test_1_source_file/GRAY.png", gray);
}

void QuickMethod::MatrixCreation(Mat& image) {
	Mat mat_clone, mat_copy;
	// 克隆
	mat_clone = image.clone();

	// 复制
	image.copyTo(mat_copy);

	// 创建空白Matrix，CV_8UC1表示8位无符号1通道
	Mat mat_zero = Mat::zeros(Size(8, 8), CV_8UC1);
	cout << "mat_zero matrix: " << mat_zero << endl;
}

void QuickMethod::MatrixCreation() {


	// 创建空白Matrix，CV_8UC1表示8位无符号3通道
	Mat mat_custom = Mat::zeros(Size(400, 400), CV_8UC3);
	// 对3通道赋予指定的数值
	mat_custom = Scalar(0, 255, 0);
	// cout << "mat_custom matrix: " << mat_custom << endl;
	imshow("自定义赋值图像", mat_custom);

	Mat mat_clone = mat_custom.clone();
	mat_clone = Scalar(255, 0, 0);
	imshow("mat_clone图像", mat_clone);

	Mat mat_copy;
	mat_custom.copyTo(mat_copy);
	mat_copy = Scalar(170, 170, 170);
	imshow("mat_copy图像", mat_custom);

	Mat mat_custom2 = mat_custom;
	mat_custom2 = Scalar(0, 0, 255);
	imshow("mat_custom图像", mat_custom);
	imshow("mat_custom2图像", mat_custom2);
}

void QuickMethod::PixelVisit(Mat& image) {
	int width = image.cols;
	int height = image.rows;
	int channels = image.channels();
	cout << "width: " << width << " height: " << height << " channels: " << channels << endl;
	// 数组方法
	/*for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (channels == 1) {	// 灰度图像
				int pv = image.at<uchar>(row, col);
				image.at<uchar>(row, col) = 255 - pv;
			} else if (channels == 3) {	// 彩色图像
				Vec3b bgr = image.at<Vec3b>(row, col);
				image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
			}
		}
	}*/

	// 指针方法
	for (int row = 0; row < height; row++) {
		uchar* current_row = image.ptr<uchar>(row);
		for (int col = 0; col < width; col++) {
			if (channels == 1) {	// 灰度图像
				int pv = *current_row;
				*current_row++ = 255 - pv;
			} else if (channels == 3) {	// 彩色图像
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}
		}
	}
	namedWindow("像素读写", WINDOW_FREERATIO);
	imshow("像素读写", image);
}

void QuickMethod::Operators(Mat& image) {
	// 正常加减乘除法
	Mat dst_1, dst_2, dst_3, dst_4;
	dst_1 = image + Scalar(20, 20, 20);
	namedWindow("加法操作", WINDOW_FREERATIO);
	imshow("加法操作", dst_1);

	dst_2 = image - Scalar(50, 50, 50);
	namedWindow("减法操作", WINDOW_FREERATIO);
	imshow("减法操作", dst_2);

	dst_3 = image / Scalar(2, 2, 2);
	namedWindow("除法操作", WINDOW_FREERATIO);
	imshow("除法操作", dst_3);

	Mat dst_temp = Mat::zeros(image.size(), image.type());
	dst_temp = Scalar(2, 2, 2);
	multiply(image, dst_temp, dst_4);
	namedWindow("乘法操作", WINDOW_FREERATIO);
	imshow("乘法操作", dst_4);

	// 数组方法加减乘除
	/*int width = image.cols;
	int height = image.rows;
	int channels = image.channels();
	Mat dst = Mat::zeros(image.size(), image.type());
	dst = Scalar(2, 2, 2);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			Vec3b bgr1 = image.at<Vec3b>(row, col);
			Vec3b bgr2 = dst.at<Vec3b>(row, col);
			// saturate_cast判断是否在0~255之间
			dst_1.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(bgr1[0] + bgr2[0]);
			dst_1.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(bgr1[1] + bgr2[1]);
			dst_1.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(bgr1[2] + bgr2[2]);
		}
	}*/

	// OpenCV的加减乘除API
	Mat dst_result = Mat::zeros(image.size(), image.type());
	Mat mat = Mat::zeros(image.size(), image.type());
	mat = Scalar(20, 20, 20);
	add(image, mat, dst_result);
	subtract(image, mat, dst_result);
	multiply(image, mat, dst_result);
	divide(image, mat, dst_result);
}

// 滚动条回调函数
static void OnLightness(int lightness, void* user_data) {
	Mat image = *((Mat*)user_data);
	Mat dst_on_track = Mat::zeros(image.size(), image.type());
	Mat mat_on_track = Mat::zeros(image.size(), image.type());
	addWeighted(image, 1, mat_on_track, 0, lightness, dst_on_track);
	imshow("亮度与对比度调整", dst_on_track);
}

static void OnContrast(int contrast, void* user_data) {
	Mat image = *((Mat*)user_data);
	Mat dst_on_track = Mat::zeros(image.size(), image.type());
	Mat mat_on_track = Mat::zeros(image.size(), image.type());
	double temp = contrast / 200.0;
	addWeighted(image, temp, mat_on_track, 0.0, 0.0, dst_on_track);
	imshow("亮度与对比度调整", dst_on_track);
}

void QuickMethod::TrackingBar(Mat& image) {
	namedWindow("亮度与对比度调整", WINDOW_FREERATIO);
	int light_max_value = 100, lightness = 50, contrast_value = 100;
	createTrackbar("Value Bar", "亮度与对比度调整", &lightness, light_max_value, OnLightness, (void*)(&image));
	createTrackbar("Contrast Bar", "亮度与对比度调整", &contrast_value, 200, OnContrast, (void*)(&image));
	OnLightness(50, &image);
}

void QuickMethod::Key(Mat& image) {
	Mat dst = Mat::zeros(image.size(), image.type());
	namedWindow("键盘响应", WINDOW_FREERATIO);
	// 不断循环，监听键盘操作
	while (true) {
		int c = waitKey(100);
		// cout << "ascii: " << c << endl;
		// 按ESC退出
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
		imshow("键盘响应", dst);
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
		// 按ESC退出
		if (c == 27) {
			break;
		}
		applyColorMap(image, dst, color_map[i]);
		if (i == 21) {
			i = 0;
		} else {
			i++;
		}
		namedWindow("图片颜色样式循环", WINDOW_FREERATIO);
		imshow("图片颜色样式循环", dst);
	}
}

void QuickMethod::Bitwise(Mat& image) {
	Mat dst_1, dst_2, dst_3, dst_4;
	// 创建两个三通道矩阵
	Mat mat_1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_2 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_3 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_4 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_5 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_6 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_7 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat mat_8 = Mat::zeros(Size(256, 256), CV_8UC3);
	// 给mat_1矩阵设置x,y和大小与颜色，thickness线宽（-1表示填充矩形，大于0表示绘制）
	rectangle(mat_1, Rect(100, 100, 50, 50), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(mat_2, Rect(130, 130, 50, 50), Scalar(0, 255, 255), -1, LINE_8, 0);
	rectangle(mat_3, Rect(100, 100, 50, 50), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(mat_4, Rect(130, 130, 50, 50), Scalar(0, 255, 255), -1, LINE_8, 0);
	rectangle(mat_5, Rect(100, 100, 50, 50), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(mat_6, Rect(130, 130, 50, 50), Scalar(0, 255, 255), -1, LINE_8, 0);
	// imshow("显示mat_1", mat_1);
	// imshow("显示mat_2", mat_2);

	// 位与操作
	bitwise_and(mat_1, mat_2, dst_1);
	imshow("像素位操作（and）", dst_1);

	// 位或操作
	bitwise_or(mat_3, mat_4, dst_2);
	imshow("像素位操作（or）", dst_2);

	// 位非操作
	bitwise_not(image,dst_3);
	// ~image取反
	// dst_3 = ~image;
	namedWindow("像素位操作（not）", WINDOW_FREERATIO);
	imshow("像素位操作（not）", dst_3);

	// 位异或操作
	bitwise_xor(mat_5, mat_6, dst_4);
	imshow("像素位操作（xor）", dst_4);

}