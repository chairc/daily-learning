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
	// imwrite("F:/program/opencv-learning/opencv_test_1_source_file/HSV.png", hsv);
	// imwrite("F:/program/opencv-learning/opencv_test_1_source_file/GRAY.png", gray);
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

void QuickMethod::Drawing(Mat& image) {
	Rect rect;
	rect.x = 300;
	rect.y = 300;
	rect.width = 500;
	rect.height = 500;
	// ���ƾ���
	// -1������䣬����0����߿��ϸ
	rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
	// ����Բ
	circle(image, Point(100, 100), 50, Scalar(0, 255, 255), -1, 8, 0);
	// ������
	line(image, Point(400, 100), Point(500, 200), Scalar(0, 255, 0), 2, 8, 0);
	// ������Բ
	RotatedRect rotate_rect;
	rotate_rect.center = Point(900, 900);
	rotate_rect.size = Size(500, 100);
	// ��Բ����ת�Ƕ�
	rotate_rect.angle = 0.0;
	ellipse(image, rotate_rect, Scalar(255, 255, 255), 2, 8);
	namedWindow("������ʾ", WINDOW_FREERATIO);
	imshow("������ʾ", image);

	// ����һ������Ч��
	Mat dst;
	Mat background = Mat::zeros(image.size(), image.type());
	rectangle(background, rect, Scalar(0, 0, 255), -1, 8, 0);
	circle(background, Point(100, 100), 50, Scalar(0, 255, 255), -1, 8, 0);
	line(background, Point(400, 100), Point(500, 200), Scalar(0, 255, 0), 2, 8, 0);
	// ����Ȩ�أ���ͼ1��ͼ1��Ȩ�أ�ͼ2��ͼ2��Ȩ�أ�Ȩ�غ���ӵ�ֵΪ3�����ͼƬsrc��
	addWeighted(image, 0.7, background, 0.3, 0, dst);
	namedWindow("����Ч��", WINDOW_FREERATIO);
	imshow("����Ч��", dst);

}

void QuickMethod::RandomDrawing() {
	Mat background = Mat::zeros(Size(512, 512), CV_8UC3);
	// ���������
	RNG rng(12345);
	int width, height;
	width = background.cols;
	height = background.rows;
	while (true) {
		int x1, x2, y1, y2, b, g, r;
		int c = waitKey(100);
		// ��ESC�˳�
		if (c == 27) {
			break;
		}
		// �������
		x1 = rng.uniform(0, width);
		x2 = rng.uniform(0, width);
		y1 = rng.uniform(0, height);
		y2 = rng.uniform(0, height);
		b = rng.uniform(0, 255);
		g = rng.uniform(0, 255);
		r = rng.uniform(0, 255);
		// ����
		background = Scalar(0, 0, 0);
		// ���������
		line(background, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 2, 8, 0);
		imshow("���������ʾ", background);
		// ����0.5��
		Sleep(500);
	}
}

void QuickMethod::PolylineDrawing() {
	Mat background = Mat::zeros(Size(512, 512), CV_8UC3);
	Point p1(100, 100), p2(200, 140), p3(250, 180), p4(300, 230), p5(80, 220);
	vector<Point> point_vector;
	point_vector.push_back(p1);
	point_vector.push_back(p2);
	point_vector.push_back(p3);
	point_vector.push_back(p4);
	point_vector.push_back(p5);
	// �������
	fillPoly(background, point_vector, Scalar(255, 255, 0), LINE_AA, 0);
	// ���ƶ����
	polylines(background, point_vector, true, Scalar(0, 0, 255), 2, LINE_AA, 0);
	// ���+����
	vector<vector<Point>> contours;
	contours.push_back(point_vector);
	// ��������������Ҫ�����ĸ�����Σ�-1����ȫ���������������������������-1�������
	//drawContours(background, contours, -1, Scalar(0, 255, 0), 2);
	imshow("����λ���", background);

}

// ������ʼ����ֹ��
Point mouse_on_drawing_start(-1, -1), mouse_on_drawing_end(-1,-1);
// ��ʱ�������ڲ���
Mat mouse_on_drawing_mat;

static void MouseOnDrawing(int event, int x, int y, int flag, void* user_data) {
	int dx, dy, temp_x, temp_y;
	Mat image = *((Mat*)user_data);
	if (event == EVENT_LBUTTONDOWN) {
		mouse_on_drawing_start.x = x;
		mouse_on_drawing_start.y = y;
		cout << "draw point start:(" << x << "," << y << ")..." << endl;
	} else if (event == EVENT_LBUTTONUP) {
		mouse_on_drawing_end.x = x;
		mouse_on_drawing_end.y = y;
		cout << "draw point end:(" << x << "," << y << ")..." << endl;
		if ((mouse_on_drawing_end.x < mouse_on_drawing_start.x)&& (mouse_on_drawing_end.y < mouse_on_drawing_start.y)) {
			swap(mouse_on_drawing_end.x, mouse_on_drawing_start.x);
			swap(mouse_on_drawing_end.y, mouse_on_drawing_start.y);
		} else if((mouse_on_drawing_end.x < mouse_on_drawing_start.x) && (mouse_on_drawing_end.y > mouse_on_drawing_start.y)) {
			swap(mouse_on_drawing_end.x, mouse_on_drawing_start.x);
		} else if ((mouse_on_drawing_end.x > mouse_on_drawing_start.x) && (mouse_on_drawing_end.y < mouse_on_drawing_start.y)) {
			swap(mouse_on_drawing_end.y, mouse_on_drawing_start.y);
		} else if ((mouse_on_drawing_end.x > mouse_on_drawing_start.x) && (mouse_on_drawing_end.y > mouse_on_drawing_start.y)) {
			
		}
		dx = abs(mouse_on_drawing_end.x - mouse_on_drawing_start.x);
		dy = abs(mouse_on_drawing_end.y - mouse_on_drawing_start.y);
		Rect rect(mouse_on_drawing_start.x, mouse_on_drawing_start.y, dx, dy);
		rectangle(image, rect, Scalar(0, 0, 255), 2, LINE_AA, 0);
		imshow("������", image);
		imshow("ROI����", image(rect));
		// ׼����һ�λ���
		mouse_on_drawing_start.x = -1;
		mouse_on_drawing_start.y = -1;

	}else if (event == EVENT_MOUSEMOVE) {
		if (mouse_on_drawing_start.x > 0 && mouse_on_drawing_start.y > 0) {
			mouse_on_drawing_end.x = x;
			mouse_on_drawing_end.y = y;
			cout << "draw point move:(" << x << "," << y << ")..." << endl;
			dx = mouse_on_drawing_end.x - mouse_on_drawing_start.x;
			dy = mouse_on_drawing_end.y - mouse_on_drawing_start.y;
			Rect rect(mouse_on_drawing_start.x, mouse_on_drawing_start.y, dx, dy);
			mouse_on_drawing_mat.copyTo(image);
			rectangle(image, rect, Scalar(0, 0, 255), 2, LINE_AA, 0);
			imshow("������", image);
		}
	}
}

void QuickMethod::MouseDrawing(Mat& image) {
	namedWindow("������", WINDOW_FREERATIO);
	setMouseCallback("������", MouseOnDrawing, (void*)(&image));
	imshow("������", image);
	mouse_on_drawing_mat = image.clone();
}

void QuickMethod::Norm(Mat& image) {
	Mat img_float;
	cout << "image data type: " << image.type() << endl;
	// ת������ CV_8UC3��ΪCV_32FC3
	image.convertTo(image, CV_32F);
	cout << "image data type: " << image.type() << endl;
	// NORM_MINMAX:�������ֵ��ƽ�ƻ����ŵ�һ��ָ���ķ�Χ�����Թ�һ����
	normalize(image, img_float, 1.0, 0, NORM_MINMAX);
	cout << "img_float data type: " << img_float.type() << endl;
	namedWindow("���ع�һ��", WINDOW_FREERATIO);
	imshow("���ع�һ��", img_float);
}

void QuickMethod::Resize(Mat& image) {
	Mat zoom_in, zoom_out;
	int width = image.rows, height = image.cols;
	imshow("ԭͼ", image);

	// INTER_LINEAR���Բ�ֵ
	resize(image, zoom_in, Size(width / 2, height / 2), 0, 0, INTER_LINEAR);
	imshow("��С", zoom_in);

	resize(image, zoom_out, Size(2 * width, 2 * height), 0, 0, INTER_LINEAR);
	imshow("�Ŵ�", zoom_out);
}

void QuickMethod::Flip(Mat& image) {
	Mat dst;
	// ���·�ת
	flip(image, dst, 0);
	namedWindow("���·�ת", WINDOW_FREERATIO);
	imshow("���·�ת", dst);
	// ���ҷ�ת
	flip(image, dst, 1);
	namedWindow("���ҷ�ת", WINDOW_FREERATIO);
	imshow("���ҷ�ת", dst);
	// ���� + ���·�ת
	flip(image, dst, -1);
	namedWindow("�������ҷ�ת", WINDOW_FREERATIO);
	imshow("�������ҷ�ת", dst);
}

void QuickMethod::Rotate(Mat& image) {
	Mat dst, M;
	double cos, sin;
	int new_width, new_height;
	int width = image.rows, height = image.cols;
	// ��һ��������ʾԭʼͼ�������λ�ã��ڶ���������ʾ��ת�Ƕȣ�������������ʾͼ���Ƿ����
	M = getRotationMatrix2D(Point2f(width / 2, height / 2), 45, 1.0);
	// ��ȡM�е�һ��һ�к�һ�ж��е�ֵ
	cos = abs(M.at<double>(0, 0));
	sin = abs(M.at<double>(0, 1));
	// �����µĸߺͿ�
	new_width = cos * width + sin * height;
	new_height = sin * width + cos * height;
	// ��������������
	M.at<double>(0, 2) = M.at<double>(0, 2) + (new_width / 2 - width / 2);
	M.at<double>(1, 2) = M.at<double>(1, 2) + (new_height / 2 - height / 2);
	warpAffine(image, dst, M, Size(new_width, new_height), INTER_LINEAR, 0, Scalar(255, 0, 0));
	namedWindow("ͼ����ת", WINDOW_FREERATIO);
	imshow("ͼ����ת", dst);
}
