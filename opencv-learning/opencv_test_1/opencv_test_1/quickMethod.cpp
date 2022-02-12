#include "quickMethod.h"
#include <iostream>
#include <vector>

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
	// imwrite("F:/program/opencv-learning/opencv_test_1_source_file/HSV.png", hsv);
	// imwrite("F:/program/opencv-learning/opencv_test_1_source_file/GRAY.png", gray);
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
	bitwise_not(image, dst_3);
	// ~image取反
	// dst_3 = ~image;
	namedWindow("像素位操作（not）", WINDOW_FREERATIO);
	imshow("像素位操作（not）", dst_3);

	// 位异或操作
	bitwise_xor(mat_5, mat_6, dst_4);
	imshow("像素位操作（xor）", dst_4);

}

void QuickMethod::Channels(Mat& image) {
	
	vector<Mat> mat_vector;
	split(image, mat_vector);
	namedWindow("通道显示-蓝色", WINDOW_FREERATIO);
	namedWindow("通道显示-绿色", WINDOW_FREERATIO);
	namedWindow("通道显示-红色", WINDOW_FREERATIO);
	namedWindow("重设通道显示-蓝色", WINDOW_FREERATIO);
	namedWindow("重设通道显示-绿色", WINDOW_FREERATIO);
	namedWindow("重设通道显示-红色", WINDOW_FREERATIO);
	// 显示三通道中没一个通道
	imshow("通道显示-蓝色", mat_vector[0]);
	imshow("通道显示-绿色", mat_vector[1]);
	imshow("通道显示-红色", mat_vector[2]);

	Mat dst_r, dst_g, dst_b, dst = Mat::zeros(image.size(), image.type());

	mat_vector[1] = 0;
	mat_vector[2] = 0;
	merge(mat_vector, dst_b);
	imshow("重设通道显示-蓝色", dst_b);

	// 合并之后需要重新拆分原始图像，不然会拆分上一个合并的图片，最终会导致通道全为0
	split(image, mat_vector);
	mat_vector[0] = 0;
	mat_vector[2] = 0;
	merge(mat_vector, dst_g);
	imshow("重设通道显示-绿色", dst_g);

	split(image, mat_vector);
	mat_vector[0] = 0;
	mat_vector[1] = 0;
	merge(mat_vector, dst_r);
	imshow("重设通道显示-红色", dst_r);

	// 混合通道
	int from_to[] = { 0,2,1,1,2,0 };
	// 参数：（输入矩阵可以为一个或多个，输入矩阵的个数，输出矩阵可以为一个或多个，
	// 输出矩阵个数，设置输入矩阵的通道对应输出矩阵的通道，参数fromTo中的有几组输入输出通道关系）
	mixChannels(&image, 1, &dst, 1, from_to, 3);
	namedWindow("通道混合", WINDOW_FREERATIO);
	imshow("通道混合", dst);
}

void QuickMethod::Inrange(Mat& image) {
	Mat hsv,mask;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv, Scalar(35,43,46), Scalar(77,255,255), mask);
	namedWindow("图像色彩空间转换", WINDOW_FREERATIO);
	imshow("图像色彩空间转换", mask);

	Mat red_background = Mat::zeros(image.size(), image.type());
	red_background = Scalar(40, 40, 220);
	bitwise_not(mask, mask);
	image.copyTo(red_background, mask);
	namedWindow("抠图填充", WINDOW_FREERATIO);
	imshow("抠图填充", red_background);
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
	// 绘制矩形
	// -1代表填充，大于0代表边框粗细
	rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
	// 绘制圆
	circle(image, Point(100, 100), 50, Scalar(0, 255, 255), -1, 8, 0);
	// 绘制线
	line(image, Point(400, 100), Point(500, 200), Scalar(0, 255, 0), 2, 8, 0);
	// 绘制椭圆
	RotatedRect rotate_rect;
	rotate_rect.center = Point(900, 900);
	rotate_rect.size = Size(500, 100);
	// 椭圆的旋转角度
	rotate_rect.angle = 0.0;
	ellipse(image, rotate_rect, Scalar(255, 255, 255), 2, 8);
	namedWindow("绘制演示", WINDOW_FREERATIO);
	imshow("绘制演示", image);

	// 绘制一个叠底效果
	Mat dst;
	Mat background = Mat::zeros(image.size(), image.type());
	rectangle(background, rect, Scalar(0, 0, 255), -1, 8, 0);
	circle(background, Point(100, 100), 50, Scalar(0, 255, 255), -1, 8, 0);
	line(background, Point(400, 100), Point(500, 200), Scalar(0, 255, 0), 2, 8, 0);
	// 增加权重，（图1，图1的权重，图2，图2的权重，权重和添加的值为3，输出图片src）
	addWeighted(image, 0.7, background, 0.3, 0, dst);
	namedWindow("叠底效果", WINDOW_FREERATIO);
	imshow("叠底效果", dst);

}

void QuickMethod::RandomDrawing() {
	Mat background = Mat::zeros(Size(512, 512), CV_8UC3);
	// 产生随机数
	RNG rng(12345);
	int width, height;
	width = background.cols;
	height = background.rows;
	while (true) {
		int x1, x2, y1, y2, b, g, r;
		int c = waitKey(100);
		// 按ESC退出
		if (c == 27) {
			break;
		}
		// 随机坐标
		x1 = rng.uniform(0, width);
		x2 = rng.uniform(0, width);
		y1 = rng.uniform(0, height);
		y2 = rng.uniform(0, height);
		b = rng.uniform(0, 255);
		g = rng.uniform(0, 255);
		r = rng.uniform(0, 255);
		// 清屏
		background = Scalar(0, 0, 0);
		// 绘制随机线
		line(background, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 2, 8, 0);
		imshow("随机绘制演示", background);
		// 休眠0.5秒
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
	// 填充多边形
	fillPoly(background, point_vector, Scalar(255, 255, 0), LINE_AA, 0);
	// 绘制多边形
	polylines(background, point_vector, true, Scalar(0, 0, 255), 2, LINE_AA, 0);
	// 填充+绘制
	vector<vector<Point>> contours;
	contours.push_back(point_vector);
	// 第三个参数代表要绘制哪个多边形，-1代表全部，第五个参数正数代表轮廓，-1代表填充
	//drawContours(background, contours, -1, Scalar(0, 255, 0), 2);
	imshow("多边形绘制", background);

}

// 绘制起始和终止点
Point mouse_on_drawing_start(-1, -1), mouse_on_drawing_end(-1,-1);
// 临时矩阵用于擦除
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
		imshow("鼠标绘制", image);
		imshow("ROI区域", image(rect));
		// 准备下一次绘制
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
			imshow("鼠标绘制", image);
		}
	}
}

void QuickMethod::MouseDrawing(Mat& image) {
	namedWindow("鼠标绘制", WINDOW_FREERATIO);
	setMouseCallback("鼠标绘制", MouseOnDrawing, (void*)(&image));
	imshow("鼠标绘制", image);
	mouse_on_drawing_mat = image.clone();
}

void QuickMethod::Norm(Mat& image) {
	Mat img_float;
	cout << "image data type: " << image.type() << endl;
	// 转换类型 CV_8UC3变为CV_32FC3
	image.convertTo(image, CV_32F);
	cout << "image data type: " << image.type() << endl;
	// NORM_MINMAX:数组的数值被平移或缩放到一个指定的范围，线性归一化。
	normalize(image, img_float, 1.0, 0, NORM_MINMAX);
	cout << "img_float data type: " << img_float.type() << endl;
	namedWindow("像素归一化", WINDOW_FREERATIO);
	imshow("像素归一化", img_float);
}

void QuickMethod::Resize(Mat& image) {
	Mat zoom_in, zoom_out;
	int width = image.rows, height = image.cols;
	imshow("原图", image);

	// INTER_LINEAR线性插值
	resize(image, zoom_in, Size(width / 2, height / 2), 0, 0, INTER_LINEAR);
	imshow("缩小", zoom_in);

	resize(image, zoom_out, Size(2 * width, 2 * height), 0, 0, INTER_LINEAR);
	imshow("放大", zoom_out);
}

void QuickMethod::Flip(Mat& image) {
	Mat dst;
	// 上下翻转
	flip(image, dst, 0);
	namedWindow("上下翻转", WINDOW_FREERATIO);
	imshow("上下翻转", dst);
	// 左右翻转
	flip(image, dst, 1);
	namedWindow("左右翻转", WINDOW_FREERATIO);
	imshow("左右翻转", dst);
	// 左右 + 上下翻转
	flip(image, dst, -1);
	namedWindow("上下左右翻转", WINDOW_FREERATIO);
	imshow("上下左右翻转", dst);
}

void QuickMethod::Rotate(Mat& image) {
	Mat dst, M;
	double cos, sin;
	int new_width, new_height;
	int width = image.rows, height = image.cols;
	// 第一个参数表示原始图像的中心位置，第二个参数表示旋转角度，第三个参数表示图像是否放缩
	M = getRotationMatrix2D(Point2f(width / 2, height / 2), 45, 1.0);
	// 获取M中第一行一列和一行二列的值
	cos = abs(M.at<double>(0, 0));
	sin = abs(M.at<double>(0, 1));
	// 计算新的高和宽
	new_width = cos * width + sin * height;
	new_height = sin * width + cos * height;
	// 计算新中心坐标
	M.at<double>(0, 2) = M.at<double>(0, 2) + (new_width / 2 - width / 2);
	M.at<double>(1, 2) = M.at<double>(1, 2) + (new_height / 2 - height / 2);
	warpAffine(image, dst, M, Size(new_width, new_height), INTER_LINEAR, 0, Scalar(255, 0, 0));
	namedWindow("图像旋转", WINDOW_FREERATIO);
	imshow("图像旋转", dst);
}
