#include "camera.h"
#include "quickMethod.h"

using namespace std;

void Camera::OpenCamera(int camera_code) {
	try {
		// 调用camera_code号摄像头
		VideoCapture cap(camera_code);
		QuickMethod qm;
		Mat frame;
		int frame_width, frame_height;
		double fps;
		frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
		frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
		fps = cap.get(CAP_PROP_FPS);
		cout << "frame width: " << frame_width << endl;
		cout << "frame height: " << frame_height << endl;
		cout << "frame fps: " << fps << endl;
		namedWindow("调用摄像头", WINDOW_FREERATIO);
		if (!cap.isOpened()) {
			cout << "camera is not opened." << endl;
		} else {
			while (true) {
				//cap >> frame;
				cap.read(frame);
				flip(frame, frame, 1);
				if (frame.empty()) {
					break;
				}
				imshow("调用摄像头", frame);
				// 可以做图像处理
				qm.ColorSpace(frame);
				int c = waitKey(100);
				// 按ESC退出
				if (c == 27) {
					break;
				}
			}
		}
		cap.release();
		destroyAllWindows();
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Camera::VideoOperate() {
	try {
		VideoCapture cap("C:/Users/lenovo/Desktop/QQ录屏20211219153820.mp4");
		QuickMethod qm;
		Mat frame;
		int frame_width, frame_height, frame_count;
		double fps;
		frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
		frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
		frame_count = cap.get(CAP_PROP_FRAME_COUNT);
		fps = cap.get(CAP_PROP_FPS);
		cout << "frame width: " << frame_width << endl;
		cout << "frame height: " << frame_height << endl;
		cout << "frame count: " << frame_count << endl;
		cout << "fps:" << fps << endl;
		namedWindow("视频操作", WINDOW_FREERATIO);
		while (true) {
			//cap >> frame;
			cap.read(frame);
			if (frame.empty()) {
				break;
			}
			imshow("视频操作", frame);
			// 可以做图像处理
			qm.ColorSpace(frame);
			int c = waitKey(1);
			// 按ESC退出
			if (c == 27) {
				break;
			}
		}
		cap.release();
		destroyAllWindows();
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Camera::VideoSave() {
	try {
		VideoCapture cap("C:/Users/lenovo/Desktop/QQ录屏20211219153820.mp4");
		QuickMethod qm;
		Mat frame;
		int frame_width, frame_height, frame_count;
		double fps;
		frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
		frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
		frame_count = cap.get(CAP_PROP_FRAME_COUNT);
		fps = cap.get(CAP_PROP_FPS);
		cout << "frame width: " << frame_width << endl;
		cout << "frame height: " << frame_height << endl;
		cout << "frame count: " << frame_count << endl;
		cout << "fps:" << fps << endl;
		VideoWriter writer("C:/Users/lenovo/Desktop/VideoSave.mp4", cap.get(CAP_PROP_FOURCC), fps, Size(frame_width, frame_height), true);
		namedWindow("视频保存", WINDOW_FREERATIO);
		while (true) {
			//cap >> frame;
			cap.read(frame);
			if (frame.empty()) {
				break;
			}
			imshow("视频保存", frame);
			// 可以做图像处理，返回Mat类型进行保存
			
			// 写入帧
			writer.write(frame);
			int c = waitKey(1);
			// 按ESC退出
			if (c == 27) {
				break;
			}
		}
		cap.release();
		writer.release();
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

void Camera::VideoFaceDetection() {
	try {
		// 调用camera_code号摄像头
		VideoCapture cap(0);
		QuickMethod qm;
		Mat frame;
		int frame_width, frame_height;
		double fps;
		frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
		frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
		fps = cap.get(CAP_PROP_FPS);
		cout << "frame width: " << frame_width << endl;
		cout << "frame height: " << frame_height << endl;
		cout << "frame fps: " << fps << endl;
		namedWindow("调用摄像头", WINDOW_FREERATIO);
		if (!cap.isOpened()) {
			cout << "camera is not opened." << endl;
		} else {
			dnn::Net net = qm.LoadNet();
			while (true) {
				//cap >> frame;
				cap.read(frame);
				flip(frame, frame, 1);
				if (frame.empty()) {
					break;
				}
				imshow("调用摄像头", frame);
				// 可以做图像处理
				qm.FaceDetection(frame, net);
				int c = waitKey(1);
				// 按ESC退出
				if (c == 27) {
					break;
				}
			}
		}
		cap.release();
		destroyAllWindows();
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

