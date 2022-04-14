#include "camera.h"
#include "quickMethod.h"

using namespace std;

void Camera::OpenCamera(int camera_code) {
	try {
		// ����camera_code������ͷ
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
		namedWindow("��������ͷ", WINDOW_FREERATIO);
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
				imshow("��������ͷ", frame);
				// ������ͼ����
				qm.ColorSpace(frame);
				int c = waitKey(100);
				// ��ESC�˳�
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
		VideoCapture cap("C:/Users/lenovo/Desktop/QQ¼��20211219153820.mp4");
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
		namedWindow("��Ƶ����", WINDOW_FREERATIO);
		while (true) {
			//cap >> frame;
			cap.read(frame);
			if (frame.empty()) {
				break;
			}
			imshow("��Ƶ����", frame);
			// ������ͼ����
			qm.ColorSpace(frame);
			int c = waitKey(1);
			// ��ESC�˳�
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
		VideoCapture cap("C:/Users/lenovo/Desktop/QQ¼��20211219153820.mp4");
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
		namedWindow("��Ƶ����", WINDOW_FREERATIO);
		while (true) {
			//cap >> frame;
			cap.read(frame);
			if (frame.empty()) {
				break;
			}
			imshow("��Ƶ����", frame);
			// ������ͼ��������Mat���ͽ��б���
			
			// д��֡
			writer.write(frame);
			int c = waitKey(1);
			// ��ESC�˳�
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
		// ����camera_code������ͷ
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
		namedWindow("��������ͷ", WINDOW_FREERATIO);
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
				imshow("��������ͷ", frame);
				// ������ͼ����
				qm.FaceDetection(frame, net);
				int c = waitKey(1);
				// ��ESC�˳�
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

