#include "camera.h"
#include <iostream>
#include <conio.h>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

void Camera::OpenCamera(int camera_code) {
	try {
		int ch = 0;
		// ����camera_code������ͷ
		VideoCapture cap(camera_code);
		namedWindow("��������ͷ", WINDOW_FREERATIO);
		if (!cap.isOpened()) {
			cout << "camera is not opened." << endl;
		} else {
			while (true) {
				Mat frame;
				cap >> frame;
				if (!frame.empty()) {
					imshow("��������ͷ", frame);
					// ˢ��ͼ��Ƶ��Ϊ30ms
					waitKey(30);
				}
			}
		}
	} catch (const std::exception&) {
		cout << "error" << endl;
	}
}

