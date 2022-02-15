#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

class Camera {
private:
	int camera_status = 0;	// ����ͷ״̬
	int camera_number = 10;	// ����ͷ����

public:
	void OpenCamera(int camera_code);	// ������ͷ
	void VideoOperate();
	void VideoSave();
	void VideoFaceDetection();
};