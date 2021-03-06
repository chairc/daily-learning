#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

class Camera {
private:
	int camera_status = 0;	// 摄像头状态
	int camera_number = 10;	// 摄像头数量

public:
	void OpenCamera(int camera_code);	// 打开摄像头
	void VideoOperate();
	void VideoSave();
	void VideoFaceDetection();
};