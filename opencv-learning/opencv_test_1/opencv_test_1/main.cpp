#include <iostream>
#include "camera.h"
#include "image.h"

using namespace std;

int main() {
	Camera camera;
	Image image;
	int camera_code, image_type, switch_num, flag = 1;
	string;
	while (flag == 1) {
		cout << "input switch number:";
		cin >> switch_num;
		switch (switch_num) {
			case 1:
				cout << "input image type(color = 0, gray = 1):";
				cin >> image_type;
				image.OpenImage(image_type);
				break;
			case 2:
				// 色彩空间转换
				image.ImageColorSpace();
				break;
			case 3:
				// 矩阵生成
				image.ImageMatrixCreation();
				break;
			case 4:
				// 像素访问
				image.ImagePixelVisit();
				break;
			case 5:
				// 操作数
				image.ImageOperator();
				break;
			case 6:
				// 滚动条亮度显示
				image.ImageTrackingBar();
				break;
			case 11:
				cout << "input camera code(default is 0):";
				cin >> camera_code;
				camera.OpenCamera(camera_code);
				break;
			default:
				flag = 0;
				break;
		}
	}
	return 0;
}
