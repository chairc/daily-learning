#pragma once

enum MethodEnum {
	ImageColorSpace,
	ImageMatrixCreation,
	ImagePixelVisit,
	ImageOperator,
	ImageTrackingBar,
	ImageKey,
	ImageColorStyle,
	ImageBitwise,
	ImageChannels,
	ImageInRange,
	ImagePixelStatistic,
};

static void ImageCommonMethod(MethodEnum type);

class Image {
private:

public:
	void OpenImage(int image_type);
	void ImageColorSpace();
	void ImageMatrixCreation();
	void ImagePixelVisit();
	void ImageOperator();
	void ImageTrackingBar();
	void ImageKey();
	void ImageColorStyle();
	void ImageBitwise();
	void ImageChannels();
	void ImageInRange();
	void ImagePixelStatistic();
};