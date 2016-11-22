#include <opencv2\opencv.hpp>
#include <ctime>
#include <cstdlib>

using namespace cv;
int main() {
	srand((unsigned int)time(NULL));
	Mat createRandomImage();
	void HSVsegment(Mat inputImage, Mat & outputImage, int segmentSize = 30, int whiteThreshold = 30);
	Mat random = createRandomImage(),random_out;
	HSVsegment(random,random_out);
	Mat red, green, blue;
	red = imread("red.png");
	HSVsegment(red,red);
	green = imread("green.png");
	cvtColor(green, green, CV_BGR2HSV);
	blue = imread("blue.png");
	cvtColor(blue, blue, CV_BGR2HSV);
	system("pause");

}

Mat createRandomImage() {
	Mat temp(Size(400, 400), CV_8UC3);
	int nRows = temp.rows;
	int nCols = temp.cols;
	for (int i = 0;i < nRows;i++) {
		for (int j = 0;j < nCols;j++) {
			temp.at<Vec3b>(i, j)[0] = rand() / 256;
			temp.at<Vec3b>(i, j)[1] = rand() / 256;
			temp.at<Vec3b>(i, j)[2] = rand() / 256;
		}
	}
	return temp;
}

void HSVsegment(Mat inputImage, Mat & outputImage, int segmentSize = 30, int whiteThreshold = 255) {
	static int prevSegmentSize = 0;
	static Mat lut(1, 256, CV_8UC3);
	if (prevSegmentSize != segmentSize) {
		for (int i = 0;i < 256;i++) {
			int extraHue = i % segmentSize;
			int Hval;

			if (i + segmentSize - extraHue >= 180)
				Hval = 0;//H channel
			else if (extraHue < segmentSize / 2)
				Hval = i - extraHue;//H channel
			else
				Hval = i - extraHue + segmentSize;//H channel

			lut.at<Vec3b>(i)[0] = Hval;
			if (i <= whiteThreshold)
				lut.at<Vec3b>(i)[1] = 0;//S channel
			else
				lut.at<Vec3b>(i)[1] = 255;//S channel

			lut.at<Vec3b>(i)[2] = 255;//V channel
		}
		prevSegmentSize = segmentSize;
	}

	cvtColor(inputImage, outputImage, CV_BGR2HSV);
	LUT(outputImage, lut, outputImage);
	cvtColor(outputImage, outputImage, CV_HSV2BGR);
}