#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/objdetect.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <sstream>

using namespace cv;
using namespace std;
// Global variables
char* window_name[] = { "output","gray threshold","canny","mser" };

int threshold_value = 127, Canny_threshold = 127, Hough_threshold, Erosion_type, Erosion_size;

Mat gray, src, dst;


int main(int argc, char** argv) {
	void thresholdFunc(int, void*);
	char* trackbarTitles[50] = { "Thresholding","Canny","Hough","Erosion Type","Erosion Kernel Size" };

	//load image
	src = imread(argv[1], 1);
	if (src.empty())
		return -1;

	//make gray
	cv::cvtColor(src, gray, CV_BGR2GRAY);

	//make window to show output
	namedWindow(window_name[0], CV_WINDOW_AUTOSIZE);

	//make trackbar to control the window
	createTrackbar(trackbarTitles[0], window_name[0], &threshold_value, 255, thresholdFunc);
	createTrackbar(trackbarTitles[1], window_name[0], &Canny_threshold, 1000, thresholdFunc);
	createTrackbar(trackbarTitles[2], window_name[0], &Hough_threshold, 150, thresholdFunc);
	createTrackbar(trackbarTitles[3], window_name[0], &Erosion_type, 2, thresholdFunc);
	createTrackbar(trackbarTitles[4], window_name[0], &Erosion_size, 21, thresholdFunc);

	//initialize
	thresholdFunc(0, 0);
	while (true)
	{
		int c;
		c = waitKey(20);
		if ((char)c == 27)
		{
			break;
		}
	}

}

void thresholdFunc(int, void*)
{
	int	start = getTickCount();
	void mserExtractor(const Mat& image, Mat& mserOutMask, vector<Mat>&output_images);
	void HoughFunc(Mat &input);
	static Mat gray_dst, canny_dst, mserdst;
	vector<Mat>imageRegions;
	dst = src.clone();
	mserdst = src.clone();
	

	//image thresholding
	threshold(gray, gray_dst, threshold_value, 255, 3);
	imshow(window_name[1], gray_dst);

	//MSER feature detection
	mserExtractor(gray_dst, mserdst,imageRegions);
	imshow(window_name[3], mserdst);
	int i = 0;
	for (vector<Mat>::iterator it = imageRegions.begin();it != imageRegions.end();it++) {
		stringstream temp;
		temp << i;
		imshow(temp.str(), (*it));
	}

	//edge detection
	Canny(gray_dst, canny_dst, Canny_threshold, Canny_threshold * 3, 3);
	imshow(window_name[2], canny_dst);
	
	//hough lines
	HoughFunc(canny_dst);

	imshow(window_name[0], dst);
	printf("%d", getTickCount() - start);
	
}

void HoughFunc(Mat &input) {

	std::vector<Vec2f>lines;
	cv::HoughLines(input, lines, 1, CV_PI / 180, 50 + Hough_threshold, 0, 0);

	for (Vec2f vec2f_line : lines) {//for each line in lines
		float r = vec2f_line[0], t = vec2f_line[1];
		float cos_t = cosf(t), sin_t = sinf(t);
		float x0 = r*cos_t, y0 = r*sin_t;
		float alpha = 50;
		Point pt1(cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t));
		Point pt2(cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t));

		line(dst, pt1, pt2, Scalar(255, 0, 0), 1, LINE_AA);
	}
}

void mserExtractor(const Mat& image, Mat& mserOutMask,vector<Mat>&output_images) {
	Ptr<MSER> mserExtractor = MSER::create();
	vector<vector<cv::Point>> mserContours;
	vector<KeyPoint> mserKeypoint;
	vector<cv::Rect> mserBbox;
	mserExtractor->detectRegions(image, mserContours, mserBbox);

	int i=0;
	stringstream windowname;
	int prev_x = 0, prev_width = 0, prev_y = 0, prev_height = 0;
	for (std::vector<cv::Rect>::iterator it = mserBbox.begin();it != mserBbox.end();it++) {
		float aspect_ratio = (((float)(*it).height) / ((float)(*it).width));
		//consider only rectangles of appropriate dimensions and that are in the bounds of the original image
		if (aspect_ratio > 0.75
			&& aspect_ratio < 1.25
			&& (*it).x + (*it).width < image.size().width
			&& (*it).y + (*it).height < image.size().height
			&& (*it).width>30
			&& (*it).height>30
			&& !(
				(*it).x == prev_x
				&& (*it).y == prev_y
				&& (*it).width == prev_width
				&& (*it).height == prev_height)
			) {
			windowname << i++;
			rectangle(mserOutMask, *it, Scalar(255, 0, 0), 4, 8, 0);
			imshow(windowname.str(), image(Range((*it).y, (*it).y + (*it).height), Range((*it).x, (*it).x + (*it).width)));
			prev_x = (*it).x;
			prev_y = (*it).y;
			prev_height = (*it).height;
			prev_width = (*it).width;
		}
	}
}

void HOGfeatureExtractor(const Mat& input_image, Mat & output_data) {
	Mat resized_image;
	resize(input_image, resized_image, Size(200,200));//get a standard size for images to work with
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	//window size=64x128 , we will resize everything to be 64x128
	//block size=16 pixels long
	//block step size (block stride)=8pixels (so each block starts at 8 pixels after the next
	//cell size =8
	//bin count =9
	vector<float> descriptorValue;
	vector<Point> locations;
	hog.compute(resized_image, descriptorValue, Size(0, 0), Size(0, 0), locations);
	
}