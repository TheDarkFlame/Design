#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/objdetect.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>

typedef struct ExtractionProperties {
	int Threshold_L;//lower threshold (anything below becomes black)
	int Threshold_U;//upper threshold (currently not implemented)
	double Ratio_L;//lower threshold for ratio of width to height of detected regions
	double Ratio_U;//upper threshold for ratio of width to height of detected regions
	int Area_max;//maximum area size of regions
	int Area_min;//minimum area size of regions
	int MatchingPercent;//this is percentage of width that the ROIs edges must match by to be considered duplicates
} ExtractionProperties;

using namespace cv;
using namespace std;
// Global variables
char* window_name[] = { "Thresholding","ROIS" };

int threshold_value = 70;
int HSV_threshold = 9;
int counter = 1;
int counter_minus1 = 0;

Mat gray, src, hsv, hsv_disp, gray_disp;

string name;
int range;
string extension;

int main(int argc, char** argv) {
	string createFilename(string extension, string baseName, int number, int maxNumber);
	string keys =
		"{name|test|this is the basename of every image, image is of form <name><number><ext> eg test1.png}"
		"{maxNumber|10|the number of the last image in the series}"
		"{ext|.png|the extension of the images}"
		;
	CommandLineParser parser(argc, argv, keys);
	name = parser.get<string>("name");
	range = parser.get<int>("maxNumber") + 1;
	extension = parser.get<string>("ext");

	if (range < 1)
		return -1;

	void imgSel(int, void*);
	void thresholdFunc(int, void*);
	cout << "Region Detection Demo" << endl;
	//load image
	src = imread(createFilename(extension, name, 1, range), 1);
	if (src.empty())
		return -1;

	//make window to show output
	namedWindow(window_name[0], CV_WINDOW_AUTOSIZE);

	//make trackbar to control the window
	createTrackbar("Thresholding", window_name[0], &threshold_value, 255, thresholdFunc);
	createTrackbar("Image Select", window_name[0], &counter_minus1, range - 2, imgSel);
	//createTrackbar("HSVthresh", window_name[0], &HSV_threshold, 60, thresholdFunc);
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
		else if (char(c) == 32) {
			((counter + 1) % range == 0) ? counter = 1 : counter = counter + 1;//wrap around, but skip 0
			string fname = createFilename(extension, name, counter, range);
			src = imread(fname);
			cout << fname << endl;
			thresholdFunc(NULL, NULL);
		}
	}

}

string createFilename(string extension, string baseName, int number, int maxNumber) {
	stringstream filename;
	filename << baseName << (number % maxNumber) << extension;
	string fullname = filename.str();
	return fullname;
}

void thresholdFunc(int, void*)
{
	void HSVsegment(Mat inputImage, Mat & outputImage, int segmentSize = 30, int whiteThreshold = 30);
	double mserExtractor(Mat inputImage, vector<Rect> & Rectangles, ExtractionProperties & props);
	int64 time = getTickCount();
	
	if (HSV_threshold < 9) {
		HSV_threshold = 9;
	}

	ExtractionProperties ExProps;
	ExProps.Area_min = 20 * 20;
	ExProps.Area_max = 0;//no max limit
	ExProps.Ratio_L = 0.8;
	ExProps.Ratio_U = 1.25;
	ExProps.MatchingPercent = 25;
	ExProps.Threshold_L = 127;
	ExProps.Threshold_U = 200;
	
	//gray thresholding
	cvtColor(src, gray, CV_BGR2GRAY);
	threshold(gray, gray, threshold_value, 255, 3);

	//HSV segmenting
	HSVsegment(src, hsv, 30, HSV_threshold);

	cout << "Segmentation   = " << (getTickCount() - time) / getTickFrequency() * 1000 << "ms" << endl;
	

	Mat dispImg;
	src.copyTo(dispImg);
	hsv.copyTo(hsv_disp);
	gray.copyTo(gray_disp);
	//MSER feature detection
	vector<Rect>Rectangles;

	cout << "Grayscale MSER = " << 1000 * mserExtractor(gray, Rectangles, ExProps) << "ms";
	cout << " | Regions = " << Rectangles.size() << endl;
	for (vector<Rect>::iterator it = Rectangles.begin();it != Rectangles.end();it++) {
		rectangle(dispImg, *it, Scalar(0, 255, 0), 1);
		rectangle(gray_disp, *it, Scalar(0, 0, 0), 2);
		rectangle(gray_disp, *it, Scalar(255, 255, 255), 1);
	}
	putText(dispImg, "Grayscale", Point(5, dispImg.rows - 5), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(0, 255, 0), 1);

	Rectangles.erase(Rectangles.begin(), Rectangles.end());

	cout << "HSV MSER       = " << 1000 * mserExtractor(hsv, Rectangles, ExProps) << "ms";
	cout << " | Regions = " << Rectangles.size() << endl << endl;
	for (vector<Rect>::iterator it = Rectangles.begin();it != Rectangles.end();it++) {
		rectangle(dispImg, *it, Scalar(0, 0, 255), 1);
		rectangle(hsv_disp, *it, Scalar(0, 0, 0), 2);
		rectangle(hsv_disp, *it, Scalar(255, 255, 255), 1);
	}
	putText(dispImg, "BaryCenters", Point(5, dispImg.rows - 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(0, 0, 255), 1);

	imshow(window_name[0], dispImg);

	imshow("HSV", hsv_disp);
	imshow("Grayscale", gray_disp);
}

void imgSel(int, void*) {
	string createFilename(string extension, string baseName, int number, int maxNumber);
	
	counter = counter_minus1 + 1;
	string fname = createFilename(extension, name, counter, range);
	src = imread(fname);
	cout << fname << endl;
	thresholdFunc(NULL, NULL);
}

double mserExtractor(Mat inputImage, vector<Rect> & Rectangles, ExtractionProperties & props) {
	//function declarations
	bool FilterROIResults(Rect RectangleUnderTesting, ExtractionProperties & props);
	//variable declarations
	int64 start = getTickCount();
	vector<Rect>temp;

	//set up the MSERextractor
	Ptr<MSER> mserExtractor = MSER::create();
	if (props.Area_min != 0)
		mserExtractor->setMinArea(props.Area_min);
	if (props.Area_max != 0)
		mserExtractor->setMaxArea(props.Area_max);
	vector<vector<Point>> dummyVector;
	mserExtractor->detectRegions(inputImage, dummyVector, temp);
	for (vector<Rect>::iterator it = temp.begin();it != temp.end();it++) {
		if (FilterROIResults(*it, props))//filter based on dimensions
			Rectangles.push_back(*it);
	}

	return (getTickCount() - start) / getTickFrequency();
}

bool FilterROIResults(Rect RectangleUnderTesting, ExtractionProperties & props) {
	bool retval = true;//return value defaults to true

	static Rect Previous(Point(0, 0), Point(0, 0));
	float ratio = (float)RectangleUnderTesting.height / (float)RectangleUnderTesting.width;
	//if image is not the right proportion return false
	if (ratio > props.Ratio_U)
		retval = false;
	else if (ratio < props.Ratio_L)
		retval = false;
	else {
		//if the image is the same as the last image return false
		int duplicitiyCount = 0;
		//test left side
		if (std::abs(RectangleUnderTesting.x - (Previous).x) <= ((Previous.width * props.MatchingPercent) / 100))
			duplicitiyCount++;
		//test top side
		if (std::abs(RectangleUnderTesting.y - (Previous).y) <= ((Previous.height * props.MatchingPercent) / 100))
			duplicitiyCount++;
		//test right side
		if (std::abs((RectangleUnderTesting.width + RectangleUnderTesting.x) - ((Previous).width + (Previous).x)) <= ((Previous.width * props.MatchingPercent) / 100))
			duplicitiyCount++;
		//test bottom side
		if (std::abs((RectangleUnderTesting.height + RectangleUnderTesting.y) - ((Previous).height + (Previous).y)) <= ((Previous.width * props.MatchingPercent) / 100))
			duplicitiyCount++;
		
		if (duplicitiyCount > 1) {//if there are 2 sides that are about the same consider it a duplicate
			retval = false;
		}
	}
	Previous = RectangleUnderTesting;
	return retval;
}

void HSVsegment(Mat inputImage, Mat & outputImage, int segmentSize = 30, int whiteThreshold = 30) {
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