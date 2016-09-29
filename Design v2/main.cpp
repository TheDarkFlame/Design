#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"
#include <iostream>
#include <vector>
#include <string>
#define classCount 100	//classify 100 signs

using namespace cv;
using namespace std;
using namespace cv::ml;
// Global variables

typedef struct {
	float aspectRatio_min;
	float aspectRatio_max;
	Size size_min = Size(0, 0);
	Size size_max = Size(0, 0);
} mserparams;

int main(int argc, char** argv) {
	//function declarations
	void HOGfeatureExtractor(const Mat& input_image, vector<float> & output_data);
	double extractRegions(const Mat input_image, vector<Mat> & output_images);

	//variable declarations
	Mat src;
	vector<Mat>features;
	string imageName = argv[1];

	//load image
	src = imread(argv[1], 1);
	if (src.empty())
		return -1;
	//extract regions of interest
	double time = extractRegions(src, features);

	//make window to show output
	namedWindow("window 1", CV_WINDOW_AUTOSIZE);

	//HOG feature extraction
	for (vector<Mat>::iterator it = features.begin();it != features.end();it++) {
		//this stuff here needs some work!!!
		vector<float> output;
		HOGfeatureExtractor(*it,output);
		//convert from vector<float> to Mat of type CV_32F
		int row = 1, col = output.size();
		Mat MatOutput(row, col, CV_32F);
		memcpy(&(MatOutput.data[0]), output.data(), col*sizeof(float));
	}
}

double extractRegions(const Mat input_image, vector<Mat> & output_images) {//finished function
	
	//function definitions
	void mserExtractor(const Mat& image, Mat& mserOutMask, vector<Mat>&output_images, mserparams parameters);
	//variables
	int threshold_value = 127;
	static Mat gray_image, thresholded_image, mserMasked_image;
	int64 start;
	mserparams parameters;
	parameters.aspectRatio_max = 1.25;
	parameters.aspectRatio_min = 0.75;
	parameters.size_min = Size(30, 30);
	parameters.size_max = Size(0, 0);
	//start timing
	start = getTickCount();

	//make gray
	cv::cvtColor(input_image, gray_image, CV_BGR2GRAY);

	//image thresholding
	threshold(gray_image, thresholded_image, threshold_value, 255, 3);

	//MSER feature detection
	mserExtractor(thresholded_image, mserMasked_image, output_images, parameters);

	//output the time taken
	cout << "feature extraction : " << (getTickCount() - start) / getTickFrequency() << endl;
	return ((double)(getTickCount() - start)) / getTickFrequency();
}

void mserExtractor(const Mat& image, Mat& mserOutMask, vector<Mat>&output_images, mserparams parameters) {
	Ptr<MSER> mserExtractor = MSER::create();
	vector<vector<cv::Point>> mserContours;
	vector<KeyPoint> mserKeypoint;
	vector<cv::Rect> mserBbox;
	mserExtractor->detectRegions(image, mserContours, mserBbox);

	int i = 0;
	stringstream windowname;
	int prev_x = 0, prev_width = 0, prev_y = 0, prev_height = 0;
	for (std::vector<cv::Rect>::iterator it = mserBbox.begin();it != mserBbox.end();it++) {
		float aspect_ratio = (((float)(*it).height) / ((float)(*it).width));
		//consider only rectangles of appropriate dimensions and that are in the bounds of the original image
		if (//aspect ratio filtering
			aspect_ratio > parameters.aspectRatio_min
			&& aspect_ratio < parameters.aspectRatio_max
			//checking box is within image boundaries
			&& (*it).x + (*it).width < image.size().width
			&& (*it).y + (*it).height < image.size().height
			//minimum size filtering
			&& (*it).width > parameters.size_min.width
			&& (*it).height > parameters.size_min.height
			//maximum size filtering (if param==0, do not check)
			&& ((parameters.size_max.width == 0) || ((*it).width < parameters.size_max.width))
			&& ((parameters.size_max.height == 0) || ((*it).height < parameters.size_max.height))
			//checking for duplicates
			&& !(
				(*it).x == prev_x
				&& (*it).y == prev_y
				&& (*it).width == prev_width
				&& (*it).height == prev_height)
			) {
			//for each image fitting the criteria above, do:
			//add to output array
			output_images.push_back(image(Range((*it).y, (*it).y + (*it).height), Range((*it).x, (*it).x + (*it).width)));
			//draw box onto mask
			rectangle(mserOutMask, *it, Scalar(255, 0, 0), 2, 8, 0);
			//set new previous image
			prev_x = (*it).x;
			prev_y = (*it).y;
			prev_height = (*it).height;
			prev_width = (*it).width;
		}
	}
}

void HOGfeatureExtractor(const Mat& input_image, vector<float> & output_data) {
	Mat resized_image;
	resize(input_image, resized_image, Size(64, 64));//get a standard size for images to work with ( double winsize of HOG descriptor)
	HOGDescriptor hog(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 5);
	//window size=32x32 , we will resize everything to be 64x64
	//block size=16x16 pixels
	//block step size (block stride)=8x8 pixels (so each block starts at 8 pixels after the next
	//cell size =8x8
	//bin count =5
	vector<Point> locations;
	hog.compute(resized_image, output_data, Size(0, 0), Size(0, 0), locations);
}

void Predict(Mat const input, Mat & output, Ptr<ANN_MLP> NeuralNetModel) {
	Mat prediction(0, 1, CV_32F);
	Mat classReponses;
	for (int i = 0; i < input.rows;i++) {
		prediction.push_back(NeuralNetModel->predict(input.row(i)));//for every input make a prediction
	}
	output = prediction;//return our prediction
}
