#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <utility>
#include <opencv2\ml.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\features2d.hpp>
#include <Windows.h>
#include <tuple>
#include <algorithm>
#include <set>

using namespace std;
using namespace cv;
using namespace cv::ml;

typedef struct ExtractionProperties {
	int Threshold_L;//lower threshold (anything below becomes black)
	int Threshold_U;//upper threshold (currently not implemented)
	double Ratio_L;//lower threshold for ratio of width to height of detected regions
	double Ratio_U;//upper threshold for ratio of width to height of detected regions
	int Area_max;//maximum area size of regions
	int Area_min;//minimum area size of regions
	int MatchingPercent;//this is percentage of width that the ROIs edges must match by to be considered duplicates
} ExtractionProperties;

typedef struct ProgramFlags {
	bool Train;//true for training mode, false for prediction
	bool Debug;//debug mode
	bool preDetected;//ROIs are pre-detected (we do not detect them)
} ProgramFlags;

typedef tuple<Mat, string, int, int> imagetuple;//0:Mat=image 1:string=imageName 2:int1=imageNumber 3:int2=classNumber

typedef tuple<Rect, int, int> recttuple;//0:Rect=ROIrectangle 1:int1=associatedImageNumber 2:int2=classNumber

class classes {
private:
	vector<pair<string, int>> class_list;
public:
	classes();
	bool convert(string from_name, int & to_number);
	bool convert(int from_number, string & to_name);
	bool initialize(string directory);
	int size();
};

class TrainingAnswers {
private:
	vector<int> answers;
	string filename;
	bool answersPreexist;
public:
	TrainingAnswers() {};
	~TrainingAnswers();
	bool initialize(string directory);
	int getAnswer(int index);
	void pushAnswer(int answer);
	void writeOut();
	bool answersExist();
	int size();
};

//new globals for review bit
vector<imagetuple> images;
classes class_list;

int main(int argc, char** argv) {
	//@
	//function declarations
	//@
	void getFileNames(vector<string> & filenames, string directory, list<string> extension);
	double ExtractROIs(Mat inputImage, vector<Rect> & ROIs, ExtractionProperties & props);
	void TrainingFilterROIs(vector<imagetuple> images, vector<recttuple> & rectROIs, classes class_list, ProgramFlags ProgFlags, string answerFileDirectory);
	double ExtractFeatures(Mat ROI, Mat & outputRow);
	void createTrainingResponse(vector<imagetuple> inputROIs, int numClasses, Mat & responses_1D, Mat & responses_2D);
	void setLayerSizes(string input, int firstLayer, int LastLayer, vector<int> & output);
	double trainNetwork(const Mat Tdata, const Mat Tresponses, string networkDir, vector<int> layerSizes, bool newNetwork);
	bool Predict(Mat data, Mat & responses1D, Mat & responses2D, string networkDir);
	Mat createResponseMat(vector<imagetuple> responses, int classCount, int imageCount);
	void drawRegions(int, void*);

	//@
	//extract all program options
	//@
	string keys =
		"{networkpath|network.xml|the network save location}"
		"{imagepath|./images|default image directory}"
		"{train||include this if you want to do training}"
		"{layers|50|the number of internal layers for the ANN}"
		"{classes|classes.txt|a text file listing all the class names}"
		"{debug||enable debug mode, this will automate answering training questions}"
		"{skipROI||skips region detection, uses the images as ROIs}"
		"{answers|answers.txt|training answers, only used in debug mode}"
		"{h,help||help}"
		;
	string help =
		"enter arguments as follows: -<arg name>=<arg val>\n"
		"enter flags as follows: -<flag name>\n"
		"arguments list: <argument name> : <argument description> : <default>\n"
		"imagepath : the directory from where images will be read : ./images\n"
		"layers : hidden layer configuration for neural network, for use in training mode : 50\n"
		"classes : path to the classes file that contains a list of all classes for use : classes.txt\n"
		"answers : the path to an answers file for ROI questions (debug mode only) : answers.txt\n"
		"flags list: <flag name> : <flag description>\n"
		"train : puts the system into training mode, else it will behave in prediction mode\n"
		"skipROI : uses images as ROIs, skipping ROI detection (debug mode only)\n"
		"debug : puts system into debug mode\n"
		;
	CommandLineParser parser(argc, argv, keys);
	
	if (argc == 1) {
		cout << "you have not entered any arguments" << endl << "displaying help and using argument defaults" << endl;
		cout << help;
	}

	//initialize extraction properties
	ExtractionProperties ExProps;
	ExProps.Area_min = 20 * 20;
	ExProps.Area_max = 0;//no max limit
	ExProps.Ratio_L = 0.8;
	ExProps.Ratio_U = 1.25;
	ExProps.MatchingPercent = 25;
	ExProps.Threshold_L = 70;
	ExProps.Threshold_U = 200;

	//initialize program flags
	ProgramFlags ProgFlags;
	ProgFlags.Debug = (parser.has("debug"));
	ProgFlags.Train = (parser.has("train"));
	ProgFlags.preDetected = (parser.has("skipROI"));

	//initialize class list
	//classes class_list;//this is now global
	string classfile = parser.get<string>("classes");
	if (!class_list.initialize(classfile)) {
		cout << "failed to initialize classes, please ensure that a valid class file " + classfile + " exists";
		system("pause");
	}
	int class_count = class_list.size();

	cout << "Program modes:" << endl << "training = " << (ProgFlags.Train ? "true" : "false")
		<< endl << "debug = " << (ProgFlags.Debug ? "true" : "false")
		<< endl << "skip ROI detection = " << (ProgFlags.preDetected ? "true" : "false") << endl << endl;
	//@
	//begin the program
	//@
	double tick = getTickCount();
	double timeTaken;

	//load the images
	string imagePath = parser.get<string>("imagepath");
	vector<string>imageNames;
	list<string> extension = { "png","jpg" };
	getFileNames(imageNames, imagePath, extension);



	//image number should just be the index number for most situations
	//vector<imagetuple> images; //this is global now

	for (int imageNumber = 0;imageNumber < imageNames.size();imageNumber++) {
		Mat temp = imread(imagePath + "/" + imageNames[imageNumber], IMREAD_COLOR);
		int imageClass;
		//push back the class this image is identified as
		class_list.convert(imageNames[imageNumber], imageClass);
		images.push_back(make_tuple(temp, imageNames[imageNumber], imageNumber, imageClass));
	}
	cout << "read in " << images.size() << " images for evaluation" << endl;

	vector<imagetuple>ROIs;//ROIs
	vector<recttuple>rectROIs;
	if (ProgFlags.Debug && ProgFlags.preDetected) {//colour segment the ROIs like a normal ROI would be
		for (vector<imagetuple> ::iterator it = images.begin();it != images.end();it++) {
			Mat temp;
			cvtColor(get<0>(*it), temp, CV_BGR2GRAY);//push back every image as a grayscale
			ROIs.push_back(make_tuple(temp, get<1>(*it), get<2>(*it), get<3>(*it)));
		}
	}
	else {//else perform extraction of ROIs
		cout << "ROI distribution as follows:" << endl << "<image> <ROIs in image>" << endl;
		timeTaken = 0;
		
		//vector<recttuple>rectROIs; this is now used later, so is no longer specific to this only
		for (vector<imagetuple>::iterator it_img = images.begin();it_img != images.end();it_img++) {
			vector<Rect>temp;
			timeTaken += ExtractROIs(get<0>(*it_img), temp, ExProps);
			cout << get<2>(*it_img) << ". " << get<1>(*it_img) << " : " << temp.size() << endl;
			for (vector<Rect>::iterator it_rect = temp.begin();it_rect != temp.end();it_rect++)
				rectROIs.push_back(make_tuple(*it_rect, get<2>(*it_img), get<3>(*it_img)));//push back the rect and the associated image number
		}
		cout << "ROI extraction complete in " << timeTaken << " seconds" << endl << endl;

		if (ProgFlags.Train && ProgFlags.preDetected == false)
			TrainingFilterROIs(images, rectROIs, class_list, ProgFlags, parser.get<string>("answers"));//mark incorrect ROIs with class=-1 (only if program is detecting ROIs)

		for (vector<recttuple>::iterator it_rect = rectROIs.begin();it_rect != rectROIs.end();it_rect++) {
			//populate a vector containing all the ROIs we just extracted
			int baseImageNumber = get<1>(*it_rect);
			Rect baseRectangle = get<0>(*it_rect);

			Mat region_entire;
			cvtColor(get<0>(images[baseImageNumber]), region_entire, CV_BGR2GRAY);//check this line in other areas
			Mat region_rect = region_entire(
				Range(baseRectangle.y, baseRectangle.y + baseRectangle.height),
				Range(baseRectangle.x, baseRectangle.x + baseRectangle.width));
			ROIs.push_back(make_tuple(region_rect, get<1>(images[baseImageNumber]), get<2>(images[baseImageNumber]), get<3>(images[baseImageNumber])));
		}
	}

	//we now have all the ROIs we want to analyze, perform extraction of feature descriptors
	cout << endl << "extracting features" << endl;

	timeTaken = 0;
	Mat imageFeatures;
	for (vector<imagetuple>::iterator it_img = ROIs.begin();it_img != ROIs.end();it_img++) {
		Mat row;
		timeTaken += ExtractFeatures(get<0>(*it_img), row);
		imageFeatures.push_back(row);
	}
	cout << "descriptor extraction for " << ROIs.size() << " ROIs complete in " << timeTaken << " seconds" << endl << endl;

	//learning algorithms
	string networkpath = parser.get<string>("networkpath");

	if (ProgFlags.Train == true) {//training mode
		//obtain the output classes
		Mat responses1D, responses2D;
		createTrainingResponse(ROIs, class_list.size(), responses1D, responses2D);//creates the expected responses

		//now set up the neural network
		string layerConfig = parser.get<string>("layers");
		vector<int>layerSizes;
		setLayerSizes(layerConfig, imageFeatures.cols, class_count, layerSizes);
		cout << "beginning training" << endl;
		timeTaken = trainNetwork(imageFeatures, responses2D, networkpath, layerSizes, true);
		cout << "training complete in " << timeTaken << " seconds" << endl;
	}
	else {//else if not training mode, predict using the network
		Mat responses_MostLikelyClass, responses_ValuePerClass;
		vector<imagetuple> responses;
		int64 Start = getTickCount();

		if (!Predict(imageFeatures, responses_MostLikelyClass, responses_ValuePerClass, networkpath))
			return -1;//if we cannot load the network return -1
		for (int i = 0;i < responses_MostLikelyClass.rows;i++) {
			int classNumber = (int)responses_MostLikelyClass.at<float>(i, 0);
			Mat rawValue = responses_ValuePerClass.row(i);
			responses.push_back(make_tuple(rawValue, get<1>(ROIs[i]), get<2>(ROIs[i]), classNumber));//responses and the associated image
		}

		cout << "prediction complete in : " << (getTickCount() - Start) / getTickFrequency() << " seconds" << endl;

		Mat results_actual = createResponseMat(images, class_list.size(), (int)images.size());
		Mat results_predicted = createResponseMat(responses, class_list.size(), (int)images.size());

		cout << endl << "classifications for each ROI, in format as follows:";
		cout << endl << "<base imagename> <classification>" << endl;
		for (vector<imagetuple>::iterator it = responses.begin();it != responses.end();it++) {
			string prediction;
			class_list.convert(get<3>(*it),prediction);
			cout << get<1>(*it) << " " << prediction << endl;
		}
		if (!ProgFlags.preDetected) {

		//make some sort of visual output
			vector<recttuple>resultingRects = rectROIs;
			for (int i = 0;i < resultingRects.size();i++) {
				get<2>(resultingRects[i]) = get<3>(responses[i]);
			}

			int threshold_value = 0;
			namedWindow("Review Window");
			createTrackbar("Image Select", "Review Window", &threshold_value, images.size() - 1, drawRegions, (void*)(&resultingRects));
			drawRegions(0, (void*)(&resultingRects));
			
			while (true)
			{
				static int i = 1;
				int c;
				c = waitKey(20);
				if ((char)c == 27)
				{
					break;
				}
			}
		}

	}

	cout << "total program runtime is : " << (getTickCount() - tick) / getTickFrequency() << " seconds" << endl;

	system("pause");
	return 0;
}

void drawRegions(int trackpos, void* data) {
	vector<recttuple>rects = *((vector<recttuple>*)data);
	Mat dispImage;
	int num_classes = class_list.size();
	(get<0>(images[trackpos])).copyTo(dispImage);//set the correct image for displaying
	for (vector<recttuple>::iterator it = rects.begin();it != rects.end();it++) {
		if (get<1>(*it) == trackpos) {
			int class_number = get<2>(*it);
			Scalar colour;
			switch (class_number) {
			case 1: colour = Scalar(255, 0, 0);
				break;
			case 2: colour = Scalar(0, 0, 255);
				break;
			case 3: colour = Scalar(0, 255, 0);
			default: colour = Scalar(255, 255, 255);
			}
			rectangle(dispImage, get<0>(*it), colour);
		}
	}
	string temp;
	class_list.convert(1, temp);
	putText(dispImage, temp, Point(5, dispImage.rows - 5), FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255, 0, 0));
	class_list.convert(2, temp);
	putText(dispImage, temp, Point(5, dispImage.rows - 15), FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(0, 0, 255));
	class_list.convert(3, temp);
	putText(dispImage, temp, Point(5, dispImage.rows - 25), FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(0, 255, 0));
	class_list.convert(0, temp);
	putText(dispImage, temp, Point(5, dispImage.rows - 35), FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255, 255, 255));
	imshow("Review Window", dispImage);
}

bool Predict(Mat data, Mat & responses1D, Mat & responses2D, string networkDir) {
	Ptr<ANN_MLP> network = StatModel::load<ANN_MLP>(networkDir);
	if (network.empty())
		return false;

	Mat prediction1D(0, 1, CV_32F), resultRow, prediction2D;//this is the NN's prediction
	for (int i = 0;i < data.rows;i++) {
		prediction1D.push_back(network->predict(data.row(i), resultRow));
		prediction2D.push_back(resultRow);
	}
	responses1D = prediction1D;
	responses2D = prediction2D;
	return true;
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

double trainNetwork(const Mat Tdata, const Mat Tresponses, string networkDir, vector<int> layerSizes, bool newNetwork) {//update indicates network is updating, instead of creating
																														//this function creates or updates an ANN and returns time taken
	int64 start = getTickCount();
	TermCriteria TC(int, double);
	Ptr<TrainData> TrainingData = TrainData::create(Tdata, ROW_SAMPLE, Tresponses);
	Ptr<ANN_MLP> network = StatModel::load<ANN_MLP>(networkDir);
	int flag;
	if (newNetwork) {
		flag = 0;
		network = ANN_MLP::create();
		network->setLayerSizes(layerSizes);
		network->setTermCriteria(TC(1000, 0));
		network->setTrainMethod(ANN_MLP::BACKPROP);
		network->setActivationFunction(ANN_MLP::SIGMOID_SYM);
	}
	else {
		flag = 1;
	}
	network->train(TrainingData, flag);
	network->save(networkDir);
	return(getTickCount() - start) / getTickFrequency();
}

void setLayerSizes(string input, int firstLayer, int LastLayer, vector<int> & output) {//generates layer sizes from a string
																					   //http://stackoverflow.com/questions/24504582/how-to-test-whether-stringstream-operator-has-parsed-a-bad-type-and-skip-it
	output.push_back(firstLayer);//insert the input layer

	stringstream temp(input);//extract from a string and input all the other layers
	int currentLayerSize;
	while (temp >> currentLayerSize || !temp.eof()) {
		if (temp.fail()) {
			temp.clear();
			string trash;
			temp >> trash;//if we get something that isn't an int, put it in trash and continue
			continue;
		}
		output.push_back(currentLayerSize);
	}

	output.push_back(LastLayer);//insert the output layer
}

void SegmentImage(Mat &inputImage, vector<Mat> &outputImages, int lower_threshold) {//segments image based on colour
	//function declarations
	void HSVsegment(Mat inputImage, Mat & outputImage, int segmentSize = 30, int whiteThreshold = 30);
	//variable definitions
	Mat temp;
	cvtColor(inputImage, temp, CV_BGR2GRAY);
	
	//function definition
	//threshold the image
	threshold(temp, temp, lower_threshold, 255, ThresholdTypes::THRESH_TOZERO);//perhaps try an adaptive thresholding rule too later
	outputImages.push_back(temp);

	//HSV segment
	HSVsegment(inputImage, temp);
	outputImages.push_back(temp);
}

double ExtractROIs(Mat inputImage, vector<Rect> & ROIs, ExtractionProperties & props) {
	//function declarations
	void SegmentImage(Mat &inputImage, vector<Mat> &outputImages, int lower_threshold);
	double mserExtractor(Mat inputImage, vector<Rect> & Rectangles, ExtractionProperties & props);
	
	//variable declarations
	vector<Mat> segmentedImages;
	
	//function definition
	int64 start = getTickCount();
	SegmentImage(inputImage, segmentedImages, props.Threshold_L);//spectral segmentation of image

	//extract ROIs using MSER
	for (vector<Mat>::iterator it = segmentedImages.begin();it != segmentedImages.end();it++)
		mserExtractor(*it, ROIs, props);

	return (getTickCount() - start) / getTickFrequency();
}

void TrainingFilterROIs(vector<imagetuple> images, vector<recttuple> & rectROIs, classes class_list, ProgramFlags ProgFlags, string answerFileDirectory) {
	//ask user about each ROI or find it all from a textfile
	enum answerTypes{YES,NO,OTHER};
	TrainingAnswers answers;
	bool FilterManually = true;
	if (ProgFlags.Debug) {
		answers.initialize(answerFileDirectory);
		if (answers.answersExist() == false) {//if answers do not exist we need to manually filter
			FilterManually = true;
		}
		if (answers.size() == rectROIs.size()) {//if the answers size does not match up, filter manually
			FilterManually = true;
		}
	}

	if (!FilterManually) {//if we already have a filter set up, extract answers from the file
		cout << "automatically filtering ROIs for training according to previous choices" << endl;
		int i = 0;
		for (vector<recttuple>::iterator it = rectROIs.begin();it != rectROIs.end();it++) {
			if (answers.getAnswer(i) == NO) {//if not correct classification
				get<2>(*it) = 0;//"none" classification
			}
			else if (answers.getAnswer(i) == OTHER) {//if it is a sign, but the wrong sign, then remove it
				vector<recttuple>::iterator temp = it;
				it++;
				rectROIs.erase(temp);
			}
			i++;
		}
	}
	else {//else manually answer each one if no file is set up or if not in debug
		Mat BlankImage = Mat::zeros(Size(1, 1), CV_8UC3);
		namedWindow("Confirmation Window");
		for (vector<recttuple>::iterator it = rectROIs.begin();it != rectROIs.end();it++) {
			Mat DisplayImage;
			(get<0>(images[get<1>(*it)])).copyTo(DisplayImage);//the image corresponding to the associated image number

			string class_name;
			class_list.convert(get<2>(*it), class_name);
			putText(DisplayImage, class_name + "(y)es/(n)o/(o)ther sign", Point(5, DisplayImage.rows-5), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(0, 0, 255), 2);

			rectangle(DisplayImage, get<0>(*it), Scalar(255, 0, 0), 2);//draw rectangle
			//imshow("Confirmation Window", BlankImage);
			imshow("Confirmation Window", DisplayImage);

			int	key;
			do {
				key = waitKey(0);
			} while (!((key == 'y') || (key == 'n') || (key == 'o')));
			if (key == 'n') {
				get<2>(*it) = 0;
				if (ProgFlags.Debug) answers.pushAnswer(NO);
			}
			else if (key == 'o') {
				vector<recttuple>::iterator temp = it;
				it--;
				rectROIs.erase(temp);// our new next is same as if there wasn't temp there.
				if (ProgFlags.Debug) answers.pushAnswer(OTHER);
			}
			else {
				if (ProgFlags.Debug) answers.pushAnswer(YES);
			}
		}
	}
	destroyWindow("Confirmation Window");
	cout << "filtering complete" << endl;
	if (ProgFlags.Debug) {
		answers.writeOut();
		cout << "answers saved for future automation" << endl;
	}
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

double ExtractFeatures(Mat ROI, Mat & outputRow) {//input a Mat, output a row of features.
	//function declarations
	double hogExtractor(const Mat input_image, Mat & outputRow);
	
	//functiona definition
	int64 start = getTickCount();

	outputRow = Mat(0, 0, CV_32F);

	//extract features using a HOG
	hogExtractor(ROI, outputRow);
	return (getTickCount() - start) / getTickFrequency();
}

double hogExtractor(const Mat input_image, Mat & outputRow) {
	int64 start = getTickCount();
	Mat resized_image;
	vector<float> results;
	resize(input_image, resized_image, Size(64, 64));//get a standard size for images to work with ( double winsize of HOG descriptor)
	HOGDescriptor hog(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 5);
	//window size=32x32 , we will resize everything to be 64x64
	//block size=16x16 pixels
	//block step size (block stride)=8x8 pixels (so each block starts at 8 pixels after the next
	//cell size =8x8
	//bin count =5
	vector<Point> locations;
	hog.compute(resized_image, results, Size(0, 0), Size(0, 0), locations);

	//get output to be in a Mat
	Mat resultRow = Mat(1, (int)results.size(), CV_32F);
	memcpy(resultRow.data, results.data(), results.size()*sizeof(float));
	outputRow = resultRow;

	return (getTickCount() - start) / getTickFrequency();
}

void getFileNames(vector<string> & filenames, string directory, list<string> extension) {
	for(list<string>::iterator it_ext = extension.begin();it_ext != extension.end();it_ext++) {

		std::string search_path = directory + "/*." + *it_ext;
		WIN32_FIND_DATA fd;
		HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);

		if (hFind != INVALID_HANDLE_VALUE)
		{
			do
			{
				// read all (real) files in current folder
				// , delete '!' read other 2 default folder . and ..
				if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				{
					filenames.push_back(fd.cFileName);
				}
			} while (::FindNextFile(hFind, &fd));
			::FindClose(hFind);
		}
	}
}

void createTrainingResponse(vector<imagetuple> inputROIs, int numClasses, Mat & responses_1D, Mat & responses_2D) {//reads images as well as creates training responses
	for (vector<imagetuple>::iterator it = inputROIs.begin();it != inputROIs.end();it++) {
		Mat Row = Mat::zeros(1, numClasses, CV_32F);
		if (get<3>(*it) != 0)
			Row.at<float>(0, get<3>(*it)) = 1.f;
		responses_1D.push_back((float)get<3>(*it));
		responses_2D.push_back(Row);
	}
}

classes::classes() {

};

bool classes::convert(string from_name, int & to_number) {
	to_number = 0;// -1 indicates no match, is changed to something if a match is found
	if (class_list.empty())
		return false;
	static pair<string, int> last_member = class_list.front();

	size_t found = from_name.find(last_member.first);
	if (found != string::npos) {//check if the string is the same as the last request
		to_number = last_member.second;
		return true;
	}
	else//else search the entire class collection
		for (vector<pair<string, int>>::iterator it = class_list.begin();it != class_list.end();it++) {
			found = from_name.find((*it).first);
			if (found != string::npos) {
				to_number = (*it).second;
				last_member = (*it);
				return true;
			}
		}
	return false;
};

bool classes::convert(int from_number, string & to_name) {
	to_name = "none";//default is no class, this is changed if a class is found

	if (from_number == -1)//if -1, return true, class is none
		return true;
	if (class_list.empty())//if empty, return false
		return false;
	if (from_number > class_list.size()) {//if too high, return false, class is not recognized
		"unrecognized class";
		return false;
	}

	to_name = (class_list[from_number]).first;//else find the appropriate class
	return true;
};

bool classes::initialize(string directory) {//reads from a file and creates a list of all classes
	ifstream file(directory);
	if (!file)
		return false;

	class_list.push_back(make_pair("none", 0));

	string  buf;
	int number = 1;//0 is the "none" entry
	while (getline(file, buf)) {
		class_list.push_back(make_pair(buf, number++));
	}
	return true;
};

int classes::size() {
	return (int)class_list.size();
}

bool TrainingAnswers::answersExist() {
	return answersPreexist;
}

bool TrainingAnswers::initialize(string directory) {
	filename = directory;
	ifstream file(filename);
	if (!file) {
		answersPreexist = false;
		return false;
	}

	string buf;
	while (getline(file, buf)) {
		if (buf[0] == '1')
			answers.push_back(true);
		else
			answers.push_back(false);
	}
	answersPreexist = true;
	return true;
}

int TrainingAnswers::getAnswer(int index) {
	return answers[index];
}

void TrainingAnswers::pushAnswer(int answer) {
	answers.push_back(answer);
}

void TrainingAnswers::writeOut() {
	ofstream file(filename);
	for (vector<int>::iterator it = answers.begin();it != answers.end();it++) {
		file << *it << endl;
	}
	file.close();
}

int TrainingAnswers::size() {
	return (int)answers.size();
}

TrainingAnswers::~TrainingAnswers() {
	if (!answersPreexist)
		writeOut();
}

void HSVsegment(Mat inputImage, Mat & outputImage, int segmentSize = 30, int whiteThreshold = 30) {
	static int prevSegmentSize = 0;
	static Mat lut(1, 256, CV_8UC3);
	if (prevSegmentSize != segmentSize) {
		for (int i = 0;i < 256;i++) {
			int extraHue = i % segmentSize;

			if (i + segmentSize - extraHue >= 180)
				lut.at<Vec3b>(i)[0] = 0;//H channel
			else if (extraHue < segmentSize / 2)
				lut.at<Vec3b>(i)[0] = i - extraHue;//H channel
			else
				lut.at<Vec3b>(i)[0] = i - extraHue + segmentSize;//H channel

			if (i < whiteThreshold)
				lut.at<Vec3b>(i)[1] = 0;//S channel
			else
				lut.at<Vec3b>(i)[1] = i;//S channel

			lut.at<Vec3b>(i)[2] = 127;//V channel
		}
		prevSegmentSize = segmentSize;
	}

	cvtColor(inputImage, outputImage, CV_BGR2HSV);
	LUT(outputImage, lut, outputImage);
	cvtColor(outputImage, outputImage, CV_HSV2BGR);
}

Mat createResponseMat(vector<imagetuple> responses, int classCount, int imageCount) {
	Mat output(imageCount, classCount, CV_8U);
	vector<bool> rowTracker;//tracks the number of non-"none" entries per a row(image)
	for (int i = 0; i < imageCount;i++) {
		rowTracker.push_back(false);
	}

	for (int i = 0;i < responses.size();i++) {//for each ROI
		//int classNumber = get<3>(responses[i])	(col)
		//int imageNumber = get<2>(responses[i])	(row)
		if (get<3>(responses[i]) != 0) {//if not a "none" class
			output.at<uchar>(get<2>(responses[i]), get<3>(responses[i])) = 1.f;//rows=image cols=response
			vector<bool>::iterator it = rowTracker.begin() + i;
			rowTracker[get<2>(responses[i])] = true;
		}
	}
	for (int i = 0;i < rowTracker.size();i++) {
		if (rowTracker[i] == false)
			output.at<uchar>(i, 0) = 1;
	}
	return output;
}