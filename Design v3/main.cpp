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

using namespace std;
using namespace cv;
using namespace cv::ml;

typedef struct ExtractionProperties {
	int Threshold_L;//lower threshold (anything below becomes black)
	int Threshold_U;//upper threshold (currently not implemented)
	float Ratio_L;//lower threshold for ratio of width to height of detected regions
	float Ratio_U;//upper threshold for ratio of width to height of detected regions
	int Area_max;//maximum area size of regions
	int Area_min;//minimum area size of regions
	int MatchingFactor;//ROIs are detected as the same if (new-old) < (new / MatchingFactor) <higher values=more duplicates>
} ExtractionProperties;

typedef struct ProgramFlags {
	bool Train;//true for training mode, false for prediction
	bool Debug;//debug mode
	bool preDetected;//ROIs are pre-detected (we do not detect them)
} ProgramFlags;

typedef struct metrics {
	float FPrate;
	float TPrate;
	float Precision;
	float Accuracy;
	float Fscore;
} metrics;

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
	vector<bool> answers;
	string filename;
	bool answersPreexist;
public:
	TrainingAnswers() {};
	~TrainingAnswers();
	bool initialize(string directory);
	bool getAnswer(int index);
	void pushAnswer(bool answer);
	void writeOut();
	bool answersExist();
	int size();
};

int main(int argc, char** argv) {
	//@
	//function declarations
	//@
	void getFileNames(vector<string> & filenames, string directory, string extension);
	double ExtractROIs(pair<Mat, int> inputImage, list<pair<Mat, int>> & ROIs, ExtractionProperties props);
	double ExtractFeatures(list<pair<Mat, int>> ROIs, Mat & outputFeatures);
	void setLayerSizes(string input, int firstLayer, int LastLayer, vector<int> & output);
	void createTrainingResponse(list<pair<Mat, int>> inputROIs, int numClasses, Mat & responses_1D, Mat & responses_2D);
	double trainNetwork(const Mat Tdata, const Mat Tresponses, string networkDir, vector<int> layerSizes, bool newNetwork);
	void TrainingFilterROIs(list<pair<Mat, int>> & ROIs, ProgramFlags ProgFlags, string answerFileDirectory);
	bool Predict(Mat data, Mat & responses1D, Mat & responses2D, string networkDir);
	void createConfusionMatrix(Mat results_predicted, Mat results_actual, Mat & confusionMatrix, int matrixClassCount, int & totalEntries);
	Mat generateROCgraph(vector<metrics> input);
	void SegmentImage(Mat &inputImage, list<Mat> &outputImages, int lower_threshold);
	//@
	//extract all program options
	//@
	string keys =
		"{train||set this to indicate training mode}"
		"{networkpath|network.xml|the network save location}"
		"{imagepath|./images|default image directory}"
		"{train||set this if you want to do training}"
		"{layers|50|the number of internal layers for the ANN}"
		"{classes|classes.txt|a text file listing all the class names}"
		"{ROIdir|ROIs|a directory for all regions of interest during for AUTO mode}"
		"{debug||enable debug mode, this will automate answering training questions}"
		"{skipROI||skips region detection, uses the images as ROIs}"
		"{answers|answers.txt|training answers, only used in debug mode}"
		;

	CommandLineParser parser(argc, argv, keys);

	//initialize extraction properties
	ExtractionProperties ExProps;
	ExProps.Area_min = 20 * 20;
	ExProps.Area_max = 0;//no max limit
	ExProps.Ratio_L = 0.6f;
	ExProps.Ratio_U = 1.4f;
	ExProps.MatchingFactor = 20;
	ExProps.Threshold_L = 127;
	ExProps.Threshold_U = 200;

	//initialize program flags
	ProgramFlags ProgFlags;
	ProgFlags.Debug = (parser.has("debug"));
	ProgFlags.Train = (parser.has("train"));
	ProgFlags.preDetected = (parser.has("skipROI"));

	//initialize class list
	classes class_list;
	string classfile = parser.get<string>("classes");
	if (!class_list.initialize(classfile)) {
		cout << "failed to initialize classes, please ensure that a valid class file " + classfile + " exists";
		return 1;
	}
	int class_count = class_list.size();

	cout << "Program modes:" << endl << "training = " << ProgFlags.Train << endl << "debug = " << ProgFlags.Debug << endl << endl;
	//@
	//begin the program
	//@
	double timeTaken;

	//load the images in given directory
	string imagePath = parser.get<string>("imagepath");
	vector<string>imageNames;
	string extension = "png";
	getFileNames(imageNames, imagePath, extension);
	
	//in training mode int=class number, in prediction mode int=image identification number
	vector<pair<Mat, int>> trainingImages;
	
	Mat temp;
	int imageClass, i = 0;
	for (vector<string>::iterator it = imageNames.begin();it != imageNames.end();it++) {
		temp = imread(imagePath + "/" + (*it), IMREAD_COLOR);
		if (ProgFlags.Train) {
			//push back the class this image is identified as
			class_list.convert(*it, imageClass);
			trainingImages.push_back(make_pair(temp, imageClass));
		}
		else//if Prediction mode
		{
			trainingImages.push_back(make_pair(temp, i++));//tracks which image number it is
		}
	}
	cout << "read in " << trainingImages.size() << " images for evaluation" << endl;

	list<pair<Mat, int>>ROIs;//ROIs and classes
	if (ProgFlags.Debug && ProgFlags.preDetected) {//colour segment the ROIs like a normal ROI would be
		for (vector < pair < Mat, int >> ::iterator it = trainingImages.begin();it != trainingImages.end();it++) {
			list<Mat>outputSegments;
			SegmentImage((*it).first, outputSegments, ExProps.Threshold_L);
			for (list<Mat>::iterator it2 = outputSegments.begin();it2 != outputSegments.end();it2++) {
				ROIs.push_back(make_pair(*it2, (*it).second));
			}
		}
	}
	else {//else perform extraction of ROIs
		cout << "ROI distribution as follows:" << endl << "<image> <ROIs in image>" << endl;
		int lastSize = 0, counter = 0;
		timeTaken = 0;
		for (vector<pair<Mat, int>>::iterator it = trainingImages.begin();it != trainingImages.end();it++) {
			timeTaken += ExtractROIs(*it, ROIs, ExProps);
			cout << counter++ << " " << ROIs.size() - lastSize << endl;
			lastSize = (int)ROIs.size();
		}
		cout << "ROI extraction complete in " << timeTaken << " seconds" << endl << endl;
	}

	//we now have all the features we want to analyze, perform extraction of features
	cout << endl << "extracting features" << endl;
	Mat imageFeatures;
	timeTaken = ExtractFeatures(ROIs, imageFeatures);
	cout << "extraction complete in " << timeTaken << " seconds" << endl << endl;
	
	string networkpath = parser.get<string>("networkpath");
	
	if (ProgFlags.Train) {//if training mode, train the network
		if (ProgFlags.preDetected == false)
			TrainingFilterROIs(ROIs, ProgFlags, parser.get<string>("answers"));//mark incorrect ROIs with class=-1 (only if program is detecting ROIs)
	
		//obtain the output classes
		Mat responses1D, responses2D;
		createTrainingResponse(ROIs, class_list.size(), responses1D, responses2D);

		//now set up the neural network
		string layerConfig = parser.get<string>("layers");
		vector<int>layerSizes;
		setLayerSizes(layerConfig, imageFeatures.cols, class_count, layerSizes);
		cout << "begining training" << endl;
		timeTaken = trainNetwork(imageFeatures, responses2D, networkpath, layerSizes, true);
		cout << "training complete in " << timeTaken << " seconds" << endl;
	}
	else {//else if not training mode, predict using the network
		Mat responses_MostLikelyClass, responses_ValuePerClass;
		int64 Start = getTickCount();
		if (!Predict(imageFeatures, responses_MostLikelyClass,responses_ValuePerClass, networkpath))
			return -1;//if we cannot load the network return -1
		cout << "prediction complete in : " << (getTickCount() - Start) / getTickFrequency() << " seconds" << endl;
		list<pair<Mat, int>>::iterator ROIit = ROIs.begin();
		for (int i = 0;i < responses_MostLikelyClass.rows;i++) {
			int ImageNumber = (*(ROIit++)).second, classNumber;
			classNumber = (int)responses_MostLikelyClass.at<float>(i, 0);
			string classification;
			if (class_list.convert(classNumber, classification)) {
				cout << "Image : " << ImageNumber << " | Classification : " << classification <<
					" | Value : " << responses_ValuePerClass.at<float>(i,classNumber) << endl;
			}
		}
	}

	system("pause");
	return 0;
}

bool Predict(Mat data, Mat & responses1D, Mat & responses2D, string networkDir) {
	Ptr<ANN_MLP> network = StatModel::load<ANN_MLP>(networkDir);
	if (network.empty())
		return false;

	Mat prediction1D(0, 1, CV_32F), resultRow, prediction2D;//this is the NN's prediction
	for (int i = 0;i < data.rows;i++) {
		prediction1D.push_back(network->predict(data.row(i),resultRow));
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

void SegmentImage(Mat &inputImage, list<Mat> &outputImages, int lower_threshold) {//segments image based on colour (basic segmentation= thresholding)
	//to one channel
	void HSVsegment(Mat inputImage, Mat & outputImage, int segmentSize = 30, int whiteThreshold = 30);
	Mat temp;
	cvtColor(inputImage, temp, CV_BGR2GRAY);

	//threshold the image
	threshold(temp, temp, lower_threshold, 255, ThresholdTypes::THRESH_TOZERO);//perhaps try an adaptive thresholding rule too later
	outputImages.push_back(temp);
	//HSV segment
	HSVsegment(inputImage, temp);
	outputImages.push_back(temp);	
}

double ExtractROIs(pair<Mat,int> inputImage, list<pair<Mat,int>> & ROIs, ExtractionProperties props) {
	int64 start = getTickCount();
	void SegmentImage(Mat &inputImage, list<Mat> &outputImages, int lower_threshold);
	double mserExtractor(pair<Mat, int> inputImage, list<pair<Mat, int>> & outputImages, ExtractionProperties props);
	list<Mat> segmentedImages;
	SegmentImage(inputImage.first, segmentedImages, props.Threshold_L);//spectral segmentation
	
	//extract ROIs using MSER
	for (list<Mat>::iterator it = segmentedImages.begin();it != segmentedImages.end();it++) {
		Mat debug = *it;
		mserExtractor(make_pair(*it, inputImage.second), ROIs, props);
	}

	return (getTickCount() - start) / getTickFrequency();
}

void TrainingFilterROIs(list<pair<Mat, int>> & ROIs, ProgramFlags ProgFlags, string answerFileDirectory) {
	//ask user about each ROI or find it all from a textfile
	TrainingAnswers answers;
	bool FilterManually = true;
	if (ProgFlags.Debug) {
		answers.initialize(answerFileDirectory);
		if (answers.answersExist() == false) {//if answers do not exist we need to manually filter
			FilterManually = true;
		}
		if (answers.size() == ROIs.size()) {//if the answers size does not match up, filter manually
			FilterManually = true;
		}
	}

	if (!FilterManually) {//if we already have a filter set up, extract answers from the file
		cout << "automatically filtering ROIs for training according to previous choices" << endl;
		int i = 0;
		for (list<pair<Mat, int>>::iterator it = ROIs.begin();it != ROIs.end();it++){
			if (answers.getAnswer(i++) == false) {//if not correct classification
				(*it).second = -1;
			}
		}
	}
	else {//else manually answer each one if no file is set up or if not in debug
		Mat BlankImage = Mat::zeros(Size(1, 1), CV_8UC3);
		cout << "for each of the following images, please enter (y/n) appropriately, y=a sign(that we want), n=not a sign (or not one we want)";
		namedWindow("Confirmation Window");
		for (list<pair<Mat, int>>::iterator it = ROIs.begin();it != ROIs.end();it++) {
			int key = 0;
			do {
				imshow("Confirmation Window", BlankImage);
				imshow("Confirmation Window", (*it).first);
				key = waitKey(0);
			} while (!((key == 'y') || (key == 'n')));
			if (key == 'n') {
				(*it).second = -1;
				if (ProgFlags.Debug) answers.pushAnswer(false);
			}
			else {
				if (ProgFlags.Debug) answers.pushAnswer(true);
			}
		}
		destroyWindow("Confirmation Window");
		cout << "filtering complete" << endl;
		if (ProgFlags.Debug) { 
			answers.writeOut();
			cout << "answers saved for future automation" << endl;
		}
	}
}

double ExtractFeatures(list<pair<Mat,int>> ROIs, Mat & outputFeatures) {//input a single channel  image
	int64 start = getTickCount();

	outputFeatures = Mat(0, 0, CV_32F);
	double hogExtractor(const Mat input_image, Mat & outputRow);
	//extract features using a HOG
	Mat outputData, row;
	int i = 0;
	for (list<pair<Mat,int>>::iterator it = ROIs.begin();it != ROIs.end();it++) {
		hogExtractor((*it).first, row);
		outputData.push_back(row);
	}
	outputFeatures = outputData;

	return (getTickCount() - start) / getTickFrequency();
}

double mserExtractor(pair<Mat,int> inputImage, list<pair<Mat,int>> & outputImages, ExtractionProperties props) {
	int64 start = getTickCount();
	bool FilterROIResults(Rect RectangleUnderTesting, Rect & Previous, ExtractionProperties props);
	//set up the MSERextractor
	Ptr<MSER> mserExtractor = MSER::create();
	if (props.Area_min != 0)
		mserExtractor->setMinArea(props.Area_min);
	if (props.Area_max != 0)
		mserExtractor->setMaxArea(props.Area_max);
	vector<vector<Point>> dummyVector;
	vector<Rect>Rectangles;
	mserExtractor->detectRegions(inputImage.first, dummyVector, Rectangles);
	
	if ((inputImage.first).channels() == 3)
		cvtColor(inputImage.first, inputImage.first, CV_BGR2GRAY);//make everything grayscale

	//filter the results of the MSER extractor
	Rect Previous = Rect(0, 0, 0, 0);//initialize a null rect
	for (vector<Rect>::iterator it = Rectangles.begin();it != Rectangles.end();it++) {
		if (FilterROIResults(*it, Previous, props))
			outputImages.push_back(make_pair(inputImage.first(Range((*it).y, (*it).y + (*it).height), Range((*it).x, (*it).x + (*it).width)),inputImage.second));
	}
	return (getTickCount() - start) / getTickFrequency();
}

double HoughExtractor(pair<Mat, int>inputImage, list<pair<Mat, int>> & outputImages, ExtractionProperties props) {
	//work in progress function that hopefully works better than the MSER extractor
	//http://www.itriacasa.it/generalized-hough-transform/
	
	return 0;
}

bool FilterROIResults(Rect RectangleUnderTesting, Rect & Previous, ExtractionProperties props) {
	float ratio = (float)RectangleUnderTesting.height / (float)RectangleUnderTesting.width;
	//if image is not the right proportion return false
	if (ratio > props.Ratio_U)
		return false;
	if (ratio < props.Ratio_L)
		return false;
	//if the image is the same as the last image return false
	int duplicitiyCount = 0;
	//test left side
	if ((RectangleUnderTesting.x - (Previous).x) < RectangleUnderTesting.x / props.MatchingFactor)
		duplicitiyCount++;
	//test top side
	if ((RectangleUnderTesting.y - (Previous).y) < RectangleUnderTesting.y / props.MatchingFactor)
		duplicitiyCount++;
	//test right side
	if (((RectangleUnderTesting.width + RectangleUnderTesting.x) - ((Previous).width + (Previous).x)) < RectangleUnderTesting.width / props.MatchingFactor)
		duplicitiyCount++;
	//test bottom side
	if (((RectangleUnderTesting.height + RectangleUnderTesting.y) - ((Previous).height + (Previous).y)) < RectangleUnderTesting.height / props.MatchingFactor)
		duplicitiyCount++;
	
	if (duplicitiyCount > 1) {//if there are 2 sides that are the same consider it a duplicate
		return false;
	}
	Previous = RectangleUnderTesting;
	return true;
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

void getFileNames(vector<string> & filenames, string directory, string extension) {
		std::string search_path = directory + "/*." + extension;
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

void createTrainingResponse(list<pair<Mat,int>> inputROIs, int numClasses, Mat & responses_1D, Mat & responses_2D) {//reads images as well as creates training responses
	for (list<pair<Mat, int>>::iterator it = inputROIs.begin();it != inputROIs.end();it++) {
		Mat Row = Mat::zeros(1, numClasses, CV_32F);
		if ((*it).second != -1)
			Row.at<float>(0, (*it).second) = 1.f;
		responses_1D.push_back((float)(*it).second);
		responses_2D.push_back(Row);
	}
}

classes::classes() {

};

bool classes::convert(string from_name,int & to_number) {
	to_number = -1;// -1 indicates no match, is changed to something if a match is found
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

bool classes::convert(int from_number,string & to_name) {
	to_name = "none";//default is no class, this is changed if a class is found

	if (from_number == -1)//if -1, return true, class is none
		return true;
	if (class_list.empty())//if empty, return false
		return false;
	if (from_number > class_list.size())//if too high, return false, class is invalid
		return false;

	to_name = (class_list[from_number]).first;//else find the appropriate class
	return true;
};

bool classes::initialize(string directory) {//reads from a file and creates a list of all classes
	ifstream file(directory);
	if (!file)
		return false;

	string  buf;
	int number = 0;
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

bool TrainingAnswers::getAnswer(int index) {
	return answers[index];
}

void TrainingAnswers::pushAnswer(bool answer) {
	answers.push_back(answer);
}

void TrainingAnswers::writeOut() {
	ofstream file(filename);
	for (vector<bool>::iterator it = answers.begin();it != answers.end();it++) {
		file << *it << endl;
	}
	file.close();
}

int TrainingAnswers::size() {
	return (int)answers.size();
}

TrainingAnswers::~TrainingAnswers() {
	if(!answersPreexist)
		writeOut();
}

void createConfusionMatrix(Mat results_predicted, Mat results_actual, Mat & confusionMatrix, int matrixClassCount, int & totalEntries) {
	Mat _confusionMatrix = Mat::zeros(matrixClassCount, matrixClassCount, CV_8U);//create an empty NxN matrix
	totalEntries = 0;
	int row, col;
	//recall that format is Mat.<type>at(y,x) where in our case, each y is the result from a different image, and x is the result for each class
	for (int i = 0;i < results_predicted.rows;i++) {//iterate through every result
		row = (uchar)results_predicted.at<float>(i, 0);
		col = (uchar)results_actual.at<float>(i, 0);
		_confusionMatrix.at<uchar>(col, row)++;//for each result set, increment the correct element of the confusion matrix
		totalEntries++;//increment the running total
	}
	confusionMatrix = _confusionMatrix;
}

void calculateMetrics(Mat results_predicted, Mat results_actual, vector<metrics> & data, int numClasses, Mat & confusionMatrix) {
	//function&var declarations
	void createConfusionMatrix(Mat results_predicted, Mat results_actual, Mat & confusionMatrix, int matrixClassCount, int & totalEntries);
	int totalEntries;
	metrics temp;
	//vector<float>FPrate;
	//vector<float>TPrate;
	//vector<float>Precision;
	//vector<float>Accuracy;
	//vector<float>Fscore;

	//body			
	createConfusionMatrix(results_predicted, results_actual, confusionMatrix, numClasses, totalEntries);//get confusion matrix and total entries into it

	for (int classSelect = 0;classSelect < numClasses;classSelect++) {//for each class

																	  //calculate the totals (per class)
																	  //--------------------------------
																	  //note all the stuff below is easier to understand when drawing a 3x3 confusion matrix
																	  //and labelling the areas of the matrix WRT a single class
																	  //	TPrate = TP/P
																	  //	FPrate = FP/N
																	  //	Precision = TP/(TP+FP)
																	  //	Fscore=Precision X TPrate
																	  //	Accuracy=(TP+TN)/(P+N)
		int actualClassTotal = 0, predictedClassTotal = 0;//per class this is the number of times the class *actually* occurs (instead of predicted occurs)
		for (int j = 0;j < numClasses;j++) {
			actualClassTotal += confusionMatrix.at<uchar>(j, classSelect);//sum of all entries on the same column -- A(i)
			predictedClassTotal += confusionMatrix.at<uchar>(classSelect, j);//sum of all entries on the same row -- P(i)
		}


		//calculate TPrate aka Recall
		//---------------------------
		float P = (float)actualClassTotal;
		float TP = (float)confusionMatrix.at<uchar>(classSelect, classSelect);
		temp.TPrate = (TP / P);

		//calculate FPrate
		//----------------
		float FP, N;
		N = (float)(totalEntries - actualClassTotal);
		FP = (float)(predictedClassTotal - confusionMatrix.at<uchar>(classSelect, classSelect));
		temp.FPrate = (FP / N);

		//calculate Accuracy,Precision,F-Score
		//------------------------------------
		float TN = totalEntries - FP - P;
		temp.Accuracy = ((TP + TN) / (P + N));
		temp.Precision = (TP / (TP + FP));
		temp.Fscore = (temp.Precision * temp.TPrate);

		//push into vector
		data.push_back(temp);

	}//end for each class
}

Mat generateROCgraph(vector<metrics> input) {
	Mat graph(100, 100, CV_8UC1);
	for (vector<metrics>::iterator it = input.begin();it != input.end();it++) {
		int x = (int)(*it).FPrate * 100;
		int y = (int)(*it).TPrate * 100;
		Point plotPoint(x, y);
		circle(graph, plotPoint, 2, 0, -1);
	}
	return graph;
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