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

enum Modes { TRAIN, PREDICT_MAN, PREDICT_AUTO };

typedef struct ExtractionProperties {
	int Threshold_L;//lower threshold (anything below becomes black)
	int Threshold_U;//upper threshold (currently not implemented)
	float Ratio_L;//lower threshold for ratio of width to height of detected regions
	float Ratio_U;//upper threshold for ratio of width to height of detected regions
	int Area_max;//maximum area size of regions
	int Area_min;//minimum area size of regions
	int MatchingFactor;//ROIs are detected as the same if (new-old) < (new / MatchingFactor) <higher values=more duplicates>
	int Mode;///needs to be moved into program flags
}ExtractionProperties;

typedef struct ProgramFlags {
	int Mode;//training or prediction
	int TMode;//training mode
	bool Debug;//debug mode
} ProgramFlags;

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
};

int main(int argc, char** argv) {
	//function declarations
	void getFileNames(vector<string> & filenames, string directory, string extension);
	double ExtractROIs(pair<Mat, int> inputImage, list<pair<Mat, int>> & ROIs, ExtractionProperties props);
	double ExtractFeatures(list<pair<Mat, int>> ROIs, Mat & outputFeatures);
	void setLayerSizes(string input, int firstLayer, int LastLayer, vector<int> & output);
	void createTrainingResponse(list<pair<Mat, int>> inputROIs, int numClasses, Mat & responses_1D, Mat & responses_2D);
	double trainNetwork(const Mat Tdata, const Mat Tresponses, string networkDir, vector<int> layerSizes, bool update);

	string keys =
		"{train||set this to indicate training mode}"
		"{networkpath|network.xml|the network save location}"
		"{imagepath|./images|default image directory}"
		"{train||set this if you want to do training}"
		"{method|manual|manual means a user must verify each ROI passed, automatic passes all ROIs (less accurate)}"
		"{layers|50|the number of internal layers for the ANN}"
		"{classes|classes.txt|a text file listing all the class names}"
		"{ROIdir|ROIs|a directory for all regions of interest during for AUTO mode}"
		"{debug||enable debug mode, this will automate answering training questions}"
		"{answers|answers.txt|training answers, only used in debug mode}"
		;

	CommandLineParser parser(argc, argv, keys);
	
	ExtractionProperties ExProps;
	ExProps.Area_min = 20 * 20;
	ExProps.Area_max = 0;//no max limit
	ExProps.Ratio_L = 0.6f;
	ExProps.Ratio_U = 1.4f;
	ExProps.MatchingFactor = 20;
	ExProps.Threshold_L = 127;
	ExProps.Threshold_U = 200;

	ProgramFlags ProgFlags;
	ProgFlags.Debug = (parser.has("debug"));


	classes class_list;
	string classfile=parser.get<string>("classes");
	if (!class_list.initialize(classfile)) {
		cout << "failed to initialize classes, please ensure that a valid class file " + classfile + " exists";
		return 1;
	}
	int class_count = class_list.size();

	if (parser.has("train")) {
		//set the training method
		string method = parser.get<string>("method");
		size_t found = method.find("auto");
		if (found != string::npos)
			ExProps.Mode = PREDICT_AUTO;
		else
			ExProps.Mode = PREDICT_MAN;//defeault is Predict Manual

		//load the images in given directory
		string imagePath = parser.get<string>("imagepath");
		vector<string>imageNames;
		string extension = "png";
		getFileNames(imageNames, imagePath, extension);
		vector<pair<Mat,int>> trainingImages;//Mat and the class it is
		Mat temp;
		int imageClass;
		for (vector<string>::iterator it = imageNames.begin();it != imageNames.end();it++) {
			temp = imread(imagePath + "/" + (*it), IMREAD_COLOR);
			class_list.convert(*it, imageClass);
			trainingImages.push_back(make_pair(temp,imageClass));
		}

		//perform extraction of ROIs
		list<pair<Mat,int>>ROIs;//ROIs and classes
		for (vector<pair<Mat,int>>::iterator it = trainingImages.begin();it != trainingImages.end();it++) {
			ExtractROIs(*it, ROIs, ExProps);
		}

		//ask user about each ROI
		if (ExProps.Mode == PREDICT_MAN) {

			TrainingAnswers answers;
			bool FilterManually = true;
			if (ProgFlags.Debug) {
				answers.initialize(parser.get<string>("answers"));
				FilterManually = !answers.answersExist();//if answers did exist we do not need to manually filter
			}

			list<pair<Mat, int>>::iterator it = ROIs.begin();
			list<pair<Mat, int>>::iterator temp;
			if (!FilterManually) {//if we already have a filter set up, extract answers from the file
				int i = 0;
				while (it != ROIs.end()) {
					if (answers.getAnswer(i++) == false) {
						temp = it;
						it++;
						ROIs.erase(temp);
					}
					else 
						it++;
				}
			}
			else {//else manually answer each one if no file is set up or if not in debug
			Mat BlankImage = Mat::zeros(Size(1, 1), CV_8UC3);
			cout << "for each of the following images, please enter (y/n) appropriately, y=a sign(that we want), n=not a sign (or not one we want)";
			namedWindow("Confirmation Window");
				while (it != ROIs.end()) {
					int key = 0;
					do {
						imshow("Confirmation Window", BlankImage);
						imshow("Confirmation Window", (*it).first);
						key = waitKey(0);
					} while (!((key == 'y') || (key == 'n')));
					if (key == 'n') {
						temp = it;
						it++;
						ROIs.erase(temp);
						if (ProgFlags.Debug) answers.pushAnswer(false);
					}
					else {
						if (ProgFlags.Debug) answers.pushAnswer(true);
						it++;
					}
				}
				destroyWindow("Confirmation Window");
				if (ProgFlags.Debug) answers.writeOut();
			}
		}
		/*else	
			//at this point there needs to be an auto system too
		*/
		//we now have all the features we want to analyze, perform extraction of features
		Mat imageFeatures;
		ExtractFeatures(ROIs, imageFeatures);

		//obtain the output classes
		Mat responses1D, responses2D;
		createTrainingResponse(ROIs, class_list.size(), responses1D, responses2D);

		//now set up the neural network
		string layerConfig = parser.get<string>("layers");
		vector<int>layerSizes;
		setLayerSizes(layerConfig, imageFeatures.cols, class_count, layerSizes);
		
		string networkpath = parser.get<string>("networkpath");
		trainNetwork(imageFeatures, responses2D, networkpath, layerSizes, false);
	}

}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

double trainNetwork(const Mat Tdata, const Mat Tresponses, string networkDir, vector<int> layerSizes, bool update) {//update indicates network is updating, instead of creating
	//this function creates or updates an ANN and returns time taken
	int64 start = getTickCount();
	TermCriteria TC(int, double);
	Ptr<TrainData> TrainingData = TrainData::create(Tdata, ROW_SAMPLE, Tresponses);
	Ptr<ANN_MLP> network = StatModel::load<ANN_MLP>(networkDir);
	int flag;
	if (!update) {
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

double ExtractROIs(pair<Mat,int> inputImage, list<pair<Mat,int>> & ROIs, ExtractionProperties props) {
	int64 start = getTickCount();
	double mserExtractor(pair<Mat, int> inputImage, list<pair<Mat, int>> & outputImages, ExtractionProperties props);
	Mat temp = inputImage.first;
	//to one channel
	cvtColor(temp, temp, CV_BGR2GRAY);

	//threshold the image
	threshold(temp, temp, props.Threshold_L, 255, ThresholdTypes::THRESH_TOZERO);//perhaps try an adaptive thresholding rule too later

	//apply canny edge
	//Canny(temp, temp, 127, 255);
	 
	inputImage.first = temp;
	//extract ROIs using MSER
	mserExtractor(inputImage, ROIs, props);

	return (getTickCount() - start) / getTickFrequency();
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
	bool FilterMserResults(Rect RectangleUnderTesting, Rect & Previous, ExtractionProperties props);
	//set up the MSERextractor
	Ptr<MSER> mserExtractor = MSER::create();
	if (props.Area_min != 0)
		mserExtractor->setMinArea(props.Area_min);
	if (props.Area_max != 0)
		mserExtractor->setMaxArea(props.Area_max);
	vector<vector<Point>> dummyVector;
	vector<Rect>Rectangles;
	mserExtractor->detectRegions(inputImage.first, dummyVector, Rectangles);
	//filter the results of the MSER extractor
	Rect Previous = Rect(0, 0, 0, 0);//initialize a null rect
	for (vector<Rect>::iterator it = Rectangles.begin();it != Rectangles.end();it++) {
		if (FilterMserResults(*it, Previous, props))
			outputImages.push_back(make_pair(inputImage.first(Range((*it).y, (*it).y + (*it).height), Range((*it).x, (*it).x + (*it).width)),inputImage.second));
	}
	return (getTickCount() - start) / getTickFrequency();
}

double HoughExtractor(pair<Mat, int>inputImage, list<pair<Mat, int>> & outputImages, ExtractionProperties props) {
	return 0;
}

bool FilterMserResults(Rect RectangleUnderTesting, Rect & Previous, ExtractionProperties props) {
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
		responses_2D.push_back(Row);
		responses_1D.push_back((float)(*it).second);
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
	to_name = "";//default is no class, this is changed if a class is found
	if (class_list.empty())
		return false;
	if (from_number > class_list.size()) {
		return false;
	}
	to_name = (class_list[from_number]).first;
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

TrainingAnswers::~TrainingAnswers() {
	if(!answersPreexist)
		writeOut();
}
