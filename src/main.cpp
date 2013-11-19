/*
 * main.cpp
 *
 *  Created on: 16.11.2013
 *      Author: Denis Neuling (denisneuling@gmail.com)
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cstring>

// 0 -> quite
// 1 -> INFO
// 2 -> DEBUG
// 3 -> TRACE
int verbosity = 1;

/* ImageLoader *****************************************/
class ImageLoader {
private:
	std::string filename;
	int index;
public:
	std::string basename;
	std::string extension;
	std::string continuousVariable;
	cv::Mat next();
	cv::Mat loadImage(int index);
};

cv::Mat ImageLoader::next() {
	return this->loadImage(this->index++);
}

cv::Mat ImageLoader::loadImage(int index) {
	const std::string fn = this->basename + this->continuousVariable + this->extension;

	char filename[FILENAME_MAX];
	snprintf(filename, sizeof(filename), fn.c_str(), index);
	if(verbosity >= 3){
		std::cout << "[TRACE] Reading file "<< filename << std::endl;
	}
	//printf("read file %s\n", filename);

	return cv::imread(filename);
}
/* END ImageLoader **************************************/

/* SimpleBackgroundSubtractor *****************************************/
class SimpleBackgroundSubtractor: public cv::BackgroundSubtractor {
protected:
	cv::Mat backgroundImg;
	cv::Mat foreground;
	bool initialized;
public:
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate);
};

void SimpleBackgroundSubtractor::operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate) {
	/*
	 * standard color deviation for bilateral Filtering
	 */
	const double colorDeviation(5);
	/*
	 * standard pixel distance deviation for gauss and bilateral filtering
	 */
	const double pixelDeviation(2.5);

	cv::Mat img2;
	cv::bilateralFilter(image, img2, 0, colorDeviation, pixelDeviation, cv::BORDER_REFLECT);
	img2.copyTo(image.getMat());

	if (initialized) {
		// current image minus background = foreground
		cv::absdiff(image, backgroundImg, foreground);

		cv::threshold(foreground, fgmask, 80, 255, cv::THRESH_BINARY);

		// update background with running average algorithm
		backgroundImg = backgroundImg * (1 - learningRate) + image.getMat() * learningRate;

	} else {
		backgroundImg = image.getMat();
		initialized = true;
		cv::threshold(image, fgmask, 80, 255, cv::THRESH_BINARY);
		backgroundImg.copyTo(fgmask);
	}
}

/* END SimpleBackgroundSubtractor *****************************************/

cv::Rect roi;
cv::Mat image;
bool mouseActivated = false;

cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar RED = cv::Scalar(0,0,255);
cv::Scalar COLOR = BLACK;

/* ROIObjectRecognizer *****************************************/
class ROIObjectRecognizer {
private:
	bool detected;
	int count;
	ImageLoader* imageLoader;
	SimpleBackgroundSubtractor backgroundSubtractor;
	cv::Mat foreground;

	cv::Mat foregroundROI;
	cv::Mat foregroundRGBChannels[3];
public:
	ROIObjectRecognizer(ImageLoader* imageLoader, SimpleBackgroundSubtractor backgroundSubtractor);
	void recognize();
};

ROIObjectRecognizer::ROIObjectRecognizer(ImageLoader* imageLoader, SimpleBackgroundSubtractor backgroundSubtractor) {
	this->imageLoader = imageLoader;
	this->backgroundSubtractor = backgroundSubtractor;
	this->detected = false;
	this->count = 0;
}
void ROIObjectRecognizer::recognize() {
	for (;;) {
		image = imageLoader->next();

		if (image.empty()) {
			break;
		} else {
			this->backgroundSubtractor(image, this->foreground, 0.025f);

			if(!mouseActivated){
				cv::Mat smeared = image.clone();
				cv::rectangle( smeared, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width, roi.y + roi.height), COLOR, 1 );
				cv::imshow( "BELEG1", smeared );
			}

			this->foregroundROI = this->foreground(roi);

			cv::split(foregroundROI, foregroundRGBChannels);
			cv::bitwise_or(foregroundRGBChannels[0], foregroundRGBChannels[1], foregroundRGBChannels[1]);
			cv::bitwise_or(foregroundRGBChannels[1], foregroundRGBChannels[2], foregroundRGBChannels[2]);

			//cv::imshow("Foreground", foregroundRGBChannels[2]);

			int rectangleSize = foregroundRGBChannels[2].cols * foregroundRGBChannels[2].rows;
			int x = cv::countNonZero(foregroundRGBChannels[2]);

			int upperThreshold = 20;
			int lowerThreshold = 15;

			if(verbosity >= 2){
				std::cout
					<< "[DEBUG] "
					<< x
					<< ">"
					<< (rectangleSize / 100 * upperThreshold)
					<< " "
					<< (x > (rectangleSize / 100 * upperThreshold))
					<< " | "
					<< x
					<< " < "
					<< (rectangleSize / 100 * lowerThreshold)
					<< " "
					<< (x < (rectangleSize / 100 * lowerThreshold))
					<< " "
					<< std::endl;
			}

			if(!detected && x > (rectangleSize / 100 * 20) ){ // 50%
				detected = true;
				COLOR = RED;

				this->count++;

				if(verbosity >= 1){
					std::cout << "[INFO ] Object {" << this->count << "} entered selected ROI." << std::endl;
				}
			}
			if(detected && x < (rectangleSize / 100 * 1) ){ // 10%
				detected = false;
				COLOR = BLACK;

				if(verbosity >= 1){
					std::cout << "[INFO ] Object {" << this->count << "} left selected ROI." << std::endl;
				}
			}
			cv::waitKey(15);
		}
	}
	std::cout << "Detected Objects in selected ROI: " << this->count << std::endl;
}
/* END ROIObjectRecognizer *****************************************/

ROIObjectRecognizer* p_roiObjectRecognizer;
cv::Mat foreGround;

int main(int argc, char ** argv) {

	if (argc < 4) {
		std::cout << argv[0] << " </my/basename> <continuous var> <.extension>" << std::endl;
		std::cout << std::endl;
		std::cout << "Usage:" << std::endl;
		std::cout << "\t</my/basename>\t\t- the path and the beginning of the filename" << std::endl;
		std::cout << "\t<continuous var>\t- the continuous variable e.g. if 0000 pick %04i" << std::endl;
		std::cout << "\t<.extension>\t\t- the file extension to use, e.g. .jpeg" << std::endl;
		std::cout << std::endl;
		std::cout << "\t" << argv[0] << " /home/user/images/frame_ %04i .jpeg" << std::endl;
		std::cout << std::endl;
		return -1;
	}

	std::string basename = argv[1];
	std::string continuousVariable = argv[2];
	std::string extension = argv[3];

	std::string windowName = "BELEG1";

	ImageLoader imageLoader = ImageLoader();
	imageLoader.basename = basename;
	imageLoader.extension = extension;
	imageLoader.continuousVariable = continuousVariable;

	image = imageLoader.next();

	if (image.empty()) {
		std::cout << "[ERROR] Could not find initial frame!" << std::endl;
		return -1;
	}

	roi.height = image.rows;
	roi.width = image.cols;
	roi.x = 0;
	roi.y = 0;

	SimpleBackgroundSubtractor backgroundSubtractor = SimpleBackgroundSubtractor();
	ROIObjectRecognizer roiObjectRecognizer = ROIObjectRecognizer(&imageLoader, backgroundSubtractor);
	p_roiObjectRecognizer = &roiObjectRecognizer;

	cv::MouseCallback mouseCallback = [] (int event, int x, int y, int, void* mouseEvent) {
		if (event == cv::EVENT_LBUTTONDOWN) {
			mouseActivated = true;
			if(verbosity >= 2){
				std::cout << "[DEBUG] EVENT_LBUTTONDOWN" <<std::endl;
			}
			roi.x = x;
			roi.y = y;

		} else if (mouseActivated && event == cv::EVENT_MOUSEMOVE) {
			cv::imshow( "BELEG1", image );
			cv::Mat smeared = image.clone();
			cv::rectangle( smeared, cv::Point(roi.x, roi.y), cv::Point(x, y), cv::Scalar(0,0,0), 1 );
			cv::imshow( "BELEG1", smeared );
		} else if (event == cv::EVENT_LBUTTONUP) {
			mouseActivated = false;

			if(verbosity >= 2){
				std::cout << "[DEBUG] EVENT_LBUTTONUP" <<std::endl;
			}

			if (x > roi.x) {
				roi.width = x - roi.x;
			} else {
				roi.width = roi.x - x;
				roi.x = x;
			}
			if (y > roi.y) {
				roi.height = y - roi.y;
			} else {
				roi.height = roi.y - y;
				roi.y = y;
			}

			p_roiObjectRecognizer->recognize();
			cv::destroyAllWindows();
		};
	};

	for (int i = 10; i > 0; i--) {
		cv::Mat frame = imageLoader.loadImage(i);
		if (frame.empty()) {
			std::cout << "[ERROR] There arent enough frames to setup the background subtractor!" << std::endl;
			return -1;
		}
		backgroundSubtractor(frame, foreGround, 0.025f);
	}

	cv::namedWindow(windowName, 1);
	cv::setMouseCallback(windowName, mouseCallback, 0);
	cv: imshow(windowName, image);

	cv::waitKey(0);
}

