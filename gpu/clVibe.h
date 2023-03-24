#pragma once

#include <CL\cl.hpp>

#include <opencv2/core.hpp>   // Include OpenCV API
#include <opencv2/highgui.hpp>   // Include OpenCV API
#include <opencv2/imgcodecs.hpp>   // Include OpenCV API
#include <opencv2/imgproc.hpp>   // Include OpenCV API
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "cl_utils.h"


#define randomSize 2048
//////////////////////////////
#define randSubSample 8
#define NCHANNELS 1
//////////////////////////////
#define N 20
#define R 20
#define R2 400
#define R2f 400.0
#define Rf 20.0
#define cMin 2

#define BLACK 0
#define WHITE 255

#define MAX_NUM_POINTS 8192

class cl_bgs 
{
public:
	int updateStep = 1;
	cl_bgs();
	cl_bgs(string cl_prog, int localWG);
	int init(std::string cl_prog, int localWG, int platformProcessingIndex = 0, int deviceProcessingIndex = 0);
	//void execute(cl::Context &context,cl::Program &program,cl::CommandQueue &queue);
	void executeRGB(cl::Context &context, cl::Program &program, cl::CommandQueue &queue);
	void operate(cv::Mat& inputFrame, cv::Mat& bfFrame, bool fillContours);
	void computeShadowMask(cv::Mat& input, cv::Mat &meanInput, cv::Mat& bFrame);
	void initialize(std::string srcCL,cv::Mat &inputFrame, int platformProcessingIndex, int deviceProcessingIndex);
	cv::Mat getMeanBack(cv::Mat& inputFrame);
	cv::Mat getSample(cv::Mat& inputFrame, int index);
	void getBack(cv::Mat& bfFrame);

private:
	int clCodeLoaded = 0;
	int* buffRandom;
	string base;
	cl::Program vibeProgram;
	int localWorkGroup;
	size_t lW, gM;
	cl::NDRange gRM, lRW;
	cv::Mat meanFrame, inFrame;
	int width, height;
	size_t whchann, wh, whNchann;
	uchar* input;
	uchar* hsamples;
	int numFrame, sizeRandomBuff;
	int iclError;
	uchar* background;
	uchar* tempBuffer;
	//ContourDetection* cd;



};

void pixelBWCount(cv::Mat& src, int &count_white, int &count_black);