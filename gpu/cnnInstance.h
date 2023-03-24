#pragma once

#pragma once
#ifndef CNN_INSTANCE_H
#define CNN_INSTANCE_H

#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>

#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// YOLO Includes
#include "../include/darknet.h"

#include "yoloopenCL.h"
#include "cl_utils.h"
#include "cl_MemHandler.h"



#ifdef TRACKING_LIB
#include "../blobs/blobsHistory.h"
#include "clVibe.h"

trackingLib::BlobsByFrame* detectOnFrameCNN(network *net, cv::Mat& mM, int nframe, bool draw);
trackingLib::BlobsByFrame* _converToBlobs(cv::Mat& input, detection* dets, int num, float thresh, char **names, image **alphabet, int classes, int nframe);
trackingLib::Blob* insideAnyBlob(trackingLib::BlobsByFrame* bb0, cv::Point2f pt, float offset);
void removeOverlapped(std::vector<trackingLib::Blob*>& blobs);
#endif

network* buildNet(std::string _dirEXE, std::string cfgfile, std::string weightfile, std::string namesSrc = "");

std::string ExePath();
std::string randomString();
image make_imageX(int w, int h, int c);
image mat_to_imageX(cv::Mat& m, image& im);
void cl_draw_detections(cv::Mat& im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

class cnnInstance
{
public:
	detection* dets;
	network *net;
	char** names;
	

	int nframe;
	int nboxes;
	int id;
	float* imData;
	float* imSizeData;

	std::mutex mtxM;
	std::mutex mtxImg;
	
	cv::Mat squared;
	cv::Mat mM;
	cv::Mat foreGround;
	cv::Mat frame;
	cv::Rect roi;
	image sized;
	image im;
	int yoloMode = 0;

	float nms = .45f;
	float hier_thresh = 0.15f;
	float thresh = 0.15f;

	cnnInstance(network* n, size_t _id);

	void setFrame(cv::Mat& frame);

	cv::Mat GetSquareImage(cv::Mat& squared, cv::Mat& img, int target_width);


	void updateInput();

	void drawResults(cv::Mat& m);

#ifdef TRACKING_LIB
	trackingLib::BlobsByFrame* readResults(bool doRemoveOverlap = true, float threshold = 0.3);
#endif

	void predict(bool useBackGroundSub);

};

#endif

