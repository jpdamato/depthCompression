#pragma once
// =================================================================================================
// This file is part of the YoloCL project. This project is a convertion from YOLO project.
// The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Juan DAmato <juan.damato@gmail.com>
// =================================================================================================
#ifndef YOLO_OPENCL
#define YOLO_OPENCL

#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "clHeader.h"

namespace FitCL
{
	std::vector<spline_ptr> getSplines(bool runOnGPU);
	void computeMemUssage(int threadsCount);
	void encodeCL(cv::Mat& m, int nthreads, bool verbose,  bool runOnGPU, int saveResidual = 0);
	unsigned short* cl_getOutputBuffer(int& outSize);
	int cl_writeOutput(std::string fn);
	int initEncoderCL(std::string cl_dir, int localWG, int platformProcessingIndex, int deviceProcessingIndex);
	//	void draw_detections(cv::Mat im, std::vector< object_detected>& objs);
}

#endif