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
#include "../include/darknet.h"

#include "../../src/layer.h"

// New !!!
struct object_detected
{
	int id;
	box reg;
	std::vector<std::pair<std::string, float>> classes;
};



namespace YoloCL
{
	void setAllocationScheme(int memMode, size_t totalAvailableMem);
	void predictYoloCL(network *net, network_state& state , int nframe, int yoloMode, bool checkIsZero = false);
	std::vector<object_detected> detectOnFrameCNN(network *net, float* frameData, int width, int height, int nframe, char** names);
	std::vector<object_detected> makeDetections(detection *dets, int num, float thresh, char **names,  int classes);
	int initYoloCL(std::string cl_dir, int localWG, int platformProcessingIndex, int deviceProcessingIndex);
	detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
	void _forward_route_layer(layer& l, network_state& state, bool useGpu = true);


	void computAndPrint(layer& l, int iter);
//	void draw_detections(cv::Mat im, std::vector< object_detected>& objs);
}

#endif