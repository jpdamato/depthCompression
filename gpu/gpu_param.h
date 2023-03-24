#pragma once
// =================================================================================================
// This file is part of the YoloCL project. This project is a convertion from YOLO project.
// The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Juan DAmato <juan.damato@gmail.com>
//
///////class gpu_param
//
// =================================================================================================


#include <iostream>
#include <fstream>
#include <vector>

#include "cl_MemHandler.h"


using namespace std;

#define FLOAT_PARAM 0
#define INT_PARAM 1
#define BUFFER_PARAM 2

class gpu_param
{
public:
	int type;
	size_t size;
	int dataChanged;
	int memAccess;
	int ivalue;
	void* pvalue;
	float fvalue;
	gpuBuffer* _gpuBuffer;


	gpu_param(gpuBuffer* buffer);
	gpu_param(size_t v);
	gpu_param(int v);
	gpu_param(void *v);
	gpu_param(int *v);

	gpu_param(float v);
	gpu_param(float *v);
	float getfValue();
	float* getpfValue();
	int getiValue();
	int* getpiValue();
	int setParamToDevice(cl::Kernel clk, int order);
	int readParamFromDevice(cl::Kernel clk, int order);
};


void kernelCall(std::string kernelName, int dims, int partitions, std::vector<gpu_param> args, std::vector<gpu_param> outputs);
