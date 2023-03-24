// =================================================================================================
// This file is part of the YoloCL project. This project is a convertion from YOLO project.
// The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Juan DAmato <juan.damato@gmail.com>
// =================================================================================================
#ifndef CL_UTILS_H
#define CL_UTILS_H

#define MESSAGE_ERROR 1
#define MESSAGE_LOG 2

#include <iostream>
#include <fstream>

#include <omp.h>
#include <cl/cl.hpp>

using namespace std;



class cl_utils{
public:
	static std::string kernel_source;
    static void cargarFuente(std::string programa);

   // static int generateClRandom(cl::Buffer &clRandom,int sizeRandomBuff,cl::CommandQueue &queue);
    static int width;
    static int height;
	static double sft_clock();
};


struct dim3
{
	int x;
	int y;
	int z;
	dim3() { this->x = 1; this->y = 1; this->z = 1; }
	dim3(int x)
	{
		this->x = x; this->y = 1; this->z = 1;
	}
	dim3(int x, int y, int z)
	{
		this->x = x; this->y = y; this->z = z;
	}
};

namespace clUtils
{
	void exportCSV(std::string outfilename, std::vector<float> values, int time);
	void exportCSV(std::string outfilename, std::vector<std::string> values, int time);
	void logMessage(std::string s, int type);
	cl::Context getDefaultContext();
	int opencl_gridSize(int v, int blockSize);
	cl::Device getDefaultDevice();
	void setCLSource(std::string kernel_src);
	cl::Program buildProgram(cl::Context _Context, cl::Device _Device, int* error);

	void logMessage(string s, int type);
	int initDevice(int platformProcessingIndex, int deviceProcessingIndex);
	void set_device(int gpu_index);
	cl::CommandQueue getDefaultQueue();
	cl::Program getDefaultProgram();
	void setActiveProgram(cl::Program& prog);

	cl::Program loadCLSources(std::string cl_prog, cl::Context _Context, cl::Device _Device, int* error);
	cl::Program loadCLSources(std::string cl_prog, int localWG, int platformProcessingIndex, int deviceProcessingIndex, int* error);
	void assertCL(int iclError);
}
#endif // CL_UTILS_H
