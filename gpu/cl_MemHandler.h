// =================================================================================================
// This file is part of the YoloCL project. This project is a convertion from YOLO project.
// The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Juan DAmato <juan.damato@gmail.com>
// =================================================================================================
#pragma once


#include <iostream>
#include <fstream>
#include <vector>

#include <omp.h>

#include <cl/cl.hpp>

#define GPU_STATE_HOST 0
#define GPU_STATE_GPU 1

#define ACCESS_READ 0
#define ACCESS_WRITE 1

#define FIX_MEM_MODE 1
#define TEMP_MEM_MODE 0
#define BUFFER_MODE 2


class gpuBuffer
{
public:
	int state;
	int fixed;
	size_t size = 0;
	std::string name;
	int lastOperation;

	void* host_ptr;
	cl_mem object_;
	int refCount;
	int readCnt;
	int writeCnt;	
	gpuBuffer();
	cl_mem mem();
	int release();
	gpuBuffer(cl_mem& mem, cl_mem_flags flags, ::size_t size, void* host_ptr = NULL, cl_int* err = NULL);
	gpuBuffer(const cl::Context& context, cl_mem_flags flags, ::size_t size, void* host_ptr = NULL, cl_int* err = NULL);
};

class memStats
{
public:
	int readcnt;
	int writecnt;
	int time;
	
	gpuBuffer* gpubuffer;
	memStats() { readcnt = 0;  writecnt = 0;  time = 0;  }
	memStats(int read, int write, int sz, gpuBuffer* buffer) { readcnt = read; writecnt = write;  gpubuffer = buffer; }
};

class gpuMemHandler
{
public:
	static void setDebugMode(bool mode);
	static void verifyMemMode();
	static void setEmulationMode(bool mode);
	static void setMaxUsedMem(size_t maxMem);
	static int getEmulationMode();
	static memStats* getStats(void* data);
	static void setDeviceHasOwnMem(bool mode);
	static int WriteBuffer(gpuBuffer* buffer, cl_bool blocking , size_t offset, bool force = false);
	static int ReadBuffer(gpuBuffer* buffer, cl_bool blocking , size_t offset , bool force = false);
	static void releaseMemmory(size_t size);
	static void allocateAuxBuffer(size_t size, int count);
	static gpuBuffer* getSharedBuffer(void* data, bool registerInDevice, size_t mem_mode, int static_mem, std::string name);
	static gpuBuffer* getBuffer(void* data, bool registerInDevice = true,size_t mem_mode = CL_MEM_READ_WRITE , int static_mem = -1, std::string name = "");
	static void startGPUInvocation();
	static void endGPUInvocation();
	static void startGPULoop();
	static void endGPULoop();
	static void printStats();
	static void clearStats();
	static void printStats(void* data);
	static size_t getAssignedMem();
	static float getMaxUsedMem();
	static int getAvailableMem();

};

gpuBuffer openclGenArray(void* data, size_t size);
