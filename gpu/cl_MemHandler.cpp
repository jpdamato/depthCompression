// =================================================================================================
// This file is part of the YoloCL project. This project is a convertion from YOLO project.
// The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Juan DAmato <juan.damato@gmail.com>
// =================================================================================================
#include "cl_MemHandler.h"
#include "cl_utils.h"
#include "../u_ProcessTime.h"

#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>  
#include <algorithm> 
#include <fstream>
#include <vector>
#include <map>

std::map<void*, memStats*> memmoryAccess;
std::map<void*, gpuBuffer*> assignedGPUData;
std::vector<void*> priorityAcceses;

size_t totalAvailableMem = 3.6 * 1024 * 1024 * 1024;
bool debugMode = false;
bool demoMode = false;
bool deviceHasOwnMem = true;
int doubleWriteAttemps = 0;

int gpuBuffer::release()
{
	int iCl = 0;
	if (object_)
	{
		iCl = clReleaseMemObject(this->object_);
	}
	object_ = NULL;
	return iCl;
}


gpuBuffer::gpuBuffer()
{
	this->object_ = NULL;
	host_ptr = NULL;
	state = 0;
	size = 0;
	readCnt = 0;
	writeCnt = 0;
	fixed = 0;
}

cl_mem gpuBuffer::mem()
{
	return object_;
}

gpuBuffer::gpuBuffer(cl_mem& mem, cl_mem_flags flags, ::size_t size, void* host_ptr, cl_int* err)
{
	object_ = mem;
	state = 0;
	this->size = size;
	this->host_ptr = host_ptr;
	readCnt = 0;
	writeCnt = 0;
	fixed = 0;
}

gpuBuffer::gpuBuffer(const cl::Context& context, cl_mem_flags flags, ::size_t size, void* host_ptr, cl_int* err)
{
	object_ = clCreateBuffer(context(), flags, size, host_ptr, err);
	
	state = 0;
	this->size = size;
	this->host_ptr = host_ptr;
	readCnt = 0;
	writeCnt = 0;
	fixed = 0;
}

std::vector<cl_mem> sharedGPUBuffer ;
void gpuMemHandler::allocateAuxBuffer(size_t size, int count)
{
	int iclError = 0;
	for (int i = 0; i < count; i++)
	{
		cl::Context ctx = clUtils::getDefaultContext();
		cl_mem  shared  = clCreateBuffer(ctx(), CL_MEM_READ_WRITE, size, NULL, &iclError);
		if (iclError != CL_SUCCESS)
		{
			std::cout << "gpuMemHandler ERROR : Error when creating Shared buffer" << "\n";
		}

		sharedGPUBuffer.push_back(shared);
	}

}

int sharedIndex = 0;
gpuBuffer* gpuMemHandler::getSharedBuffer(void* data, bool registerInDevice, size_t mem_mode, int static_mem, std::string name)
{
	gpuBuffer* device_B = assignedGPUData[data];

	if (device_B)
	{
		return device_B;
	}

	int iclError = 0;
	size_t size = _msize(data);

	device_B = new gpuBuffer(sharedGPUBuffer[sharedIndex], mem_mode | CL_MEM_COPY_HOST_PTR, size, data, &iclError);

	sharedIndex = (sharedIndex + 1) % sharedGPUBuffer.size();

	device_B->name = name;
	device_B->fixed = 1;

	if (iclError != CL_SUCCESS)
	{
		std::cout << "gpuMemHandler ERROR : Error when creating buffer" << "\n";
	}

	if (registerInDevice)
	{
		assignedGPUData[data] = device_B;
	}
	
	clUtils::assertCL(iclError);
	return device_B;
}

////////////////////////////////////////////////////////////////////////////////////////////
gpuBuffer* gpuMemHandler::getBuffer(void* data, bool registerInDevice, size_t mem_mode , int static_mem , std::string name )
{
	gpuBuffer* device_B = assignedGPUData[data];

	if (device_B)
	{
		return device_B;
	}
	
	int iclError = 0;
	size_t size = _msize(data);

	if (deviceHasOwnMem) 	device_B  = new gpuBuffer(clUtils::getDefaultContext(), mem_mode | CL_MEM_COPY_HOST_PTR, size, data, &iclError);
	else 	device_B = new gpuBuffer(clUtils::getDefaultContext(), mem_mode | CL_MEM_COPY_HOST_PTR, size, data, &iclError);
	
	device_B->name = name;

	if (iclError != CL_SUCCESS)
	{
		std::cout << "gpuMemHandler ERROR : Error when creating buffer" << "\n";
	}

	if (registerInDevice)
	{
		assignedGPUData[data] = device_B;
	}
	if (demoMode) return device_B;
	
	if (static_mem == 1)
	{
		device_B->fixed = static_mem;
		WriteBuffer(device_B, true, 0, true);
	}
	else
	{
		// nothing to do
	}
		
	clUtils::assertCL(iclError);
	return device_B;
}


size_t maxMemCreated = 0;

void gpuMemHandler::startGPUInvocation()
{
	
}

void gpuMemHandler::verifyMemMode()
{
	int supported = 0;
	// Check if device has own memmory or add a flag
	clUtils::getDefaultDevice().getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &supported);
	gpuMemHandler::setDeviceHasOwnMem(!supported);
}

// Compares two intervals according to staring times. 
bool compareInterval(memStats* i1, memStats* i2)
{
	return (i1->writecnt > i2->writecnt);
}

void gpuMemHandler::setDeviceHasOwnMem(bool mode)
{
	deviceHasOwnMem = mode;
}

void gpuMemHandler::setDebugMode(bool mode)
{
	debugMode = mode;
}

int gpuMemHandler::getEmulationMode()
{
	return demoMode;
}

void gpuMemHandler::setEmulationMode(bool mode)
{
	demoMode = mode;
/*
	for (int i = 0; i < memmoryAccess.size(); i++)
	{
			memmoryAccess.at(i)->gpubuffer->release();
	}*/
	// ... 
	for (std::map<void*, memStats*>::value_type& x : memmoryAccess)
	{
		x.second->gpubuffer->release();
	}

	memmoryAccess.clear();
	assignedGPUData.clear();

}

size_t gpuMemHandler::getAssignedMem()
{
	size_t totalMem = 0;
	for (std::map<void*, gpuBuffer*>::value_type& x : assignedGPUData)
	{
		totalMem += x.second->size;

	}

	/*
	for (int i = 0; i < assignedGPUData.size(); i++)
	{
		totalMem += assignedGPUData[i]->size;
	}
	*/
	maxMemCreated = max(totalMem, maxMemCreated);
	return totalMem;
}

int gpuMemHandler::getAvailableMem()
{
	
	return max(0, (int)totalAvailableMem - (int)getAssignedMem());
}


void gpuMemHandler::setMaxUsedMem(size_t maxMem)
{
	std::cout << " Set MAX mem " << maxMem << "\n"; 
	totalAvailableMem = maxMem;
}



void gpuMemHandler::releaseMemmory(size_t size)
{
	// Release only some buffer
	size_t totalMem = gpuMemHandler::getAssignedMem();
	if (totalMem + size > totalAvailableMem)
	{
		std::vector< memStats*> memAcc;
		std::vector<gpuBuffer*> assignGPU;
		/*
//BUG
	(memmoryAccess.begin(), memmoryAccess.end(), compareInterval);

		for (int i = 0; i < memmoryAccess.size(); i++)
		{
			//..to check
			if (totalMem + size > totalAvailableMem)
			{
				totalMem -= memmoryAccess[i]->gpubuffer->size;
				memmoryAccess[i]->gpubuffer->release();

			}
			else
			{
				memAcc.push_back(memmoryAccess[i]);
				assignGPU.push_back(memmoryAccess[i]->gpubuffer);
			}
		}
		memmoryAccess.swap(memAcc);
		assignedGPUData.swap(assignGPU);
		*/
	}
}
void gpuMemHandler::endGPUInvocation()
{
	if (demoMode) return ;

	std::vector< memStats*> memAcc;
	std::vector<gpuBuffer*> assignGPU;
	
	size_t totalMem = 0;
	//sort(memmoryAccess.begin(), memmoryAccess.end(), compareInterval);

	// ... 

	for (std::map<void*, gpuBuffer*>::value_type& x : assignedGPUData)
	{
		if (!x.second) continue;
		totalMem += x.second->size;

		if (!x.second->fixed)
		{
			x.second->release();
		}
		else
		{
			//memAcc.push_back(x.second);
			assignGPU.push_back(x.second);
		}
	}
	
	//memmoryAccess.clear();
	for (int i = 0; i < memAcc.size(); i++)
	{
		//memmoryAccess[memAcc[i]->gpubuffer->host_ptr] = memAcc[i];
	}
	

	assignedGPUData.clear();
	for (int i = 0; i < assignGPU.size(); i++)
	{
		assignedGPUData[assignGPU[i]->host_ptr] = assignGPU[i];
	}

	
	//assigned//GPUData.swap(assignGPU);
	maxMemCreated = max(maxMemCreated, totalMem);
}


float gpuMemHandler::getMaxUsedMem()
{
	return maxMemCreated /(1024.0f*1024.0f);
}

void gpuMemHandler::startGPULoop()
{
	maxMemCreated = 0;
}
void gpuMemHandler::endGPULoop()
{
	getAssignedMem();
}

void gpuMemHandler::clearStats()
{
	for (std::map<void*, memStats*>::value_type& x : memmoryAccess)
	{
		if (!x.second) continue;
		x.second->readcnt = 0;
		x.second->writecnt = 0;

	}
	
	doubleWriteAttemps = 0;
}

memStats* gpuMemHandler::getStats(void* data)
{
	return memmoryAccess[data];
}

void gpuMemHandler::printStats()
{
	std::vector<memStats*> copyT;
	float meanAcceses = 0.0f;
	float totalMemReq = 0.0f;

	int readOnlyMem = 0;
	int equallyAccessBuffers = 0;
	int multiUsedAccess = 0;

	for (std::map<void*, memStats*>::value_type& x : memmoryAccess)
	{
		copyT.push_back(x.second);
		meanAcceses += (x.second->writecnt + x.second->readcnt);
		totalMemReq += x.second->gpubuffer->size;

		if (x.second->readcnt == 0) readOnlyMem++;
		else if (x.second->readcnt == x.second->writecnt) equallyAccessBuffers++;
		else multiUsedAccess++;
	}
	meanAcceses /= copyT.size();
	std::cout << " Mean access count: " << meanAcceses << " Total mem req : " << totalMemReq / (1024.0f*1024.0f)
				 << " Available Mem :" << totalAvailableMem / (1024.0f*1024.0f) << "\n";
	//sort(copyT.begin(), copyT.end(), compareInterval);
	totalMemReq = 0.0f;
	priorityAcceses.clear();
	int i = 0;
	std::cout << " .. STORE IN STATIC MEM : " << "\n";
	for (std::map<void*, memStats*>::value_type& x : memmoryAccess)
		{ 			
			
			totalMemReq += x.second->gpubuffer->size;
			// No more enough memmory
			if (totalMemReq > totalAvailableMem) break;
			std::cout << " .... " << x.second->gpubuffer->host_ptr << " access count. read: " << x.second->readcnt
				<< " write:" << x.second->writecnt
				<< " Total writes " << x.second->gpubuffer->size << "(" << x.second->gpubuffer->name << ")" <<"\n";
			priorityAcceses.push_back(x.second->gpubuffer->host_ptr);
			
		}
	
	std::cout << "Double WRITE MEM attempts " << doubleWriteAttemps << "\n";
	std::cout << "ReadOnly  attempts " << readOnlyMem << "\n";
	std::cout << "TempBuffers attempts " << equallyAccessBuffers << "\n";
	std::cout << "MultiUsed attempts " << multiUsedAccess << "\n";

	copyT.clear();
}

void gpuMemHandler::printStats(void* data)
{
	std::cout << " MEMMORY ACCESSES. Total access " << memmoryAccess.size() << "\n";
	//for (int i = 0; i < memmoryAccess.size(); i++)
	for (std::map<void*, memStats*>::value_type& x : memmoryAccess)
	{
		if (x.second->gpubuffer->host_ptr != data) continue;
		std::cout << " .... " << x.second->gpubuffer->host_ptr << " access count. read: " << x.second->readcnt
			<< " write:" << x.second->writecnt
			<< " Total writes " << x.second->gpubuffer->size << "\n";
		
		
	}
}

void computeAccess(gpuBuffer* buffer, int mode)
{
	bool find = false;

	memStats* mst = (memStats*) memmoryAccess[buffer->host_ptr];

	if (mst)
	{
		if (mode == ACCESS_WRITE)		mst->writecnt += 1;
		else mst->readcnt += 1;

		find = true;
	}
	else
	{
		memmoryAccess[buffer->host_ptr] = new memStats(!mode, mode, buffer->size, buffer);
		
	}
/*
	for (int i = 0; i < memmoryAccess.size(); i++)
	{
		if (memmoryAccess[i]->gpubuffer->host_ptr == buffer->host_ptr)
		{
			if (mode == ACCESS_WRITE)		memmoryAccess[i]->writecnt += 1;
			else memmoryAccess[i]->readcnt += 1;
			
			find = true;
			break;
		}
	}

	if (!find)
	{
		memmoryAccess[buffer->host_ptr] = new memStats(!mode, mode, buffer->size, buffer);
	}
*/
	
}


int gpuMemHandler::WriteBuffer( gpuBuffer* buffer, cl_bool blocking, size_t offset, bool force)
{
	if (buffer->lastOperation == ACCESS_WRITE)
	{
		doubleWriteAttemps++;
	}

	computeAccess(buffer, ACCESS_WRITE);
	buffer->lastOperation = ACCESS_WRITE;

	if (demoMode) return 0;

	startProcess((char*)"GPU_WRITE");
	int error = 0;
	cl_command_queue cq_plain = clUtils::getDefaultQueue()();
	if (buffer->state != GPU_STATE_GPU || force)
	{
		 error = clEnqueueWriteBuffer(cq_plain, buffer->mem(), blocking, 0, buffer->size, buffer->host_ptr, 0, NULL, NULL);
	}
	buffer->state = GPU_STATE_GPU;
	endProcess((char*)"GPU_WRITE");
	return error;
}

int gpuMemHandler::ReadBuffer( gpuBuffer* buffer,  cl_bool blocking, size_t offset, bool force)
{
	std::string st = "GPU_Read_";
	
	computeAccess(buffer, ACCESS_READ);
	buffer->lastOperation = ACCESS_READ;

	if (demoMode) return 0;

	startProcess((char*)st.c_str());
	cl_command_queue cq_plain = clUtils::getDefaultQueue()();
	int error = 0;
	
	if (buffer->state != GPU_STATE_GPU || force || debugMode)
	{
		error = clEnqueueReadBuffer(cq_plain, buffer->mem(), blocking, 0, buffer->size, buffer->host_ptr, 0, NULL, NULL);
	}
	endProcess((char*)st.c_str());
	return error;

}

gpuBuffer openclGenArray(void* data, size_t size)
{
	return *gpuMemHandler::getBuffer(data, false);

}

