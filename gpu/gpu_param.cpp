// =================================================================================================
// This file is part of the YoloCL project. This project is a convertion from YOLO project.
// The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Juan DAmato <juan.damato@gmail.com>
// =================================================================================================


/////////////////////////////////////
///////class gpu_param
/////////////////////////////////////
#include "gpu_param.h"
#include "cl_utils.h"

gpu_param::gpu_param(gpuBuffer* buffer) 
{ 
	_gpuBuffer = buffer; type = BUFFER_PARAM; size = buffer->size; dataChanged = 0;
}

gpu_param::gpu_param(size_t v) 
{
	ivalue = (int)v;  type = INT_PARAM;  size = 4; dataChanged = 0; pvalue = NULL; fvalue = 0.0f;
}

gpu_param::gpu_param(int v) { ivalue = v; type = INT_PARAM; size = 4; dataChanged = 0;	pvalue = NULL; fvalue = 0.0f; }

gpu_param::gpu_param(void *v)
{
	pvalue = v;
	type = BUFFER_PARAM;
	this->size = _msize(v);
	dataChanged = 0;

}

gpu_param::gpu_param(int *v)
{
	pvalue = v;
	type = BUFFER_PARAM;
	this->size = _msize(v);
	dataChanged = 0;
	this->_gpuBuffer = gpuMemHandler::getBuffer(v);

}

gpu_param::gpu_param(float v)
{
	fvalue = v;
	type = FLOAT_PARAM;
	dataChanged = 0; size = 4; pvalue = NULL; ivalue = 0;
}
gpu_param::gpu_param(float *v)
{
	pvalue = v;
	dataChanged = 0;
	type = BUFFER_PARAM;
	this->size = _msize(v);
	this->_gpuBuffer = gpuMemHandler::getBuffer(v);
}
float gpu_param::getfValue() { return fvalue; }
float* gpu_param::getpfValue() { return (float*)pvalue; }
int gpu_param::getiValue() { return ivalue; }
int* gpu_param::getpiValue() { return (int*)pvalue; }

int gpu_param::setParamToDevice(cl::Kernel clk, int order)
{
	int iclError = 0;

	if (this->type == FLOAT_PARAM)
		iclError |= clk.setArg(order, fvalue);
	else
		if (this->type == INT_PARAM)
			iclError |= clk.setArg(order, ivalue);
		else
			if (this->type == BUFFER_PARAM)
			{
				_gpuBuffer = gpuMemHandler::getBuffer(pvalue);

				gpuMemHandler::WriteBuffer(_gpuBuffer, CL_TRUE, 0);


				iclError |= clk.setArg(order, _gpuBuffer->mem());
			}
	return iclError;
}



int gpu_param::readParamFromDevice(cl::Kernel clk, int order)
{
	int iclError = 0;
	if (this->type == BUFFER_PARAM)
	{
		_gpuBuffer = gpuMemHandler::getBuffer(pvalue);
		iclError = gpuMemHandler::ReadBuffer(_gpuBuffer, CL_TRUE, 0, true);
	}
	return iclError;
}


void kernelCall(std::string kernelName, int dims, int partitions, std::vector<gpu_param> args, std::vector<gpu_param> outputs )
{

	cl_int iclError = 0;
	// Kernels
	cl::Kernel clkKernel = cl::Kernel(clUtils::getDefaultProgram(), kernelName.c_str(), &iclError);

	//--- Init Kernel arguments ---------------------------------------------------
	for (int i = 0; i < args.size(); i++)
	{

		iclError |= args[i].setParamToDevice(clkKernel, i);

		if (iclError)
		{
			std::cout << "Error at set Param" << kernelName << " args:" << i << "\n";
		}
	}

	clUtils::assertCL(iclError);

	//invoke
	if (gpuMemHandler::getEmulationMode())
	{

	}
	else
	{
		iclError |= clUtils::getDefaultQueue().enqueueNDRangeKernel(clkKernel, cl::NullRange, clUtils::opencl_gridSize(dims, partitions), partitions);
		iclError |= cl::finish();
		clUtils::assertCL(iclError);
	}
	// read 
	for (int i = 0; i < outputs.size(); i++)
	{
		iclError |= outputs[i].readParamFromDevice(clkKernel, i);
	}
	clUtils::assertCL(iclError);


}

