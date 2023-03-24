// =================================================================================================
// This file is part of the YoloCL project. This project is a convertion from YOLO project.
// The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Juan DAmato <juan.damato@gmail.com>
// =================================================================================================

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the C++ OpenCL API. If not yet available, it can be found here:
// https://www.khronos.org/registry/cl/api/1.1/cl.hpp
#include <CL\cl.hpp>
#include <CL\cl.h>
// Includes the CLBlast library
#include <clblast.h>
// YOLO Includes
#include "../include/darknet.h"
#include "../src/yolo_layer.h"
#include "../src/softmax_layer.h"
#include "../src/reorg_layer.h"
#include "../src/region_layer.h"
//#include "../src/half.hpp"
#include "gpu_param.h"
#include "cl_MemHandler.h"
#include "cl_utils.h"
#include "yoloopenCL.h"
#include "../u_ProcessTime.h"
#include "../liteBlast/liteCLBlast.h"
#include "../src/shortcut_layer.h"

//////////////////////////////////////
/// CONVOLUTIONAL
///////////////////////////////////////
#include <assert.h>
#include <vector>

dim3 blockIdx, blockDim, threadIdx, gridDim;

liteClBlast::LiteClBlast lclB;
/// GLOBAL VARS
int writeCounter = 0;
int readCounter = 0;
int sizeWrite = 0;
int sizeRead = 0;
std::vector<gpu_param> bufferList;
gpuBuffer device_a, device_b, device_c;
bool emulationMode = false;
int blastInitialized = false;

// GENERAL METHODS

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
	
}


void opencl_set_device(int gpu_index)
{

}

//int nframe = 0;

void _binarize_cpu(float *input, int n, float *binary)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		binary[i] = (input[i] > 0.0f) ? 1.0f : -1.0f;
	}
}

void _swap_binary(layer *l)
{
	float *swap = l->weights;
	l->weights = l->binary_weights;
	l->binary_weights = swap;
}

int entry_indexF(int l_w, int l_h, int l_classes, int l_outputs, int batch, int location, int entry)
{
	int n = location / (l_w*l_h);
	int loc = location % (l_w*l_h);
	return batch*l_outputs + n*l_w*l_h*(4 + l_classes + 1) + entry*l_w*l_h + loc;
}


void __softmax(float *input, int n, float temp, int stride, float *output)
{
	int i;
	float sum = 0;
	float largest = -FLT_MAX;
	for (i = 0; i < n; ++i)
	{
		if (input[i*stride] > largest) largest = input[i*stride];
	}
	for (i = 0; i < n; ++i)
	{
		float e = exp(input[i*stride] / temp - largest / temp);
		sum += e;
		output[i*stride] = e;
	}
	for (i = 0; i < n; ++i)
	{
		output[i*stride] /= sum;
	}
}


void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	int g, b;
	for (b = 0; b < batch; ++b)
	{
		for (g = 0; g < groups; ++g)
		{
			__softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
		}
	}
}

void forward_softmax_layerCL(layer& l, network* net)
{
	gpuBuffer* b = gpuMemHandler::getBuffer(net->input);
	gpuBuffer* c = gpuMemHandler::getBuffer(l.output);

	gpuMemHandler::ReadBuffer(b, CL_TRUE, 0,true);
	gpuMemHandler::ReadBuffer(c, CL_TRUE, 0,true);
	softmax_gpu(net->input, l.inputs / l.groups, l.batch, l.inputs, l.groups, l.inputs / l.groups, 1, l.temperature, l.output);
	gpuMemHandler::WriteBuffer(c, CL_TRUE, 0, true);

}

//int* indexes = NULL;
//int* lengths = NULL;


void _scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX)
{
	int i;
	for (i = 0; i < N; ++i) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}


int _entry_index(layer l, int batch, int location, int entry)
{
	int n = location / (l.w*l.h);
	int loc = location % (l.w*l.h);
	return batch * l.outputs + n * l.w*l.h*(4 + l.classes + 1) + entry * l.w*l.h + loc;
}

void _forward_yolo_layerCL(layer& l, network_state& state, int mode)
{
	int input_size = _msize(l.output) / 4;


	// delta is zeroed
	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

	if (mode == 0)
	{
		/*
		gpuBuffer* c = gpuMemHandler::getBuffer(l.output);

		kernelCall("kernel_copy_offset", clUtils::opencl_gridSize(input_size, 256), 256, { input_size  , net->input ,1,l.output,1,0 }, { l.output });

		//if (!l.indexAux) l.indexAux = new int[l.batch * l.n * 2 ];
		//if (!l.lenghts) l.lenghts = new int[l.batch * l.n * 2 ];
		int i = 0;
		//Compute indexes
		for (int b = 0; b < l.batch; b++)
			for (int n = 0; n < l.n; n++)
			{
				int index0 = entry_indexF(l.w, l.h, l.classes, l.outputs, b, n*l.w*l.h, 0);
				int	index1 = entry_indexF(l.w, l.h, l.classes, l.outputs, b, n*l.w*l.h, 4);
				l.indexAux[i + 0] = index0;
				l.indexAux[i + 1] = index1;
				l.lenghts[i + 0] = 2 * l.w*l.h;
				l.lenghts[i + 1] = (1 + l.classes)*l.w*l.h;
				i += 2;

			}
		int lastIndex = l.lenghts[l.batch * l.n * 2 - 1] + l.indexAux[l.batch * l.n * 2 - 1];

		gpuBuffer* ib = gpuMemHandler::getBuffer(l.indexAux, true, false, CL_MEM_READ_ONLY);
		gpuBuffer* il = gpuMemHandler::getBuffer(l.lenghts, true, false, CL_MEM_READ_ONLY);

		gpuMemHandler::WriteBuffer(ib, CL_TRUE, 0, true);
		gpuMemHandler::WriteBuffer(il, CL_TRUE, 0, true);

		kernelCall("kernel_yolo_layerAcc", clUtils::opencl_gridSize(l.outputs, 256), 256, { l.output ,l.indexAux,l.lenghts,l.n*l.batch * 2 }, { l.output });

		gpuMemHandler::ReadBuffer(c, CL_TRUE, 0, true);
		*/
	}
	else
	{
		// uncomment to debug
		gpuBuffer* c = gpuMemHandler::getBuffer(l.output);
			int i, j, b, t, n;
			memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
			for (b = 0; b < l.batch; ++b) 
			{
				for (n = 0; n < l.n; ++n) 
				{
					int index = _entry_index(l, b, n*l.w*l.h, 0);
					activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);        // x,y,
					_scal_add_cpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + index, 1);    // scale x,y
					index = _entry_index(l, b, n*l.w*l.h, 4);
					activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
				}
			}
#endif

			gpuMemHandler::WriteBuffer(c, CL_TRUE, 0, true);
			
	}



}


void _copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
	int i;
	for (i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void _forward_upsample_layerCL(layer& l, network* net)
{

	size_t SIZE = l.outputs*l.batch;
	// Copy the matrices to the device

	kernelCall("kernel_upsample", clUtils::opencl_gridSize(net->inputs, 256), 256, 
		                    { net->input,l.w,l.h,l.c,l.batch, l.stride, 1, l.scale,  l.output }, { l.output });


}

void _forward_maxpool_layer(layer& l, network* net)
{
	int SIZE = l.batch * l.c * l.out_h* l.out_w;

	int iclError = 0;
	kernelCall("kernel_maxpool", clUtils::opencl_gridSize(SIZE, 256), 256, { l.out_h,l.out_w,l.c,l.h,l.w,l.stride,l.pad,l.size,l.output,l.indexes, net->input }, { l.output,l.indexes });
	clUtils::assertCL(iclError);
}

void _forward_shortcut_layer(layer& l, network* net)
{
	int iclError = 0;

	size_t sizeout = _msize(l.output);

	int SIZE = l.outputs*l.batch;

	SIZE = l.outputs*l.batch;
	kernelCall("kernel_copy", clUtils::opencl_gridSize((int)_msize(net->input), 256), 256, { l.outputs*l.batch , net->input ,1,l.output,1 }, { l.output });

	// Kernel ShortCut
	//if (abs(l.alpha * l.beta) > 0.0)
	{
		l.alpha = 1.0;
		l.beta = 1.0;
		kernelCall("kernel_shortcut", clUtils::opencl_gridSize(_msize(net->layers[l.index].output), 256), 256, { l.batch, l.w, l.h, l.c, net->layers[l.index].output,
																	   l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output }, { l.output });

	}
	//		activate_array_opencl(device_out, l.outputs*l.batch, l.activation);
	kernelCall("kernel_activate", clUtils::opencl_gridSize(l.outputs*l.batch, 256), 256, { l.output, l.activation,l.outputs*l.batch }, { l.output });

	clUtils::assertCL(iclError);
}

void _forward_shortcut_layer_v2(layer& l, network* net)
{
	int iclError = 0;

	size_t sizeout = _msize(l.output);

	int size = l.batch * l.w * l.h * l.c;
	kernelCall("kernel_shortcut_v2", clUtils::opencl_gridSize(l.outputs*l.batch, 256), 256, 
		                         { net->input,net->layers[l.index].output ,size ,l.output }, { l.output });

	//		activate_array_opencl(device_out, l.outputs*l.batch, l.activation);
	kernelCall("kernel_activate", clUtils::opencl_gridSize(l.outputs*l.batch, 256), 256, { l.output, l.activation,l.outputs*l.batch }, { l.output });


	clUtils::assertCL(iclError);
}

void _forward_avgpool_layerCL(layer &l, network* net)
{

	kernelCall("kernel_avg_pool", clUtils::opencl_gridSize(l.batch * l.c, 256), 256, { net->input ,l.output, l.h, l.w,l.c,l.batch }, { l.output });


}

void forward_convolutional_layerCL(layer& l, network* net, bool& ouput, bool checkIsZero)
{
	int iclError = 0;

	int SIZE = l.outputs*l.batch * 1;

	// OPT 0
	kernelCall("kernel_fill", clUtils::opencl_gridSize(SIZE, 256), 256, { SIZE,l.output,1,0.0f }, { l.output });

	if (l.xnor)
	{
		std::cout << " NOT IMPLEMENTED " << "\n";
		//binarize_weights(l.weights, l.n, l.c / l.groups*l.size*l.size, l.binary_weights);
		//_swap_binary(&l);
		//_binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
		//net.input = l.binary_input;
	}
	clUtils::assertCL(iclError);

	int m = l.n / l.groups;
	int k = l.size*l.size*l.c / l.groups;
	int n = l.out_w*l.out_h;
	int channels = l.c / l.groups;
	int SIZE_SRC = channels * l.h * l.w;

	//startProcess("GEMM");

	if (l.size == 1)
	{
		auto queue_plain = clUtils::getDefaultQueue()();

		gpuBuffer* a = gpuMemHandler::getBuffer(l.weights);
		gpuBuffer* b = gpuMemHandler::getBuffer(net->input);
		gpuBuffer* c = gpuMemHandler::getBuffer(l.output);

		gpuMemHandler::WriteBuffer(a, CL_TRUE, 0);
		gpuMemHandler::WriteBuffer(b, CL_TRUE, 0);
#ifdef USE_OPENCL
#else 
		auto status2 = clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
			(size_t)m, (size_t)n, (size_t)k, 1.0f, a->mem(), 0, k, b->mem(), 0, n,
			1.0f, c->mem(), 0, n, &queue_plain, NULL);
#endif
		gpuMemHandler::ReadBuffer(c, CL_TRUE, 0);
		
	}
	else
	{
		int height_col = (l.h + 2 * l.pad - l.size) / l.stride + 1;
		int width_col = (l.w + 2 * l.pad - l.size) / l.stride + 1;
		int channels_col = channels * l.size * l.size;
		int SIZE_DST = height_col * width_col * channels_col;

		kernelCall("kernel_im2col", clUtils::opencl_gridSize(SIZE_DST, 256), 256, 
			             { net->input,net->workspace, l.h, l.w, l.pad,l.size, l.stride, channels }, { net->workspace });

		auto queue_plain = clUtils::getDefaultQueue()();
		gpuBuffer* a = gpuMemHandler::getBuffer(l.weights);
		gpuBuffer* b = gpuMemHandler::getBuffer(net->workspace);
		gpuBuffer* c = gpuMemHandler::getBuffer(l.output);

		gpuMemHandler::WriteBuffer(a, CL_TRUE, 0);
		gpuMemHandler::WriteBuffer(b, CL_TRUE, 0);
#ifdef USE_OPENCL
#else 
		// Call the SGEMM routine. Note that the type of alpha and beta (float) determine the precision.	
		auto status2 = clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
			(size_t)m, (size_t)n, (size_t)k, 1.0f, a->mem(), 0, k, b->mem(), 0, n,
			1.0f, c->mem(), 0, n, &queue_plain, NULL);
#endif		
			
		gpuMemHandler::ReadBuffer(c, CL_TRUE, 0);
	
	}
//	clUtils::getDefaultQueue().finish();
//	endProcess("GEMM");


//
	if (l.batch_normalize)
	{
		if (l.type == BATCHNORM)
		{
			kernelCall("kernel_copy", clUtils::opencl_gridSize(l.outputs*l.batch, 256), 256, { l.outputs*l.batch, net->input, 1, l.output, 1 }, { l.output });
		}

		// copy from OUTPUT to X
		int SIZE = l.outputs*l.batch * 1;
		//kernelCall("kernel_copy", opencl_gridSize(SIZE, 256), 256, { l.outputs*l.batch, l.output, 1, net->input, 1 }, {});
		// normalize OUTPUT
		SIZE = l.batch * l.out_c * l.out_h*l.out_w;
		kernelCall("kernel_normalize", clUtils::opencl_gridSize(SIZE, 256), 256, { l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w }, { l.output });
		//scale Bias
	
		
		kernelCall("kernel_scale_bias", clUtils::opencl_gridSize(SIZE, 256), 256, { l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w }, { l.output });
		//add Bias
		kernelCall("kernel_add_bias", clUtils::opencl_gridSize(SIZE, 256), 256, { l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w }, { l.output });
		//activate
		if (l.activation == MISH)
		{
			kernelCall("kernel_activate_mish", clUtils::opencl_gridSize(l.outputs*l.batch, 256), 256, 
				           { l.output, l.activation_input,l.outputs*l.batch }, { l.output,l.activation_input });
		}
		else
		{
			kernelCall("kernel_activate", clUtils::opencl_gridSize(l.outputs*l.batch, 256), 256, { l.output, l.activation,l.outputs*l.batch }, { l.output });
		}
		ouput = true;
		clUtils::assertCL(iclError);
	}
	else
	{
		int SIZE = l.batch*l.n* l.out_h*l.out_w;
		//add Bias
		kernelCall("kernel_add_bias", clUtils::opencl_gridSize(SIZE, 256), 256, { l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w }, { l.output });
			
		//activate
		if (l.activation == MISH)
		{
			kernelCall("kernel_activate_mish", clUtils::opencl_gridSize(l.outputs*l.batch, 256), 256,
				{ l.output,l.outputs*l.batch }, { l.output });
		}
		else
		{
			kernelCall("kernel_activate", clUtils::opencl_gridSize(l.outputs*l.batch, 256), 256, { l.output, l.activation,l.outputs*l.batch }, { l.output });
		}
		clUtils::assertCL(iclError);
	}


	if (l.binary || l.xnor) _swap_binary(&l);

//	endProcess("Normalize");
}


namespace YoloCL
{

	cl::Program yoloProgram;

	int memAllocationScheme = 1; //dynamic -1, fixed 1
	

	void _forward_route_layer(layer& l, network_state& state, bool useGpu)
	{

		int i;
		int offsetY = 0;
		int offsetX = 0;
		int j = 0;

		for (i = 0; i < l.n; i++)
		{
			int index = l.input_layers[i];
			float *input = state.net.layers[index].output;
			int input_size = l.input_sizes[i];
			int part_input_size = input_size / l.groups;
			float* ouput = l.output;
			if (useGpu)
			{
			//	gpuBuffer *c = gpuMemHandler::getBuffer(input);
			//	gpuMemHandler::ReadBuffer(c, CL_TRUE, 1, true);
			}
			//_copy_cpu(part_input_size, input + j * input_size + part_input_size * l.group_id, 1, l.output + offsetY + j * l.outputs, 1);
				kernelCall("kernel_copy_offset", clUtils::opencl_gridSize(input_size, 256), 256, 
					                 { part_input_size  ,input ,1,ouput,1,offsetY,offsetX }, { ouput });
			// YoloCL::computAndPrint(input, input_size, index*10+i);
			offsetY += input_size;
		}

		if (useGpu)
		{
		//	gpuBuffer *o = gpuMemHandler::getBuffer(l.output);
		//	gpuMemHandler::WriteBuffer(o, CL_TRUE, 1, true);
			
		}

		//YoloCL::computAndPrint(l.output, l.outputs, 1111);
	}

	detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
	{
		detection *dets = NULL;
		return dets;
	}


	void setAllocationScheme(int memMode, size_t totalAvailableMem)
	{
		if (memMode < 0)
		{
			std::cout << " MemMgr :: using dynamic allocation mode" << "\n";
		}
		else
		{
			std::cout << " MemMgr :: using pre-allocation mode" << "\n";
		}
		memAllocationScheme = memMode;
		gpuMemHandler::setMaxUsedMem(totalAvailableMem * 1024 * 1024);
	}

	///////////////////////////////////////////////////////////////////////
	// ITERATE THROUGH ALL LAYERS
	///////////////////////////////////////////////////////////////////////

	void predict(network *net, network_state& state, int nframe, int yoloMode, bool checkIsZero)
	{
				
		for (int i = 0; i < net->n; ++i)
		{
			//std::cout << i << ".";
			clUtils::setActiveProgram(yoloProgram);
			gpuMemHandler::startGPUInvocation();
			
			net->index = i;
			state.index = i;

			bool hasToSwap = false;


			switch (net->layers[i].type) {
			case CONVOLUTIONAL:
				forward_convolutional_layerCL(net->layers[i], net, hasToSwap, checkIsZero);
				break;
			case MAXPOOL:
				_forward_maxpool_layer(net->layers[i], net);
				break;
			case ROUTE:
				_forward_route_layer(net->layers[i], state);
				break;
			case UPSAMPLE:
				_forward_upsample_layerCL(net->layers[i], net);
				break;
			case SHORTCUT:
			{
				gpuBuffer* c = gpuMemHandler::getBuffer(net->input);
				gpuMemHandler::ReadBuffer(c, CL_TRUE, 0, true);

				_forward_shortcut_layer_v2(net->layers[i], net);
			}
			break;
			case YOLO:
			{
				// RUNS on CPU
				gpuBuffer* c = gpuMemHandler::getBuffer(net->input);
				gpuMemHandler::ReadBuffer(c, CL_TRUE, 0, true);

				_forward_yolo_layerCL(net->layers[i], state, 1);
			}
			break;
			case AVGPOOL:
				_forward_avgpool_layerCL(net->layers[i], net);
				break;
			case SOFTMAX:
			{
				//force a buffer read
				gpuBuffer* c = gpuMemHandler::getBuffer(net->input);
				gpuMemHandler::ReadBuffer(c, CL_TRUE, 0, true);
				forward_softmax_layerCL(net->layers[i], net);
			}
				break;
			case REORG:
				startProcess((char*)"reorg");
				//forward_reorg_layer(net->layers[i], *net);
				std::cout << "Not yet implemented" << net->layers[i].type << "\n";
				endProcess((char*)"reorg");
				break;
			case REGION:
				startProcess((char*)"region");
				//forward_region_layer(net->layers[i], *net);
				std::cout << "Not yet implemented" << net->layers[i].type << "\n";
				endProcess((char*)"region");
				break;

			default:
				//
				std::cout << "Not yet implemented" << net->layers[i].type << "\n";
			}
			
			net->input = net->layers[i].output;
			state.input = net->layers[i].output;
			

			// uncomment for debugging in detail
			/*if (i > 100)
			{

				gpuBuffer* c = gpuMemHandler::getBuffer(net->layers[i].output);
				gpuMemHandler::ReadBuffer(c, CL_TRUE, 0, true);
				YoloCL::computAndPrint(net->layers[i],i);
			}
		*/
			int SIZE = net->layers[i].outputs*net->layers[i].batch;
			int iclError = 0;


			clUtils::assertCL(iclError);
			gpuMemHandler::endGPUInvocation();

		
		}

		gpuBuffer* c = gpuMemHandler::getBuffer(net->input);
		gpuMemHandler::ReadBuffer(c, CL_TRUE, 0, true);

		clUtils::getDefaultQueue().finish();
		// Not used right now
		// calc_network_cost(net);
		//std::cout << "\n";
	}

	float *network_predict(network *net, float *input, int nframe, bool checkIsZero)
	{
	network orig = *net;
	net->input = input;
	net->truth = 0;
	net->train = 0;
	net->delta = 0;

	network_state state = { 0 };
	net->seen += net->batch;
	state.index = 0;
	state.net = *net;
	state.delta = 0;
	state.train = 0;
	state.input = input;


	predictYoloCL(net, state, nframe,1, checkIsZero);
	float *out = net->output;
	*net = orig;
	return out;
	}

	bool firstFrame = true;
	void predictYoloCL(network *net, network_state& state, int nframe, int yoloMode, bool checkIsZero)
	{
	
		int iclError = 0;
		clUtils::setActiveProgram(yoloProgram);
		

		// Start the timer
#ifdef DEBUG
		gpuMemHandler::setDebugMode(true);
#else
		gpuMemHandler::setDebugMode(false);
#endif
		gpuMemHandler::startGPULoop();
		
		if (firstFrame)
		{
			std::cout << "... RUNNING OPENCL YOLO VERSION 1.0 ...." << "\n";
			//// Compute the amount of neccesary aux BUFFERs
			int numRouteLayers = 0;

			size_t auxMem = 0;
			size_t mxSize = 0;
			// create extra indexes

			for (int i = 0; i < net->n; i++)
			{
				layer* l = &net->layers[i];

				if (l->type == YOLO)
				{
					// aux data
					l->indexAux = new int[l->batch * l->n * 2];
					l->lenghts = new int[l->batch * l->n * 2];
				}

				size_t size = _msize(l->output);
				auxMem += size;
				mxSize = MAX(mxSize, size);
				if ((l->type == ROUTE) || (l->type == SHORTCUT))
				{
					numRouteLayers++;
				}
			}

			std::cout << "... ... estimating mem req ...." << "\n";
			gpuMemHandler::setDebugMode(true);
			predict(net, state,nframe, yoloMode, false);
			gpuMemHandler::setDebugMode(false);
	
			std::cout << "... ... computing optimal mem req ...." << "\n";
			gpuBuffer* input = gpuMemHandler::getBuffer(net->input, true, CL_MEM_READ_WRITE, 1, "INPUT");
			gpuBuffer* workspace = gpuMemHandler::getBuffer(net->workspace, true, CL_MEM_READ_WRITE, 1, "WORKSPACE");
		
		////////////////////////////////
			/// OPTIMIZATION 1 : OUTPUTS ARE MANAGED AS INNER SHARED
			gpuMemHandler::allocateAuxBuffer(mxSize,2);

			for (int i = 0; i < net->n; i++)
			{
				layer* l = &net->layers[i];
								
				if (!l->output) continue;
				//assign the same space
				memStats* ms = gpuMemHandler::getStats(l->output);

				if ((ms) &&(ms->readcnt == ms->writecnt))
				{
					gpuMemHandler::getBuffer(l->output, true, CL_MEM_READ_WRITE, 1, "output" + to_string(i));
					//gpuMemHandler::getSharedBuffer(l->output, true, CL_MEM_READ_WRITE, 1, "outputSHARED" + to_string(i));
				}
				else
				{
					gpuMemHandler::getBuffer(l->output, true, CL_MEM_READ_WRITE, 1, "output" + to_string(i));
				}

			
			}

			std::cout << "Allocating (AUX):" << auxMem << " Neccesary buffers " << numRouteLayers+2 << "\n";
			std::cout << "MAX BUFFER SIZE:" << mxSize << "\n";

			//gpuMemHandler::printStats();
			gpuMemHandler::clearStats();
			////////////////////////////////
			/// OPTIMIZATION 2 : FIXED MEM COULD BE SHARED BETWEEN APPLICATIONS
			// Compute the amount of neccesary memmory
			for (int i = 0; i < net->n; i++)
			{
				layer* l = &net->layers[i];
				if (l->rolling_mean) gpuMemHandler::getBuffer(l->rolling_mean, true, CL_MEM_READ_ONLY, 1, "rolling_mean" + to_string(i));
				if (l->scales) gpuMemHandler::getBuffer(l->scales, true,  CL_MEM_READ_ONLY, 1, "scales" + to_string(i));
				if (l->biases) gpuMemHandler::getBuffer(l->biases, true,  CL_MEM_READ_ONLY, 1, "biases" + to_string(i));
				if (l->weights) gpuMemHandler::getBuffer(l->weights, true,  CL_MEM_READ_ONLY, 1, "weights" + to_string(i));
				if (l->indexes) gpuMemHandler::getBuffer(l->indexes, true,  CL_MEM_READ_WRITE, 1, "indexes" + to_string(i));
				if (l->rolling_variance) gpuMemHandler::getBuffer(l->rolling_variance, true, CL_MEM_READ_ONLY, 1, "rolling_variance" + to_string(i));
				if (l->lenghts) gpuMemHandler::getBuffer(l->lenghts, true, CL_MEM_READ_WRITE, memAllocationScheme, "lenghts" + to_string(i));
				if (l->indexAux) gpuMemHandler::getBuffer(l->indexAux, true, CL_MEM_READ_WRITE, memAllocationScheme, "indexAux" + to_string(i));
				
			}

			firstFrame = false;
		}
		else
		{
			gpuMemHandler::setDebugMode(yoloMode);
			gpuBuffer* input = gpuMemHandler::getBuffer(net->input, true, CL_MEM_READ_WRITE, 1, "INPUT");
			gpuBuffer* workspace = gpuMemHandler::getBuffer(net->workspace, true, CL_MEM_READ_WRITE, 1, "WORKSPACE");
			gpuMemHandler::WriteBuffer(input, true, 0, true);
			int SIZE = _msize(net->workspace) / 4;
			kernelCall("kernel_fill", clUtils::opencl_gridSize(SIZE, 256), 256, { SIZE,net->workspace,1,0.0f }, { net->workspace });
			predict(net, state, nframe, yoloMode, checkIsZero);
		}
	
		
		
		gpuMemHandler::endGPULoop();
	}


	std::vector<object_detected> makeDetections(detection *dets, int num, float thresh, char **names, int classes)
	{
		std::vector<object_detected> objects;
		for (int i = 0; i < num; ++i)
		{
			object_detected objC;
			int _class = -1;
			float prob = 0.0f;
			for (int j = 0; j < classes; ++j)
			{
				if (dets[i].prob[j] > thresh)
				{
					prob = dets[i].prob[j] * 100;

					objC.classes.push_back(std::make_pair(names[j], dets[i].prob[j] * 100));
					_class = j;
				}
			}
			if (_class >= 0)
			{
				box b = dets[i].bbox;
				objC.reg = b;
				objects.push_back(objC);
			}
		}

		return objects;
	}


	std::vector<object_detected> detectOnFrameCNN(network *net, float* frameData,int width, int height, int nframe, char** names)
	{

		//Resize for network
		int key = 0;
		net->input = frameData;
		net->truth = 0;
		net->train = 0;
		net->delta = 0;

		// predict
		int origLayers = net->n;
		int lIndex = 3;

		network_state state = { 0 };
		net->seen += net->batch;
		state.index = 0;
		state.net = *net;
		state.delta = 0;
		state.train = 0;
		state.input = frameData;

		YoloCL::predictYoloCL(net, state, 1, 0);

			
		// read results
		int nboxes = 0;
		layer l = net->layers[net->n - 1];
		detection* dets = YoloCL::get_network_boxes(net, width, height, 0.5, 0.5, 0, 1, &nboxes);
		float nms = 0.45f;
		
		std::vector<object_detected> detections = YoloCL::makeDetections(dets, nboxes, 0.45, names,  l.classes);
		return detections;
	}

	int initYoloCL(std::string cl_dir, int localWG, int platformProcessingIndex, int deviceProcessingIndex)
	{
		int error = 0;
	
			
		yoloProgram = clUtils::loadCLSources(cl_dir + "yolo.cl", localWG, platformProcessingIndex, deviceProcessingIndex, &error);
		// If it is not GPU
		if (platformProcessingIndex == 0)
		{
			error = lclB.init(cl_dir + "clBlast_Kernels.cl", 256, clUtils::getDefaultContext(), clUtils::getDefaultDevice());
			if (error == 0)
			{
				blastInitialized = true;
			}
		}
		else
		{
			// nothing
		}
		
	
		int supported = 0;
		// Check if device has own memmory or add a flag
		clUtils::getDefaultDevice().getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &supported);
		gpuMemHandler::setDeviceHasOwnMem(!supported);
		return error;

	}

	void computAndPrint(float* input, int length, int iter, int layer_index)
	{
		float avg_val = 0;
		int k;
		for (k = 0; k < length; ++k) avg_val += input[k];
		printf(" it %d Layer(%d) - avg_val = %f  first value %f \n", iter, layer_index, avg_val / length, input[0]);
	}
	void computAndPrint(layer& l, int iter)
	{
		computAndPrint(l.output, l.outputs,iter, l.index);
	}
	
}
