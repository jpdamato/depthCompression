#include "clEncoder.h"
#include "cl_utils.h"
#include "gpu_param.h"

#include "../u_ProcessTime.h"


#ifdef ZSTD
#include "zstd.h"
#endif

namespace FitCL
{
	cl::Program encoderProgram;

	cl_memManager* mmgr = NULL;
	spline_ptr splines = new cl_spline_data[50000];
	int max_splines = 0;

	ZSTD_CDict* cdictPtr = NULL;
	ZSTD_DDict* ddictPtr = NULL;

	size_t dictionaryCompression(char* srcBuffer, size_t srcSize)
	{
		size_t outSize = 0;

		size_t const cBuffSize = ZSTD_compressBound(srcSize);

		return cBuffSize;

	}


	bool cl_splineSort(void* p0, void* p1)
	{
		spline_ptr sp0 = (spline_ptr)p0;
		spline_ptr sp1 = (spline_ptr)p1;

		if (!sp0 || !sp1) return 0;

		

		return ( (sp0->y0*2000+ sp0->x0) < (sp1->y0 * 2000 + sp1->x0));
	}


	std::vector<spline_ptr> getSplines()
	{
		
		std::vector<spline_ptr> sp;

		for (int i = 0; i < max_splines; i++)
		{
			sp.push_back(&splines[i]);
		}
		return sp;
	}

	gpuBuffer* gb_Mgr;
	gpuBuffer* gb_Image;
	gpuBuffer* gb_Splines;

	bool runOnGPU = true;
	spline_ptr allocated_splines = NULL;
	unsigned short* pixels = NULL;
	unsigned short* vectorized = NULL;
	int outputSize = 0;
	int w, h;

	void testMemAssigment(cv::Mat& m)
	{
		cl_memManager* mmgr_out = new cl_memManager();
		gpuMemHandler::getBuffer(mmgr, true, 2, -1, "manager_temp");

#pragma omp parallel for
		for (int y = 0; y < m.rows; y++)
		{
			cl_test(m.cols, (Short_PTR)pixels, CL_SPLINE_COMPRESSION, 5, mmgr, 0);
		}
		std::cout << " Test CPU before ";
		for (int i = 0; i < m.rows; i++)
		{
			if (i < 20)	std::cout << mmgr->allocated_ids[i] << ";";

			mmgr->allocated_ids[i] = CL_ALLOCATION_PER_THREAD * i;
		}
		std::cout << "\n";

		gpuMemHandler::startGPUInvocation();
		clUtils::setActiveProgram(encoderProgram);
		kernelCall("cl_test", clUtils::opencl_gridSize(m.rows, 4), 4,
			{ m.cols,pixels,CL_SPLINE_COMPRESSION,5, mmgr_out,0 }, { mmgr_out });
		gpuMemHandler::endGPUInvocation();

		std::cout << " Test GPU after ";
		for (int i = 0; i < m.rows; i++)
		{
			if (i < 20)	std::cout << mmgr_out->allocated_ids[i] << ";";

			mmgr_out->allocated_ids[i] = CL_ALLOCATION_PER_THREAD * i;
		}
		std::cout << "\n";
	}

	int frameIndex = 0; 
	cl_memManager* out_mmgr = NULL;

	void encodeCL(cv::Mat& m, int nthreads, bool verbose)
	{
		if (!pixels) pixels = new unsigned short[m.cols * m.rows ];
		if (!vectorized) vectorized = new unsigned short[m.cols * m.rows * 2];

		if (!out_mmgr)
			out_mmgr = new cl_memManager();
			
		// Start iteration
		if (runOnGPU)
		{
			gpuMemHandler::startGPUInvocation();
			clUtils::setActiveProgram(encoderProgram);
		}
		w = m.cols;
		h = m.rows;
		
		// copy pixel data
		for (int i = 0; i < m.cols * m.rows; i++)
		{
			pixels[i] = ((unsigned short*)( m.data))[i];
		}

		// Init manager
		if (mmgr == NULL)
		{
			mmgr = new cl_memManager();
			mmger_init(mmgr, m.rows, nthreads);

			allocated_splines = mmgr->allocated_splines;

			gb_Splines = gpuMemHandler::getBuffer(mmgr->allocated_splines, true,2, 1, "splines");
			gb_Mgr = gpuMemHandler::getBuffer(mmgr,true, 2,1, "manager");
			
		}
		
		// pass image data t oGPU
		gb_Image = gpuMemHandler::getBuffer(pixels, false, 1, 0, "image");

		mmgr->allocated_size = 0;
		max_splines = 0;

		omp_set_num_threads(m.rows);
		//run test first time
		if (frameIndex == 0)
		{
			testMemAssigment(m);
		}
		startProcess("encodeCL");

		startProcess("cl_createFromImage" );
		
		// call kernels
		if (runOnGPU)
		{
			
			
			kernelCall("cl_mem_init", clUtils::opencl_gridSize(m.rows, 16), 16,
				{ mmgr , allocated_splines,m.rows }, {  });


			kernelCall("cl_encodeRowIter", clUtils::opencl_gridSize(m.rows, 16), 16,
				{ m.cols,m.rows, pixels,CL_SPLINE_COMPRESSION,5, mmgr,allocated_splines,0  }, {  });
			
		}
		else
		{
			/// For all pixels
#pragma omp parallel for
			for (int y = 0; y < m.rows; y++)
			{
				cl_mem_init( mmgr, allocated_splines, m.rows);
			}
			/// For all pixels
#pragma omp parallel for
			for (int y = 0; y < m.rows; y++)
			{
				cl_encodeRowIter( m.cols,m.rows,  (Short_PTR)pixels, CL_SPLINE_COMPRESSION, 5, mmgr, allocated_splines, false);
			}
		}
		endProcess("cl_createFromImage");

		int64 nw = cv::getTickCount();
		std::cout << "Time " << (cv::getTickCount() - nw) / cv::getTickFrequency() << "\n";

		endProcess("encodeCL");

		startProcess("cl_vectorize_2");
		if (runOnGPU)
		{

			kernelCall("cl_vectorizeSplines", clUtils::opencl_gridSize(1, 1), 1,
				{ allocated_splines ,m.rows * CL_ALLOCATION_PER_THREAD,vectorized,mmgr, CL_SPLINE_COMPRESSION, 8, true }, { vectorized, mmgr });
			
		}
		else
		{
			
			for (int i = 0; i < mmgr->total_size; i++)
			{
				if (allocated_splines[i].effective > 0)
				{
					splines[max_splines] = allocated_splines[i];
					max_splines++;

				}
			}

			//std::sort(splines, splines + max_splines, cl_splineSort);
			cl_vectorizeSplines(splines, max_splines, (Short_PTR)vectorized, mmgr, CL_SPLINE_COMPRESSION, 8, true);
		}
		// Read Back
		endProcess("cl_vectorize_2");

		
		if (verbose)
		{
			std::cout << " cl_splines:" << mmgr->validSplines << "\n";

			std::cout << " Obtained size " << mmgr->final_OuputSize << " of " << m.cols * m.rows * 2 << "\n";
		}
		frameIndex++;

		if (runOnGPU)
		{
			gpuMemHandler::endGPUInvocation();
		}
	}

	unsigned short* cl_getOutputBuffer(int& outSize)
	{
		outSize = outputSize;
		return vectorized;
	}

	int initEncoderCL(std::string cl_dir, int localWG, int platformProcessingIndex, int deviceProcessingIndex)
	{
		int error = 0;

		
		encoderProgram = clUtils::loadCLSources(cl_dir + "encoder.cl", localWG, platformProcessingIndex, deviceProcessingIndex, &error);
		
		int supported = 0;
		// Check if device has own memmory or add a flag
		clUtils::getDefaultDevice().getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &supported);
		gpuMemHandler::setDeviceHasOwnMem(!supported);


		gpuMemHandler::setMaxUsedMem(2048 * 1024 * 1024);
		

		return error;
	}

	int cl_writeOutput(std::string fn)
	{

		if (fn != "")
		{
			std::ofstream out(fn, std::ios::out | std::ios::binary);
			if (!out)
			{
				std::cout << "Cannot open output file\n";
				return 0;
			}
			int encodedMode = CL_SPLINE_COMPRESSION;
			int zstd_compression_level = 0;

			out.write((const char*)&w, 2);
			out.write((const char*)&h, 2);
			out.write((const char*)&encodedMode, 2);
			out.write((const char*)&zstd_compression_level, 2);

			// bin mask
			out.write((const char*)vectorized, outputSize);


			out.close();

			return 0;
		}
		else
		{
			return -1;
		}
	}
	//	void draw_detections(cv::Mat im, std::vector< object_detected>& objs);
}