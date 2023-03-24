// =================================================================================================
// This file is part of the YoloCL project. This project is a convertion from YOLO project.
// The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Juan DAmato <juan.damato@gmail.com>
// =================================================================================================
#include "cl_utils.h"
#include <Windows.h>


std::string cl_utils::kernel_source;
cl::Context _defaultContext;
cl::Device _defaultDevice;
cl::Platform _defaultPlatform;
std::vector<cl::Device> all_devices;
std::vector<cl::Platform> all_platforms;
cl::CommandQueue _defaultqueue;
cl::Program _defaultProgram;
cl::Program::Sources sources;
std::vector<int> bufferSizes;

double cl_utils::sft_clock(void)
{
#ifdef _WIN32
	/* _WIN32: use QueryPerformance (very accurate) */
	LARGE_INTEGER freq , t ;
	/* freq is the clock speed of the CPU */
	QueryPerformanceFrequency(&freq) ;
	/* cout << "freq = " << ((double) freq.QuadPart) << endl; */
	/* t is the high resolution performance counter (see MSDN) */
	QueryPerformanceCounter ( & t ) ;
	return ( t.QuadPart /(double) freq.QuadPart ) ;
#else
	/* Unix or Linux: use resource usage */
	struct rusage t;
	double procTime;
	/* (1) Get the rusage data structure at this moment (man getrusage) */
	getrusage(0,&t);
	/* (2) What is the elapsed time ? - CPU time = User time + System time */
	/* (2a) Get the seconds */
	procTime = t.ru_utime.tv_sec + t.ru_stime.tv_sec;
	/* (2b) More precisely! Get the microseconds part ! */
	return ( procTime + (t.ru_utime.tv_usec + t.ru_stime.tv_usec) * 1e-6 ) ;
#endif
}


void cl_utils::cargarFuente(std::string programa)
{
    if (std::ifstream(programa.c_str())){
        std::ifstream infile;
        infile.open(programa, std::ifstream::in);
        char c = infile.get();
		kernel_source.clear();
        while (!infile.eof()){
            kernel_source.push_back(c);
            c = infile.get();
        }
        infile.close();
    }
}

int cl_utils::width;
int cl_utils::height;
/*
void cl_utils::loadDataMatToUchar(uchar *data, cv::Mat &image,int nchannels)
{
    width = image.cols;
    height = image.rows;
//#pragma omp parallel for
    for (int y=0; y<height;y++)
	{
        for (int x = 0 ; x<width ; x++)
		{
			data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 0] = image.data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 0];
            if (nchannels==3){
                data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 1] = image.data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 1];
                data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 2] = image.data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 2];
            }
        }
    }
}


int cl_utils::generateClRandom(cl::Buffer &clRandom,int sizeRandomBuff,cl::CommandQueue &queue){
    cv::RNG rng(0xFFFFFFFF);
    int* buff = new int[sizeRandomBuff];
    for (int i=0;i<sizeRandomBuff;i++){
        buff[i]= rng.operator unsigned int();
    }
    int iclError = queue.enqueueWriteBuffer(clRandom,CL_TRUE,0,sizeof(unsigned int)*sizeRandomBuff,&buff[0]);
    free(buff);
    return iclError;
}


bool cl_utils::outOfBorder(vector<cv::Point> contour,int height,int width){
    //Chequear si el blob toca el borde
    for (unsigned int i=0;i<contour.size();i++){
        if ((contour[i].x==1) || (contour[i].x==width-1) || (contour[i].y==1) || (contour[i].y==height-1))
            return false;
        if ((contour[i].x==0) || (contour[i].x==width) || (contour[i].y==0) || (contour[i].y==height))
            return false;
    }
    return true;
}
*/

namespace clUtils
{
	void exportCSV(std::string outfilename, std::vector<std::string> values, int time)
	{
		ofstream myfile(outfilename, ios::out | ios::app);
		myfile << time;
		for (int i = 0; i < values.size(); i++)
		{
			myfile << ";" << values[i];
		}
		myfile << "\n";
		myfile.close();
	}
	
	void exportCSV(std::string outfilename, std::vector<float> values, int time)
	{
		ofstream myfile(outfilename, ios::out | ios::app);
		myfile << time;
		for (int i = 0; i < values.size(); i++)
		{
			myfile << ";" << values[i] ;
		}
		myfile <<  "\n";
		myfile.close();
	}
	int opencl_gridSize(int v, int blockSize)
	{
		// Round up according to array size 
		int gridSize = (v + blockSize - 1) / blockSize;

		return gridSize * blockSize;
	}

	void assertCL(int iclError)
	{

		if (iclError != CL_SUCCESS)
		{
			std::cout << "OpenCL error : " << iclError << "\n";
		}
	}

	void logMessage(string s, int type)
	{
		std::cout << s << "\n";
	}


	cl::Context getDefaultContext()
	{
		return _defaultContext;
	}


	cl::Device getDefaultDevice()
	{
		return _defaultDevice;
	}

	cl::CommandQueue getDefaultQueue()
	{
		if (_defaultqueue() == NULL)
		{
			_defaultqueue = cl::CommandQueue(_defaultContext, _defaultDevice);
		}
		return _defaultqueue;
	}

	cl::Program getDefaultProgram()
	{
		return _defaultProgram;
	}

	void setActiveProgram(cl::Program& prog)
	{
		_defaultProgram = prog;
	}

	void setCLSource(std::string kernel_src)
	{
		std::pair<const char*, ::size_t> x(kernel_src.c_str(), kernel_src.length());
		sources.push_back(x);
	}

	cl::Program buildProgram(cl::Context _Context, cl::Device _Device, int* error)
	{
		cl::Program program = cl::Program(_Context, sources);
		VECTOR_CLASS<cl::Device> devices;
		devices.push_back(_Device);
		*error = CL_SUCCESS;

		if (program.build(devices) != CL_SUCCESS)
		{
			logMessage("CL: Error building: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(_Device), MESSAGE_ERROR);
			*error = EXIT_FAILURE;
			return program;
		}

		std::cout << " CL Compiled. Prepared to run " << "\n";
		return program;

	}
	
	cl::Program loadCLSources(std::string cl_prog, cl::Context _Context, cl::Device _Device,  int* error)
	{
		// kernel
		std::cout << " CL: Cargando archivo : " << cl_prog << "\n";
		sources.clear();
		if (std::ifstream(cl_prog.c_str()))
		{

			cl_utils::cargarFuente(cl_prog);
			std::pair<const char*, ::size_t> x(cl_utils::kernel_source.c_str(), cl_utils::kernel_source.length());
			sources.push_back(x);
		}
		else
		{
			logMessage("CL: Archivo no encontrado : " + cl_prog, MESSAGE_ERROR);
			exit(1);
		}
		return buildProgram(_Context, _Device, error);

	}
	cl::Program loadCLSources(std::string cl_prog, int localWG, int platformProcessingIndex, int deviceProcessingIndex, int* error)
	{
		return loadCLSources(cl_prog,  _defaultContext, _defaultDevice, error);
	}


	int initDevice(int platformProcessingIndex, int deviceProcessingIndex)
	{
		//get all platforms (drivers)
		cl::Platform::get(&all_platforms);
		if (all_platforms.size() == 0)
		{
			logMessage(" No platforms found. Check OpenCL installation!\n", MESSAGE_ERROR);
			return EXIT_FAILURE;
		}

		std::cout << " Platform list <<<<";
		for (int i = 0; i < all_platforms.size(); i++)
		{
			std::cout << cl::Platform(all_platforms[i]).getInfo<CL_PLATFORM_NAME>(); " || ";
		}
		std::cout << " >>>>>>" << "\n";

		if (all_platforms.size() >= platformProcessingIndex)
			_defaultPlatform = cl::Platform(all_platforms[platformProcessingIndex]);
		else {
			logMessage("Check platformProcessingIndex. The number is out of range\n", MESSAGE_ERROR);
			_defaultPlatform = cl::Platform(all_platforms[0]);
		}

		std::cout << "Using platform: " << _defaultPlatform.getInfo<CL_PLATFORM_NAME>() << "\n";
		//get default device of the default platform

		_defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
		if (all_devices.size() == 0)
		{
			logMessage(" No devices found. Check OpenCL installation!\n", MESSAGE_ERROR);
			return EXIT_FAILURE;
		}
		std::cout << " Devices list <<<<";
		for (int i = 0; i < all_devices.size(); i++)
		{
			std::cout << cl::Device(all_devices[i]).getInfo<CL_DEVICE_NAME>(); " |||| ";
		}
		std::cout << " >>>>>>" << "\n";

		if (all_platforms.size() >= platformProcessingIndex)
			_defaultDevice = cl::Device(all_devices[deviceProcessingIndex]);
		else {
			logMessage("Check deviceProcessingIndex. The number is out of range\n", MESSAGE_ERROR);
			_defaultDevice = cl::Device(all_devices[0]);
		}

		std::cout << "Using device: " << _defaultDevice.getInfo<CL_DEVICE_NAME>() << "\n";


		_defaultContext = cl::Context(_defaultDevice);

		

		return EXIT_SUCCESS;
	}

}

