// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.


#include <experimental/filesystem>
#include <omp.h>

#include "bitstream.hpp"

#include "quality_metrics_OpenCV.h"
#include "cameraCapturer.h"

#ifdef ZSTD
#include "jp2CPUEncoder.h"
#endif

#include "gpu/clEncoder.h"
#include "gpu/cl_utils.h"
#include "gpu/clHeader.h"

#include "testTools.h"

#include <chrono>
using namespace std::chrono;

#define CODE_VERSION "3Dic2021"

splineCompression* _splLinear, *_splBiQubic, *_splLossLess, *_spl5;

/**
Class to encapsulate a filter alongside its options
*/
class filter_options
{
public:
	std::string filter_name;                                   //Friendly name of the filter
#ifdef REAL_SENSE

	rs2::filter& filter;                                       //The filter in use
	                       //A boolean controlled by the user that determines whether to apply the filter or not
#endif

};



///////////////////////////////////////////////////////
// Process Camera in real time
int processCamera(std::string model)
{
	try
	{
		
		starCapturing(1024,768, model);

		using namespace cv;
		const auto window_name = "Display Image";
		namedWindow(window_name, WINDOW_AUTOSIZE);
		int frameIndex = 0;

		cv::Mat accum;
		
		_splLossLess = new splineCompression(LOSSLESS_COMPRESSION);
		_splLinear = new splineCompression(LINEAR_COMPRESSION);
		_splBiQubic = new splineCompression(SPLINE_COMPRESSION);
		_spl5 = new splineCompression(SPLINE5_COMPRESSION);

		splineCompression* spl = _splLossLess;

		std::string outDir = "temp/";

		// create dir
		if (!dirExists(outDir)) createDirectory(outDir);
						
		double encodeRate = 0;

		int scanrow = 200;

		cv::Mat prevDepth;

		while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
		{
			
			cv::Mat depth = getLastDepthFrame();
			cv::Mat image = getLastRGBFrame();

			if (depth.cols == 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}
		
			startProcess("encoding");
			spl->createFromImage(depth);

			cv::Mat depthRestored = spl->restoreAsImage();
			double time = endProcess("encoding");

			if (frameIndex % 10 == 0)
			{
				// create Files`
				std::string fn = outDir + std::to_string(frameIndex) + ".bin";
				encodeRate = (double)(spl->encode(depth, fn))/(depth.cols * depth.rows * 2);

				fn = outDir + std::to_string(frameIndex) + ".png";
				cv::imwrite(fn, depth);
			}
			
			spl->display(scanrow, spl->encodedMode,1);

			// Update the window with new data
			if (image.cols > 0) cv::imshow(window_name, image);



			cv::Mat idst;
			cv::absdiff(depth, depthRestored, idst);
			// map color ORIG
			Mat im_color, im_gray;
			depth.convertTo(im_gray, CV_8UC1, 1.0/160.0);
			cv::applyColorMap(im_gray, im_color, COLORMAP_HSV);

			// map color RESTORED
			Mat im_colorR, im_grayR;
			depthRestored.convertTo(im_grayR, CV_8UC1, 1.0 / 160.0);
			cv::applyColorMap(im_grayR, im_colorR, COLORMAP_HSV);

			// map color DIFF
			Mat im_colorR2, im_grayR2;
			idst.convertTo(im_grayR2, CV_8UC1, 1.0 );
			cv::applyColorMap(im_grayR2, im_colorR2, COLORMAP_HSV);

			double pp = qm::psnr(depth, depthRestored, 1);
			double pNoise = 0;

			if (prevDepth.cols > 0)
			{
				pNoise = qm::psnr(depth, prevDepth, 1);
			}

			cv::putText(im_colorR, "Alg:" + spl->compressionModes[spl->encodedMode], cv::Point(20, 20), 1, 1, cv::Scalar(255, 255, 255));
			cv::putText(im_colorR, "Time " + std::to_string(time), cv::Point(20, 50), 1, 1, cv::Scalar(255, 255, 255));
			cv::putText(im_colorR, "Rate " + std::to_string(encodeRate), cv::Point(20, 80), 1, 1, cv::Scalar(255, 255, 255));
			cv::putText(im_colorR, "PSNR " + std::to_string(pp), cv::Point(20, 110), 1, 1, cv::Scalar(255, 255, 255));
			cv::putText(im_colorR, "Noise " + std::to_string(pNoise), cv::Point(20, 130), 1, 1, cv::Scalar(255, 255, 255));


			cv::imshow("Restore SPL", im_colorR);
			cv::imshow("depth", im_color);
			cv::imshow("diff", im_colorR2);

			
			if (accum.cols == 0)	accum = depth.clone();
			else cv::addWeighted(accum, 0.60, depth, 0.4, 0, accum);


			cv::putText(accum, "row " + std::to_string(scanrow), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0));
						
			cv::line(accum, cv::Point(0, scanrow), cv::Point(accum.cols - 1, scanrow), cv::Scalar(255, 0, 0), 3);
			if (accum.cols > 0 ) cv::imshow("smooth", accum);


			int key = cv::waitKey(1);

			if (key == '1')  spl = _splLossLess;
			if (key == '2') spl = _splLinear;
			if (key == '3')  spl = _splBiQubic;
			if (key == '4')  spl = _spl5;
			if (key == 'w')  spl->lonelyPixelsRemoval = 1;
			if (key == 'e')  spl->lonelyPixelsRemoval = 2;
			if (key == 'r')  spl->lonelyPixelsRemoval = 3;

			if (key == 27)  break;

			prevDepth = depth.clone();
			frameIndex++;
		}


		stopCapturing();

		return EXIT_SUCCESS;
	}
	
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

}

////////////////////////////////////
// Record Scenario
int recordScenario(std::string outdir,std::string cameraModel, int maxFrames)
{
	try
	{
		
		using namespace cv;
		const auto window_name = "Display Image";
		namedWindow(window_name, WINDOW_AUTOSIZE);
		int frameIndex = 0;
		
		int w = 1024;
		int h = 768;


		int waittime = 5000;

		starCapturing(w, h, cameraModel);

		while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
		{
			cv::Mat depth = getLastDepthFrame();

			if (depth.cols == 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}

			if (waittime > 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				waittime -= 100;
				continue;
			}

			std::string date = return_current_time_and_date();
			
			// create dir
			if (!dirExists(outdir )) createDirectory(outdir);
			
			// create Files`
			std::string fn = outdir + std::to_string(frameIndex) + ".pgm";
			cv::imwrite(fn, depth);

			// Store Scripts
			createScripts(outdir, frameIndex);

			Mat im_color, im_gray;
			depth.convertTo(im_gray, CV_8UC1, 1.0 / 160.0);
			cv::applyColorMap(im_gray, im_color, COLORMAP_HSV);
			
			cv::imshow("depth", im_color);
			

			int key = cv::waitKey(1);


			if (key == 27)  break;

			frameIndex++; 
			
			if (frameIndex == maxFrames) break;
		}

		stopCapturing();

		return EXIT_SUCCESS;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}

////////////////////////////////////////
// Process Directory
int processDir(std::string outDir, std::string iformat, bool doPreProcess)
{
	_splLossLess = new splineCompression(LOSSLESS_COMPRESSION);
	_splLinear = new splineCompression(LINEAR_COMPRESSION);
	_splBiQubic = new splineCompression(SPLINE_COMPRESSION);
	_spl5 = new splineCompression(SPLINE5_COMPRESSION);


	std::vector<std::string> files = listFilesInDir(outDir, ".png");

	if (files.size() == 0)  files = listFilesInDir(outDir, ".pgm");

	int i = 0;
	for (auto s : files)
	{
		cv::Mat depth = cv::imread(s, -1);

		if (doPreProcess)
		{
			

			cv::Mat temp;


			cv::GaussianBlur(depth, temp, cv::Size(0, 0), 3);
			cv::addWeighted(depth, 1.25, temp, -0.25, 0, depth);

			depth = depth * 0.25;
		}

		compressionMetrics(depth, outDir, i, false,true,1);

		if (i % 10 == 0) showProcessTime();
		i++;

		if (i > 150)break;
	}

	showProcessTime();

	cv::waitKey(-1);

	return 1;
}



void encodeGPU(std::string inputF, std::string ouputF, std::string format)
{


	std::vector<std::string> files = listFilesInDir(inputF, ".png");

	splineCompression* spl = NULL;
	spl = new splineCompression(SPLINE_COMPRESSION);
	spl->zstd_compression_level = 0;

	int fileIndex = 0;

	for (auto f : files)
	{

		cv::Mat m = cv::imread(f, -1);

		if (m.cols == 0)
		{
			std::cout << "Input image could not be loaded";
			return;
		}

		FitCL::encodeCL(m, m.rows, fileIndex % 30 == 0);

		
		std::vector<spline_ptr> cl_splines = FitCL::getSplines();

		
		spl->encode(m, "");



		bool hasDifference = false;
		// compare SPLINES
		for (int i = 3; i < cl_splines.size(); i++)
		{
			cl_spline_data* csp = (cl_spline_data*)cl_splines[i];

			if (!csp) continue;
			if (csp->values_count != spl->splines[i]->values.size())
			{
				std::cout << "Difference count!!" << csp->x0 << "\n";
				hasDifference = true;
				break;
			}
			else
			if (csp->x0 != spl->splines[i]->x0 || csp->y0 != spl->splines[i]->y0)
			{
				std::cout << "Difference Positions!!" << csp->x0 << "\n";
				hasDifference = true;
				break;
			}
			else
			if (abs(csp->coefs[0] - spl->splines[i]->coefs[0]) > 0.001 )
			{
				std::cout << "Difference coef!!" << csp->x0 << ":" << csp->y0 << "\n";
				hasDifference = true;
				//break;
			}

		}

		if (!hasDifference && fileIndex % 30 == 0)
		{
			std::cout << "EXACT " << "\n";
		}
		// compare output buffer
		unsigned short* bufferCPU = spl->outBuffer;
		int outbufferSize = 0;
		unsigned short* bufferGPU = FitCL::cl_getOutputBuffer(outbufferSize);

		std::cout << "difference buffer ";
		for (int i = 1; i < outbufferSize; i++)
		{
			if (bufferCPU[i] != bufferGPU[i])
			{
				std::cout << ".. "<< i << "(" << bufferCPU[i] - bufferGPU[i] << ")..";
				break;
			}
		}

		if (fileIndex % 30 == 0)
		{
			showProcessTime();
		}
		cv::imshow("image", m);
		cv::waitKey(1);
		fileIndex++;
	}

	return;
}
/////////////////////////////////////////////////
// Encode image
void encode(std::string inputF, std::string ouputF, std::string format)
{


	cv::Mat m = cv::imread(inputF, -1);

	if (m.cols == 0)
	{
		std::cout << "Input image could not be loaded";
		return;
	}

	splineCompression* spl = NULL;
	
	if (format == "LOSSLESS")
	{
		spl = new splineCompression(LOSSLESS_COMPRESSION);
	}
	else
		if (format == "LINEAR")
		{
			spl = new splineCompression(LINEAR_COMPRESSION);
		}
		else
			if (format == "CUBIC")
			{
				spl = new splineCompression(SPLINE_COMPRESSION);
			}
			else
			if (format == "FIVE")
			{
				spl = new splineCompression(SPLINE5_COMPRESSION);
			}
			else
			{
				std::cout << "Unknown compression parameter";
				return;
			}

	startProcess("encode");
	spl->encode(m,ouputF);

	double time = endProcess("encode");

	std::cout << "File generated ok. Encode time " << time << "ms \n";

	return;
}

void decode(std::string inputF, std::string ouputF, std::string format)
{
	splineCompression* spl;
	spl = new splineCompression(LOSSLESS_COMPRESSION);
	
	startProcess("decode");
	cv::Mat m = spl->decode(inputF);

	double time = endProcess("decode");

	cv::imwrite(ouputF, m);

	std::cout << "File generated ok. Decode time " << time << "ms \n";

	return;
}

//////////////////////////////////////////////////////////////////////////
///
void multiCameraRecording()
{
	try
	{
		cameraCapturer* camCap = new cameraCapturer(1, 1, 1, "MP4", "");
		camCap->outputDir = "d:\\temp\\Paper\\";
		camCap->setRecordingState(true);
		camCap->startCapturing();
	}
	catch (std::exception ex)
	{
		std::cout << ex.what() << "\n";
	}

}

void showDepthInColors()
{
	
	std::vector<std::string> filesDEPTH = getAllFilesInDir("E:\\Resources\\DataSetRGBD\\home_office_0007\\depth\\", {".pgm"});
	std::vector<std::string> filesRGB = getAllFilesInDir("E:\\Resources\\DataSetRGBD\\home_office_0007\\rgb\\", { ".ppm" });

	splineCompression* splLinear = new splineCompression(LINEAR_COMPRESSION);

	for (int i = 0;i < filesDEPTH.size();i++)
	{
		cv::Mat m = cv::imread(filesDEPTH[i], -1);

		cv::Mat mColor = cv::imread(filesRGB[i], -1);
		cv::Mat m2;
		if (m.cols == 0) continue;

		cv::resize(mColor, mColor, cv::Size(), 2.0, 2.0);
		cv::resize(m, m, cv::Size(), 2.0, 2.0);

		m.convertTo(m2, CV_16UC1, 1.0 / 6.0);

		splLinear->encode(m2, "e://out.bin");

		splLinear->display(320, splLinear->encodedMode, 1);

		// Apply the colormap:
		// Holds the colormap version of the image:
		cv::Mat img_color, combined;

		m.convertTo(img_color, CV_8UC1, 1.0 / 256.0);
		cv::applyColorMap(img_color, img_color, cv::COLORMAP_JET);

		

		cv::imshow("depth_color", img_color);

		cv::imshow("depth", m);

		cv::imshow("rgb", mColor);

		cv::waitKey(-1);
	}

	cv::waitKey(-1);
}

int main(int argc, char * argv[])
{
	


	InputParser input(argc, argv);

	if (input.cmdOptionExists("-h") || (argc == 1))
	{

		// Do stuff
		std::cout << "Version  : " << CODE_VERSION << "\n";
		std::cout << "Commands" << "\n";
		std::cout << "-encode  -out {outputFile} -in {inputFile} -encoder {LOSSLESS/LINEAR/CUBIR}. Encode file" << "\n";
		std::cout << "-decode  -out {outputFile} -in {inputFile} -decode {LOSSLESS/LINEAR/CUBIR} " << "\n";
		std::cout << "-camera  -out {outputDir} -cameraModel {INTEL/OCCIPITAL} . Process detected camera" << "\n";
		std::cout << "-J2K  -out {outputDir} . Compare to J2K encoder results" << "\n";
		std::cout << "-metrics -out {outputDir}. Compute metrics from recorded images" << "\n";
		std::cout << "-record  -out {outputDir} -cameraModel {INTEL/OCCIPITAL}.  Record frames in PGM format " << "\n";

		return 0;
	}

	std::string outDir = input.getCmdOption("-out");
	std::string in = input.getCmdOption("-in");
	std::string encoder = input.getCmdOption("-encoder");
	std::string decoder = input.getCmdOption("-decoder");
	std::string model = input.getCmdOption("-camera");

	bool doPreProcess = input.cmdOptionExists("-preprocess");

	std::string procCount = input.getCmdOption("-procCount");

	clUtils::initDevice(0, 0);
	FitCL::initEncoderCL("D:\\Proyects\\DepthCompression\\depthCompression\\gpu\\", 256, 0, 0);

	if (input.cmdOptionExists("-demo"))
	{
		showDepthInColors();
	}
	else
	if (input.cmdOptionExists("-encodeGPU"))
	{
		encodeGPU(in, outDir, encoder);
	}
	else
	if (input.cmdOptionExists("-encode"))
	{
		encode(in, outDir, encoder);
	}
	else
	if (input.cmdOptionExists("-decode"))
		{
			decode(in, outDir, decoder);
		}
	else
	if (input.cmdOptionExists("-live"))
	{
		processCamera(model);
	}
	else
		if (input.cmdOptionExists("-J2K"))
		{
		
			compareToJ2K(outDir);
		}
		else
	if (input.cmdOptionExists("-record"))
	{
		
		recordScenario(outDir,model, 500);
	}
	else
	if (input.cmdOptionExists("-metrics"))
	{
		processDir(outDir, ".pgm", doPreProcess);
	}
	else
	if (input.cmdOptionExists("-test0"))
	{
		test0(outDir,1, doPreProcess);
	}
	else
		if (input.cmdOptionExists("-testMultiCam"))
		{
			multiCameraRecording();
		}
	else
		if (input.cmdOptionExists("-test1"))
		{

			if (procCount != "")
			{
				std::cout << " Set threads count " << procCount << "\n";
				omp_set_num_threads(std::stoi(procCount));

			}

			std::cout << "Test 1 .. dir " << in << "\n";
			std::cout << " Num threads " << omp_get_num_threads() << "\n";

			test1(in, ".png");
		}
	else std::cout << " Unknown command" << "\n";
}



