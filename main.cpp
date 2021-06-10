// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.


#include <experimental/filesystem>
#include "bitstream.hpp"

#include "quality_metrics_OpenCV.h"
#include "cameraCapturer.h"

#include <chrono>
using namespace std::chrono;

#define CODE_VERSION "7Jun2021"

splineCompression* splLinear, *splBiQubic, *splLossLess;

/**
Class to encapsulate a filter alongside its options
*/
class filter_options
{
public:
	std::string filter_name;                                   //Friendly name of the filter
	rs2::filter& filter;                                       //The filter in use
	                       //A boolean controlled by the user that determines whether to apply the filter or not
};

class InputParser {
public:
	InputParser(int &argc, char **argv) {
		for (int i = 1; i < argc; ++i)
			this->tokens.push_back(std::string(argv[i]));
	}
	/// @author iain
	const std::string& getCmdOption(const std::string &option) const {
		std::vector<std::string>::const_iterator itr;
		itr = std::find(this->tokens.begin(), this->tokens.end(), option);
		if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
			return *itr;
		}
		static const std::string empty_string("");
		return empty_string;
	}
	/// @author iain
	bool cmdOptionExists(const std::string &option) const {
		return std::find(this->tokens.begin(), this->tokens.end(), option)
			!= this->tokens.end();
	}
private:
	std::vector <std::string> tokens;
};


void run_screaming(const char* message, const int code) {
	printf("%s \n", message);
	exit(code);
}


std::string return_current_time_and_date() {
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);

	std::ostringstream oss;
	oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
	auto str = oss.str();
	return str;
}


bool dirExists(const std::string& dirPath)
{

	return std::experimental::filesystem::exists(dirPath) && std::experimental::filesystem::is_directory(dirPath);

}


bool createDirectory(const std::string& name)
{
	return std::experimental::filesystem::create_directory(name);
}

void createScripts(std::string outDir, int frame)
{

	std::string fn = outDir + "scripts_lossless_encodeJ2K.bat";
	std::string sc0 = "fast_j2k.exe -i " + std::to_string(frame) + ".pgm -o " + std::to_string(frame) + ".lossless.jp2 -a REV -l 7 -c 32";
	std::ofstream myfile(fn, std::ios::out | std::ios::app);
	myfile << sc0 << "\n";
	myfile.close();

	// Lossy encode 85 Q
	fn = outDir + "scripts_lossy_encodeJ2K_85.bat";
	std::string sc1 = "fast_j2k.exe -i " + std::to_string(frame) + ".pgm -o " + std::to_string(frame) + ".q85.jp2 -a IRREV -l 7 -c 32 -q 85";
	std::ofstream myfile2(fn, std::ios::out | std::ios::app);
	myfile2 << sc1 << "\n";
	myfile2.close();


	// Lossy encode 95 Q
	fn = outDir + "scripts_lossy_encodeJ2K_95.bat";
	std::string sc2 = "fast_j2k.exe -i " + std::to_string(frame) + ".pgm -o " + std::to_string(frame) + ".q95.jp2 -a IRREV -l 7 -c 32 -q 95";
	std::ofstream myfile3(fn, std::ios::out | std::ios::app);
	myfile3 << sc2 << "\n";
	myfile3.close();

	// Lossy decode 85 Q
	fn = outDir + "scripts_lossy_decodeJ2K_85.bat";
	std::string dec0 = "fast_j2k.exe -i " + std::to_string(frame) + ".q85.jp2 -o " + std::to_string(frame) + ".q85.pgm";
	std::ofstream myfile4(fn, std::ios::out | std::ios::app);
	myfile4 << dec0 << "\n";
	myfile4.close();

	// Lossy decode 95 Q
	fn = outDir + "scripts_lossy_decodeJ2K_95.bat";
	std::string dec1 = "fast_j2k.exe -i " + std::to_string(frame) + ".q95.jp2 -o " + std::to_string(frame) + ".q95.pgm";
	std::ofstream myfile5(fn, std::ios::out | std::ios::app);
	myfile5 << dec1 << "\n";
	myfile5.close();


	


	
	myfile << sc1 << "\n";
	myfile << sc2 << "\n";

	myfile << dec0 << "\n";
	myfile << dec1 << "\n";

	

}

void compressionMetrics(cv::Mat& depth, std::string outDir, int frame, bool verbose)
{
	uint32_t width = 1024;
	uint32_t height = 768;
	size_t orig_size = 1024 * 768 * 2;

	std::vector<float> valuesToExport;

	if (!dirExists(outDir + "ZSTD//")) createDirectory(outDir + "ZSTD//");
	if (!dirExists(outDir + "Linear//")) createDirectory(outDir + "Linear//");
	if (!dirExists(outDir + "Cubic//")) createDirectory(outDir + "Cubic//");
	if (!dirExists(outDir + "PNG//")) createDirectory(outDir + "PNG//");
		

	// ZSTD
	auto start = high_resolution_clock::now();
	double compZ = (double)splLossLess->encode(depth, outDir + "ZSTD//" + std::to_string(frame) + ".bin") / (orig_size);
	
	auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedZ = duration.count();

	// LINEAR
	start = high_resolution_clock::now();
	double compL = (double)splLinear->encode(depth,outDir + "Linear//" + std::to_string(frame) + ".bin") / (orig_size);
	
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedL = duration.count();


	// BIQUBIC
	start = high_resolution_clock::now();
	double compQ = (double)splBiQubic->encode(depth, outDir + "Cubic//" + std::to_string(frame) + ".bin") / (orig_size);

	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedQ = duration.count();
	//bitS->readFromFile("d://temp//l515//ZSTD.bin");


	//PNG
	start = high_resolution_clock::now();
	cv::imwrite(outDir + "PNG//" + std::to_string(frame) + ".png", depth);
	double compPNG = (double)get_file_size(outDir +"PNG//" + std::to_string(frame) + ".png") / (orig_size);
	
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedPNG = duration.count();
	
	
	start = high_resolution_clock::now();
	cv::Mat depthRestored = splLossLess->restoreAsImage();
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double decodeZ = duration.count();

	start = high_resolution_clock::now();
	cv::Mat depthRestoredLinear = splLinear->decode(outDir + "Linear//" + std::to_string(frame) + ".bin");
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double decodeLinear = duration.count();

	start = high_resolution_clock::now();
	cv::Mat depthRestoredCubic = splBiQubic->decode(outDir + "Cubic//" + std::to_string(frame) + ".bin");
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double decodeCubic = duration.count();


	start = high_resolution_clock::now();
	cv::Mat depth2 = cv::imread(outDir + "PNG//" + std::to_string(frame) + ".png", -1);
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double decodePNG = duration.count();


	std::string fn = outDir + "RAW//" + std::to_string(frame) + ".raw";
	std::ofstream out(fn, std::ios::out | std::ios::binary);
	// bin mask
	out.write((const char*)depth.data, orig_size);
	out.close();

	
	////////////////////////////////////////////////////////////////////////
	fn = outDir + "RAW//" + std::to_string(frame) + ".pgm";
	cv::imwrite(fn, depth);
	
	if (verbose)
	{
		std::cout << " LINEAR " << elapsedL << " ms" << "\n";
		std::cout << " CUBIC " << elapsedQ << " ms" << "\n";
		std::cout << " LOSSLESS " << elapsedZ << " ms" << "\n";
		std::cout << " PNG " << elapsedPNG << " ms" << "\n";
	}

	// Compute Error
	double error = qm::psnr(depth, depthRestored, 1);
	double errorL = qm::psnr(depth, depthRestoredLinear, 1);
	double errorB = qm::psnr(depth, depthRestoredCubic, 1);

	
	valuesToExport.push_back(compZ);
	valuesToExport.push_back(error);
	valuesToExport.push_back(elapsedZ);
	valuesToExport.push_back(decodeZ);

	valuesToExport.push_back(compL);
	valuesToExport.push_back(errorL);
	valuesToExport.push_back(elapsedL);
	valuesToExport.push_back(decodeLinear);

	valuesToExport.push_back(compQ);
	valuesToExport.push_back(errorB);
	valuesToExport.push_back(elapsedQ);
	valuesToExport.push_back(decodeCubic);

	valuesToExport.push_back(compPNG);
	valuesToExport.push_back(0);
	valuesToExport.push_back(elapsedPNG);
	valuesToExport.push_back(decodePNG);

	exportCSV(outDir + "data.csv", valuesToExport, frame);
}



void histDisplay(std::vector<double>& histogram, const char* name)
{
	//std::cout<<"Start HistDisplay" << std::endl;

	if (histogram.size() < 10) return;
	std::vector<double> hist;
	cv::Mat histImg;
	histImg.create(cv::Size(600, 150), CV_8UC3);

	hist.clear();

	for (int i = 0; i < histogram.size(); i = i + 1)
	{
		hist.push_back(histogram[i]);
	}
	// draw the histograms
	int hist_w = histogram.size(); int hist_h = histImg.rows;
	int bin_w = 1;

	histImg.setTo(cv::Scalar(255, 255, 255));
	// find the maximum intensity element from histogram
	double max = hist[0];
	double min = 10000.0;
	for (int i = 1; i < histogram.size(); i++)
	{
		if (max < hist[i])
		{
			max = hist[i];
		}

		if (min > hist[i])
		{
			min = hist[i];
		}
	}

	double minReal = min;

	min = 0.0;
	// normalize the histogram between 0 and histImage.rows
	for (int i = 0; i < histogram.size(); i = i + 1)
	{
		hist[i] = ((double)(hist[i] - min) / (max - min))*histImg.rows;
	}


	// draw the intensity line for histogram
	for (int i = 0; i < histogram.size(); i = i + 1)
	{
		int x = (i * histImg.cols) / histogram.size();

		line(histImg, cv::Point(x, hist_h), cv::Point(x, hist_h - hist[i]), cv::Scalar(0, 0, 0), 1, 8, 0);
	}

	double pos0 = (0 - min) / (max - min)*histImg.rows;
	line(histImg, cv::Point(0, pos0), cv::Point(histImg.cols - 1, pos0), cv::Scalar(0, 0, 255), 1, 8, 0);

	cv::putText(histImg, "max:" + std::to_string(max), cv::Point(20, 20), 1, 0.8, cv::Scalar(255, 0, 0));
	cv::putText(histImg, "min:" + std::to_string(minReal), cv::Point(20, 40), 1, 0.8, cv::Scalar(255, 0, 0));


	// display histogram
//	namedWindow(name, CV_WINDOW_AUTOSIZE);

	cv::imshow(name, histImg);
	//std::cout << "End HistDisplay" << std::endl;
}

///////////////////////////////////////////////////////
// Process Camera in real time
int processCamera(int cam)
{
	try
	{
		
		starCapturing(1024,768, "INTEL_REALSENSE");

		using namespace cv;
		const auto window_name = "Display Image";
		namedWindow(window_name, WINDOW_AUTOSIZE);
		int frameIndex = 0;

		cv::Mat accum;

		int scanrow = 500;

		splLossLess = new splineCompression(LOSSLESS_COMPRESSION);
		splLinear = new splineCompression(LINEAR_COMPRESSION);
		splBiQubic = new splineCompression(SPLINE_COMPRESSION);

		int w = 1024;
		int h = 768;

		
		while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
		{
			
			cv::Mat depth = getLastDepthFrame();
			cv::Mat image = getLastRGBFrame();
		

			if (frameIndex < 100)
			{
				compressionMetrics(depth, "d://temp//l515//", frameIndex, false);
			}
			cv::Mat depthRestored = splLossLess->restoreAsImage();
			cv::Mat depthRestoredLinear = splLinear->restoreAsImage();
			cv::Mat depthRestoredCubic = splBiQubic->restoreAsImage();


			splLinear->display(scanrow);

			splBiQubic->display(scanrow);

			// Update the window with new data
			cv::imshow(window_name, image);

			Mat im_color, im_gray;
			depth.convertTo(im_gray, CV_8UC1, 1.0/160.0);
			cv::applyColorMap(im_gray, im_color, COLORMAP_HSV);

			Mat im_colorR2, im_grayR2;
			depthRestoredCubic.convertTo(im_grayR2, CV_8UC1, 1.0 / 160.0);
			cv::applyColorMap(im_grayR2, im_colorR2, COLORMAP_HSV);

			cv::Mat idst;
			cv::absdiff(depth, depthRestoredCubic, idst);


			Mat im_colorR, im_grayR;
			idst.convertTo(im_grayR, CV_8UC1, 2.0);
			cv::applyColorMap(im_grayR, im_colorR, COLORMAP_HSV);


			cv::imshow("Restore SPL", im_colorR2);
			cv::imshow("depth", im_color);
			cv::imshow("diff", im_colorR);

			
			if (accum.cols == 0)	accum = image.clone();
			else cv::addWeighted(accum, 0.60, image, 0.4, 0, accum);


			cv::putText(accum, "row " + std::to_string(scanrow), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0));
						
			cv::line(accum, cv::Point(0, scanrow), cv::Point(accum.cols - 1, scanrow), cv::Scalar(255, 0, 0), 3);
			cv::imshow("smooth", accum);


			int key = waitKey(1);

			if (key == '1') scanrow = (scanrow + 1) % 768;
			if (key == '2') scanrow = (scanrow + 10) % 768;
			if (key == '3') scanrow = (scanrow + 100) % 768;

			if (key == 27)  break;

			frameIndex++;
		}


		stopCapturing();

		return EXIT_SUCCESS;
	}
	catch (const rs2::error & e)
	{
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

}

////////////////////////////////////
// Record Scenario
int recordScenario(std::string outdir, int maxFrames)
{
	try
	{
		

		using namespace cv;
		const auto window_name = "Display Image";
		namedWindow(window_name, WINDOW_AUTOSIZE);
		int frameIndex = 0;

		cv::Mat accum;

		int scanrow = 500;
		int w = 1024;
		int h = 768;

		starCapturing(1024, 768, "INTEL_REALSENSE");

		while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
		{
			cv::Mat depth = getLastDepthFrame();
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
			
			cv::putText(accum, "row " + std::to_string(scanrow), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0));


			cv::line(accum, cv::Point(0, scanrow), cv::Point(accum.cols - 1, scanrow), cv::Scalar(255, 0, 0), 3);
			cv::imshow("smooth", accum);


			int key = cv::waitKey(1);


			if (key == 27)  break;

			frameIndex++;
			if (frameIndex == maxFrames) break;
		}

		stopCapturing();

		return EXIT_SUCCESS;
	}
	catch (const rs2::error & e)
	{
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}

////////////////////////////////////////
// Process Directory
int processDir(std::string outDir)
{
	splLossLess = new splineCompression(LOSSLESS_COMPRESSION);
	splLinear = new splineCompression(LINEAR_COMPRESSION);
	splBiQubic = new splineCompression(SPLINE_COMPRESSION);


	for (int i = 0; i < 150; i++)
	{
		cv::Mat depth = cv::imread(outDir + std::to_string(i) + ".pgm", -1);
		compressionMetrics(depth, outDir, i, false);
	}

	showProcessTime();

	cv::waitKey(-1);

	return 1;
}


///////////////////////////////////
// Test 0 : simple file processing
int test0()
{
	std::cout << "Starting compression TEST" << "\n";
	cv::Mat depth = cv::imread("d://temp//l515//test_rgb0.png", -1);

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,cv::Size(5,5),cv::Point(2, 2));
	cv::dilate(depth, depth, element);

	cv::erode(depth, depth, element);
	
	
	splLinear = new splineCompression(LINEAR_COMPRESSION);
	splBiQubic = new splineCompression(SPLINE_COMPRESSION);
	splLossLess = new splineCompression(LOSSLESS_COMPRESSION);

	// Apply threshold
	int iter = 1;
	int scanrow = 500;
	size_t orig_size = 1024 * 768 * 2;
	while (true)
	{
		compressionMetrics(depth, "d://temp//l515//", 0, true);
		
		cv::Mat depthRestored = splLossLess->restoreAsImage();
		cv::Mat depthRestoredLinear = splLinear->restoreAsImage();
		cv::Mat depthRestoredCubic = splBiQubic->restoreAsImage();


		splLinear->display(scanrow);

		splBiQubic->display(scanrow);

		double pp = qm::psnr(depth, depthRestored, 1);
		double ppL = qm::psnr(depth, depthRestoredLinear, 1);
		double pp85 = qm::psnr(depth, depthRestoredCubic, 1);


		cv::Mat im_color, im_gray;
		depthRestored.convertTo(im_gray, CV_8UC1, 1.0 / 160.0);
		cv::applyColorMap(im_gray, im_color, cv::COLORMAP_HSV);

		cv::putText(im_color, std::to_string(pp), cv::Point(20, 20), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_color, std::to_string(ppL), cv::Point(20, 50), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_color, std::to_string(pp85), cv::Point(20, 80), 1, 1, cv::Scalar(255, 255, 255));
		

		cv::line(im_color, cv::Point(0, scanrow), cv::Point(depthRestored.cols - 1, scanrow), cv::Scalar(255, 255, 255), 3);
		cv::imshow("M1", im_color);
		cv::imshow("M2", depthRestoredLinear);
		cv::imshow("M3", depthRestoredCubic);

		int key = cv::waitKey(-1);

		if (key == '1') scanrow = (scanrow + 1) % 768;
		if (key == '2') scanrow = (scanrow + 10) % 768;
		if (key == '3') scanrow = (scanrow + 100) % 768;
		
	}
	return 1;
}



/////////////////////////////////////////////////
// Extract metrics comparing to J2K compression
void compareToJ2K(std::string outDir)
{
	splBiQubic = new splineCompression(SPLINE_COMPRESSION);
	splLinear = new splineCompression(LINEAR_COMPRESSION);

	std::vector<double> r0;
	std::vector<double> r1;

	for (int i = 0; i < 150; i++)
	{
		std::string fRaw = outDir +   std::to_string(i)+".pgm"  ;
		std::string fJ2K85 = outDir  + std::to_string(i) + ".q85.pgm";
		std::string fJ2K95 = outDir  + std::to_string(i) + ".q95.pgm";
		std::string fCubic = outDir + "Cubic\\" + std::to_string(i) + ".bin";
		std::string fLinear = outDir + "Linear\\" + std::to_string(i) + ".bin";

		cv::Mat mRaw = cv::imread(fRaw,-1);
		cv::Mat mJ2K85 = cv::imread(fJ2K85, -1);
		cv::Mat mJ2K95 = cv::imread(fJ2K95, -1);
		cv::Mat mCubic = splBiQubic->decode(fCubic);

		cv::Mat mLinear = splLinear->decode(fLinear);

		// Update the window with new data
		

		cv::Mat im_color, im_gray;
		mRaw.convertTo(im_gray, CV_8UC1, 1.0 / 160.0);
		cv::applyColorMap(im_gray, im_color, cv::COLORMAP_HSV);

		cv::Mat im_colorR2, im_grayR2;
		mCubic.convertTo(im_grayR2, CV_8UC1, 1.0 / 160.0);
		cv::applyColorMap(im_grayR2, im_colorR2, cv::COLORMAP_HSV);

		cv::Mat im_colorR3, im_grayR3;
		mJ2K95.convertTo(im_grayR3, CV_8UC1, 1.0 / 160.0);
		cv::applyColorMap(im_grayR3, im_colorR3, cv::COLORMAP_HSV);

		double pp = qm::psnr(mRaw, mCubic,1);
		double ppL = qm::psnr(mRaw, mLinear, 1);
		double pp85 = qm::psnr(mRaw, mJ2K85, 1);
		double pp95 = qm::psnr(mRaw, mJ2K95, 1);

		cv::putText(im_colorR2, std::to_string(pp), cv::Point(20, 20), 1, 1, cv::Scalar(255,255,255));
		cv::putText(im_colorR2, std::to_string(ppL), cv::Point(20, 50), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_colorR2, std::to_string(pp85), cv::Point(20, 80), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_colorR2, std::to_string(pp95), cv::Point(20, 110), 1, 1, cv::Scalar(255, 255, 255));

		cv::imshow("RAW", im_color);
	//	cv::imshow("J2K", mJ2K);
		cv::imshow("CUBIC", im_colorR2);
		cv::imshow("J2K95", im_colorR3);

		r0.push_back(pp);
		r1.push_back(pp95);

		histDisplay(r0, "hist0");
		histDisplay(r1, "hist1");

		cv::waitKey(50);


	}

	showProcessTime();

	cv::waitKey(-1);


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

	if (input.cmdOptionExists("-encode"))
	{
		std::string in = input.getCmdOption("-in");
		std::string out = input.getCmdOption("-out");
		std::string encoder = input.getCmdOption("-encoder");
		encode(in, out, encoder);
	}
	else
	if (input.cmdOptionExists("-decode"))
		{
		std::string in = input.getCmdOption("-in");
		std::string out = input.getCmdOption("-out");
		std::string encoder = input.getCmdOption("-decoder");
		decode(in, out, encoder);
		}
	else
	if (input.cmdOptionExists("-camera"))
	{
		processCamera(-1);
	}
	else
		if (input.cmdOptionExists("-J2K"))
		{
			std::string outDir = input.getCmdOption("-out");
			compareToJ2K(outDir);
		}
		else
	if (input.cmdOptionExists("-record"))
	{
		std::string outDir = input.getCmdOption("-out");
		recordScenario(outDir,200);
	}
	else
	if (input.cmdOptionExists("-metrics"))
	{
		std::string outDir = input.getCmdOption("-out");
		processDir(outDir);
	}
	else
	if (input.cmdOptionExists("-test0"))
	{
		test0();
	}
	else std::cout << " Unknown command" << "\n";
}



