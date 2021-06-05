// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <librealsense2/rs.hpp>
#include <experimental/filesystem>
#include "bitstream.hpp"

#include "quality_metrics_OpenCV.h"

#include <chrono>
using namespace std::chrono;


//bitStream* bitS;
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
	if (!dirExists(outDir + "RAW//")) createDirectory(outDir + "RAW//");


	// ZSTD
	auto start = high_resolution_clock::now();
	splLossLess->createFromImage(depth);
	double compZ = (double)splLossLess->saveToFile(outDir + "ZSTD//" + std::to_string(frame) +".bin") / (orig_size);
	
	auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedZ = duration.count();

	// LINEAR
	start = high_resolution_clock::now();
	splLinear->createFromImage(depth);
	double compL = (double)splLinear->saveToFile(outDir + "Linear//" + std::to_string(frame) + ".bin") / (orig_size);
	
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedL = duration.count();


	// BIQUBIC
	start = high_resolution_clock::now();
	splBiQubic->createFromImage(depth);
	double compQ = (double)splBiQubic->saveToFile(outDir + "Cubic//" + std::to_string(frame) + ".bin") / (orig_size);

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
	cv::Mat depthRestoredLinear = splLinear->readFromFile(outDir + "Linear//" + std::to_string(frame) + ".bin");
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double decodeLinear = duration.count();

	start = high_resolution_clock::now();
	cv::Mat depthRestoredCubic = splBiQubic->readFromFile(outDir + "Cubic//" + std::to_string(frame) + ".bin");
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double decodeCubic = duration.count();


	start = high_resolution_clock::now();
	depth = cv::imread(outDir + "PNG//" + std::to_string(frame) + ".png", -1);
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double decodePNG = duration.count();


	std::string fn = outDir + "RAW//" + std::to_string(frame) + ".raw";
	std::ofstream out(fn, std::ios::out | std::ios::binary);
	// bin mask
	out.write((const char*)depth.data, orig_size);
	out.close();

	if (verbose)
	{
		std::cout << " LINEAR " << elapsedL << " ms" << "\n";
		std::cout << " CUBIC " << elapsedQ << " ms" << "\n";
		std::cout << " LOSSLESS " << elapsedZ << " ms" << "\n";
		std::cout << " PNG " << elapsedPNG << " ms" << "\n";
	}

	// Compute Error
	double error = qm::absDif(depth, depthRestored, 4);
	double errorL = qm::absDif(depth, depthRestoredLinear, 4);
	double errorB = qm::absDif(depth, depthRestoredCubic, 4);

	
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

int processCamera(int cam)
{
	try
	{
		// Declare depth colorizer for pretty visualization of depth data
		rs2::colorizer color_map(2);

		// Declare RealSense pipeline, encapsulating the actual device and sensors
		rs2::pipeline pipe;
		rs2::config cfg;

		rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise

						

	
		// Use a configuration object to request only depth from the pipeline
		cfg.enable_stream(RS2_STREAM_DEPTH, 1024, 0, RS2_FORMAT_Z16, 30);
		// Start streaming with the above configuration
		pipe.start(cfg);

		using namespace cv;
		const auto window_name = "Display Image";
		namedWindow(window_name, WINDOW_AUTOSIZE);
		int frameIndex = 0;

		cv::Mat accum;

		int scanrow = 500;

		splLossLess = new splineCompression(LOSSLESS_COMPRESSION);
		splLinear = new splineCompression(LINEAR_COMPRESSION);
		splBiQubic = new splineCompression(SPLINE_COMPRESSION);


		
		while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
		{
			rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
			rs2::frame depthRaw = data.get_depth_frame();

			////////////////////////
			rs2::frame filtered = temp_filter.process(depthRaw); // Does not copy the frame, only adds a reference
			depthRaw = filtered;

			rs2::frame depthC = depthRaw.apply_filter(color_map);

			// Query frame size (width and height)
			const int w = depthC.as<rs2::video_frame>().get_width();
			const int h = depthC.as<rs2::video_frame>().get_height();

			std::string date = return_current_time_and_date();
			// read depth matrix
			Mat depth(Size(w, h), CV_16UC1, (void*)depthRaw.get_data(), Mat::AUTO_STEP);

			// Create OpenCV matrix of size (w,h) from the colorized depth data
			Mat image(Size(w, h), CV_8UC3, (void*)depthC.get_data(), Mat::AUTO_STEP);

			compressionMetrics(depth, "d://temp//l515//", frameIndex, false);

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

			
			cv::putText(image, date, cv::Point(20, 20), 1, 1, cv::Scalar(255, 255, 255));
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


int processFile()
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

		cv::line(depthRestored, cv::Point(0, scanrow), cv::Point(depthRestored.cols - 1, scanrow), cv::Scalar(255, 255, 255), 3);
		cv::imshow("M1",depthRestored);
		cv::imshow("M2", depthRestoredLinear);
		cv::imshow("M3", depthRestoredCubic);

		int key = cv::waitKey(-1);

		if (key == '1') scanrow = (scanrow + 1) % 768;
		if (key == '2') scanrow = (scanrow + 10) % 768;
		if (key == '3') scanrow = (scanrow + 100) % 768;
		
	}
	return 1;
}


int main(int argc, char * argv[])
{
	if (argc == 1) processCamera(-1);
	else processFile();
}



