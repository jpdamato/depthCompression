#pragma once

#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <librealsense2/rs.hpp>
#include <thread>

cv::Mat _lastDepthFrame, _lastRGBFrame;
bool _finishCapture = false;
std::thread captureThread;

cv::Mat getLastDepthFrame()
{
	return _lastDepthFrame;
}


cv::Mat getLastRGBFrame()
{
	return _lastRGBFrame;
}

void captureFromRealSense()
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

	while (_finishCapture)
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

		
		// read depth matrix
		cv::Mat depth(cv::Size(w, h), CV_16UC1, (void*)depthRaw.get_data(), cv::Mat::AUTO_STEP);

		// Create OpenCV matrix of size (w,h) from the colorized depth data
		cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)depthC.get_data(), cv::Mat::AUTO_STEP);

		_lastDepthFrame = depth.clone();
		_lastRGBFrame = image.clone();

	}
}



void starCapturing()
{
	captureThread = std::thread(captureFromRealSense);
}

void stopCapturing()
{
	_finishCapture = true;
}