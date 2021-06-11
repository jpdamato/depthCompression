#pragma once

#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <librealsense2/rs.hpp>

#include <ST/CaptureSession.h>

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

void captureFromRealSense(int width, int height)
{
	// Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map(2);

	// Declare RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline pipe;
	rs2::config cfg;

	rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise

	// Use a configuration object to request only depth from the pipeline
	cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, 30);
	// Start streaming with the above configuration
	pipe.start(cfg);

	while (!_finishCapture)
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

////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
float* depthData = NULL;

struct SessionDelegate : ST::CaptureSessionDelegate {
	void captureSessionEventDidOccur(ST::CaptureSession *session, ST::CaptureSessionEventId event) override 
	{
		printf("Received capture session event %d (%s)\n", (int)event, ST::CaptureSessionSample::toString(event));
		switch (event) 
		{
		case ST::CaptureSessionEventId::Booting: break;
		case ST::CaptureSessionEventId::Connected:
			printf("Starting streams...\n");
			printf("Sensor Serial Number is %s \n ", session->sensorInfo().serialNumber);
			session->startStreaming();
			break;
		case ST::CaptureSessionEventId::Disconnected:
		case ST::CaptureSessionEventId::Error:
			printf("Capture session error\n");
			exit(1);
			break;
		default:
			printf("Capture session event unhandled\n");
		}
	}

	void captureSessionDidOutputSample(ST::CaptureSession *, const ST::CaptureSessionSample& sample) override 
	{
		printf("Received capture session sample of type %d (%s)\n", (int)sample.type, ST::CaptureSessionSample::toString(sample.type));
		switch (sample.type) 
		{
		case ST::CaptureSessionSample::Type::DepthFrame:
		{
			int w = sample.depthFrame.width();
			int h = sample.depthFrame.height();

			printf("Depth frame: size %dx%d\n", sample.depthFrame.width(), sample.depthFrame.height());

			// read depth matrix
			cv::Mat depth(cv::Size(w, h), CV_32FC1, (void*)sample.depthFrame.depthInMillimeters(), cv::Mat::AUTO_STEP);

			depth.convertTo(_lastDepthFrame, CV_16UC1);


			break;
		}
		case ST::CaptureSessionSample::Type::VisibleFrame:
			printf("Visible frame: size %dx%d\n", sample.visibleFrame.width(), sample.visibleFrame.height());
			break;
		default:
			printf("Sample type unhandled\n");
		}
	}
};


void captureFromStructure(int width, int height)
{

	ST::CaptureSessionSettings settings;
	settings.source = ST::CaptureSessionSourceId::StructureCore;
	settings.structureCore.depthEnabled = true;
	settings.structureCore.visibleEnabled = false;
	settings.structureCore.infraredEnabled = false;
	settings.structureCore.accelerometerEnabled = false;
	settings.structureCore.gyroscopeEnabled = false;
	settings.structureCore.depthResolution = ST::StructureCoreDepthResolution::VGA;

	settings.structureCore.depthRangeMode = ST::StructureCoreDepthRangeMode::Medium;
	settings.structureCore.initialInfraredExposure = 0.020f;
	settings.structureCore.initialInfraredGain = 1;


	SessionDelegate delegate;
	ST::CaptureSession session;
	session.setDelegate(&delegate);
	if (!session.startMonitoring(settings)) {
		printf("Failed to initialize capture session!\n");
		return ;
	}

	/* Loop forever. The SessionDelegate receives samples on a background thread
	   while streaming. */
	while (!_finishCapture) {
		std::this_thread::sleep_for(std::chrono::seconds(10));
	}

}

void starCapturing(int width, int height , std::string cameraModel)
{
	if (cameraModel == "INTEL_REALSENSE")
	{
		captureThread = std::thread(captureFromRealSense, width, height);
	}
	else
	if (cameraModel == "OCCIPITAL")
	{
		captureThread = std::thread(captureFromStructure, width, height);
	}
}

void stopCapturing()
{
	_finishCapture = true;
}