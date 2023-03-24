
#include <experimental/filesystem>
#include <ST/CaptureSession.h>

#include "cameraCapturer.h"

#include "bitstream.hpp"

std::mutex mtxSave;
cv::Mat _lastDepthFrame, _lastRGBFrame;
bool _finishCapture = false;
std::thread captureThread,  storeThread;
std::string empty_string = "";
// Declare object that handles camera pose calculations
std::mutex lckFrameRead;
int counter = 1;
cameraCapturer* _instance = NULL;
bool saveToPNG = false;
std::mutex mtxFrame;

// This example assumes camera with depth and color
// streams, and direction lets you define the target stream
enum class direction
{
	to_depth,
	to_color, 
	none
};

void logData(std::string s, int level )
{
	std::cout << "[ERROR] " << s << "\n";
}

std::string  toUpperCase(std::string& str)
{
	std::transform(str.begin(), str.end(), str.begin(), ::toupper);

	return str;
}

std::string return_current_time_and_date()
{
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);

	std::ostringstream oss;
	oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
	auto str = oss.str();
	return str;
}

void exportCSV(std::string outfilename, std::vector<std::string> values, int time)
{
	std::ofstream myfile(outfilename, std::ios::out | std::ios::app);
	myfile << time;
	for (int i = 0; i < values.size(); i++)
	{
		myfile << ";" << values[i];
	}
	myfile << "\n";
	myfile.close();
}
// check if all sensors has IMU
bool check_imu_is_supported(std::string serial)
{
	bool found_gyro = true;
	bool found_accel = true;
	rs2::context ctx;
	for (auto dev : ctx.query_devices())
	{
	 if ( std::string(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) ) == serial )
	 {
		// The same device should support gyro and accel
		found_gyro = false;
		found_accel = false;

		for (auto sensor : dev.query_sensors())
		{

			for (auto profile : sensor.get_stream_profiles())
			{
				if (profile.stream_type() == RS2_STREAM_GYRO)
					found_gyro = true;

				if (profile.stream_type() == RS2_STREAM_ACCEL)
					found_accel = true;
			}
		}

		if (found_gyro && found_accel)
			return true;
	 }
		
	}
	return false;
}

bool dirExists(const std::string& dirPath)
{

	return std::experimental::filesystem::exists(dirPath) && std::experimental::filesystem::is_directory(dirPath);

}


bool createDirectory(const std::string& name)
{
	return std::experimental::filesystem::create_directory(name);
}


void enableFilters()
{
	// Declare filters
	rs2::decimation_filter dec_filter;  // Decimation - reduces depth frame density
	rs2::threshold_filter thr_filter;   // Threshold  - removes values outside recommended range
	rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
	rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise


}


cv::Mat cvtToMAT(rs2::frame color_frame, int color_height, int color_width, bool rescale = false)
{
	cv::Mat color_mat;

	if (!color_frame) {
		return color_mat;
	}

	// Create cv::Mat form Color Frame
	const rs2_format color_format = color_frame.get_profile().format();
	switch (color_format) {
		// RGB8
	case rs2_format::RS2_FORMAT_RGB8:
	{
		color_mat = cv::Mat(color_height, color_width, CV_8UC3, const_cast<void*>(color_frame.get_data()));
		cv::cvtColor(color_mat, color_mat, cv::COLOR_RGB2BGR);
		break;
	}
	// RGBA8
	case rs2_format::RS2_FORMAT_RGBA8:
	{
		color_mat = cv::Mat(color_height, color_width, CV_8UC4, const_cast<void*>(color_frame.get_data()));
		cv::cvtColor(color_mat, color_mat, cv::COLOR_RGBA2BGRA);
		break;
	}
	// BGR8
	case rs2_format::RS2_FORMAT_BGR8:
	{
		color_mat = cv::Mat(color_height, color_width, CV_8UC3, const_cast<void*>(color_frame.get_data()));
		break;
	}
	// BGRA8
	case rs2_format::RS2_FORMAT_BGRA8:
	{
		color_mat = cv::Mat(color_height, color_width, CV_8UC4, const_cast<void*>(color_frame.get_data()));
		break;
	}
	// Y16 (GrayScale)
	case rs2_format::RS2_FORMAT_Y16:
	{
		color_mat = cv::Mat(color_height, color_width, CV_16UC1, const_cast<void*>(color_frame.get_data()));
		if (rescale)
		{
			constexpr double scaling = static_cast<double>(std::numeric_limits<uint8_t>::max()) / static_cast<double>(std::numeric_limits<uint16_t>::max());
			color_mat.convertTo(color_mat, CV_8U, scaling);
		}
		break;
	}
	// Z16 (GrayScale)
	case rs2_format::RS2_FORMAT_Z16:
	{
		color_mat = cv::Mat(color_height, color_width, CV_16UC1, const_cast<void*>(color_frame.get_data()));
		if (rescale)
		{
			constexpr double scaling = 0.01;
			color_mat.convertTo(color_mat, CV_8U, scaling);
		}
		break;
	}
	// YUYV
	case rs2_format::RS2_FORMAT_YUYV:
	{
		color_mat = cv::Mat(color_height, color_width, CV_8UC2, const_cast<void*>(color_frame.get_data()));
		cv::cvtColor(color_mat, color_mat, cv::COLOR_YUV2BGR_YUYV);
		break;
	}
	default:
		//	std::cout << "unknown color format" << "\n";
		break;
	}

	return color_mat;
}


void renderCapturers(std::map<std::string, cv::Mat>&  lastFrames)
{
	// render
	for (auto m : lastFrames)
	{
		cv::imshow(m.first, m.second);
	}

	cv::waitKey(1);
}

//////////////////////////////////////////////////////
///// InputParser
//////////////////////////////////////////////////////


InputParser::InputParser(int &argc, char **argv) 
{
	for (int i = 1; i < argc; ++i)
		this->tokens.push_back(std::string(argv[i]));
}

/// @author iain
std::string InputParser::getCmdOption(const std::string &option) 
{
	std::vector<std::string>::const_iterator itr;
	itr = std::find(this->tokens.begin(), this->tokens.end(), option);
	if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
		return *itr;
	}
	static const std::string empty_string("");
	return empty_string;
}

/// @author iain
bool InputParser::cmdOptionExists(const std::string &option) 
{
	return std::find(this->tokens.begin(), this->tokens.end(), option)
		!= this->tokens.end();
}


//////////////////////////////////////////////////////
/// Thread To Record
//////////////////////////////////////////////////////

void thRecord(rsCameraRecorder* cs)
{
	// 
	int frameNumber = 0;

	cv::Mat d, rgb, rgbNA;


	std::string outDirPNG = cameraCapturer::getInstance()->outputDir + cameraCapturer::getInstance()->startRecordingTime + "/" + cameraCapturer::getInstance()->startRecordingTime + "_" + cs->serial + "/";

	std::cout << "RS-MP4 : Creating directory for  PNG: " + outDirPNG + " camera serial " + cs->serial << "\n";


	if (!dirExists(outDirPNG))
	{
		createDirectory(outDirPNG);
	}


	std::map <std::string, splineCompression*> spl;

	while (!_finishCapture && !cs->finishCapture)
	{
		try
		{
			// check if there is a frame to save
			if (!cameraCapturer::getInstance()->isRecording)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			}

			auto last_time = cv::getTickCount();

			auto milliseconds = 1000 * (last_time - cs->timeNow) / cv::getTickFrequency();


			// image is empty
			if (cs->depth.cols == 0)
			{
				continue;
			}

			if (spl[cs->serial] == NULL)
			{
				spl[cs->serial] = new splineCompression(LINEAR_COMPRESSION);
				spl[cs->serial]->zstd_compression_level = 0;
			}

			// ya pasaron 33 ms
			if (milliseconds >= 1000.0 / globalFrameRate)
			{
				auto nw0 = cv::getTickCount();
				cv::Mat d, rgb;

				cs->setM.lock();
				if (d.cols == 0)
				{
					d = cs->depth.clone();
					if (cameraCapturer::getInstance()->alignment[cs->serial]) rgb = cs->rgb.clone();
					rgbNA = cs->rgbNotAligned.clone();
				}
				else
				{
					cs->depth.copyTo(d);
					if (cameraCapturer::getInstance()->alignment[cs->serial]) cs->rgb.copyTo(rgb);
					cs->rgbNotAligned.copyTo(rgbNA);

				}
				cs->setM.unlock();

				//comes in MP4 Format
				if (cs->recorderRGBNotAligned == NULL)
				{
					
					cs->startCapturers(cameraCapturer::getInstance()->alignment[cs->serial]);
				}
				// Save frame
				if (cs->recorderRGBNotAligned)
				{
					cs->recorderRGBNotAligned->write(rgbNA);
				}

				if (saveToPNG)
				{
					if (cameraCapturer::getInstance()->alignment[cs->serial]) cs->recorderRGB->write(rgb);
					cv::imwrite(outDirPNG + std::to_string(frameNumber) + ".png", d);
				}
				else
				{
					std::string outputF = outDirPNG + std::to_string(frameNumber) + ".fit";
					startProcess("encode");
					spl[cs->serial]->encode(d, outputF);
					double time = endProcess("encode");

				}

				// Save angles
				if (frameNumber % 20 == 0)
				{
					cv::Vec3f acc = cameraCapturer::getInstance()->getAccelerometerAngles(cs->serial);
					cv::Vec3f gyro = cameraCapturer::getInstance()->getGyroAngles(cs->serial);
					std::ofstream myfile(outDirPNG + "angles.txt", std::ios::out | std::ios::app);
				
					myfile << gyro[0] << ";" << gyro[1] << ";" << gyro[2];
					myfile << "\n";
					myfile.close();
				}

				cs->timeNow = last_time;
				frameNumber++;

				double ms0 = (cv::getTickCount() - nw0) / cv::getTickFrequency();

				if (frameNumber % 100 == 0) std::cout << "RS-MP4 : saving frame: " + cs->outputFile << " time for saving " << ms0*1000 << "\n";
			}
			else
			{
				//std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}
		catch (std::exception ex)
		{
			logData("Excetion at recording thread " + std::string( ex.what() ));
		}
	}


	logData("RS-MP4 :Finished recording " + cs->serial);

}

//////////////////////////////////////////////////////
/// cameraCapturer
//////////////////////////////////////////////////////

cv::Mat cameraCapturer::getFrame(std::string cameraID, std::string frameType)
{
	cv::Mat copy;

	if (lastFrames.find(cameraID+ frameType) != lastFrames.end() )
	{
		lckFrameRead.lock();
		copy = lastFrames[cameraID+ frameType].clone();

		lckFrameRead.unlock();

		// If it is depth, change to other format
		if (copy.type() == CV_16UC1)
		{
			copy.convertTo(copy, CV_8U, 1.0 / 128.0);
		}
	}	

	return copy;
}


float cameraCapturer::getFrameRate(std::string cameraID)
{
	
	if (frameRate.find(cameraID) != frameRate.end())
	{
		
		if (frameAccum[cameraID] == 0) return 1;

		float fps = frameRate[cameraID] / frameAccum[cameraID];
		if (fps > 30)
			fps = 30;
		return fps;
	}

	return 0.0;
}

void cameraCapturer::setAutoExposure(std::string cameraID, bool state)
{
	if (profiles.find(cameraID) != profiles.end())
	{
		auto sensor = profiles[cameraID].get_device().first<rs2::color_sensor>();
		// Set the exposure for Color Sensor
	// CHECK THIS or AUTO EXPOSURE
		if (sensor && sensor.is<rs2::color_sensor>())
		{
			sensor.as<rs2::color_sensor>().set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, state);
		}
	}
}


void cameraCapturer::setAlignment(std::string cameraID, bool state)
{
	if (alignment.find(cameraID) != alignment.end())
	{
		alignment[cameraID] = state;
	}
}


cv::Vec3f cameraCapturer::getGyroAngles(std::string cameraID)
{
	cv::Vec3f v(0, 0, 0);

	if (gyroscopes.find(cameraID) != gyroscopes.end())
	{
		v = gyroscopes[cameraID];
	}

	return v;
}


cv::Vec3f cameraCapturer::getAccelerometerAngles(std::string cameraID)
{
	cv::Vec3f v(0,0,0);

	if (accelerometers.find(cameraID) != accelerometers.end())
	{
		v = accelerometers[cameraID];
	}
	
	return v;
}


int cameraCapturer::processBagFile(std::string inputFilename, std::string outputVideoRGB, std::string outputVideoDepth)
{
		// Declare RealSense pipeline, encapsulating the actual device and sensors
		rs2::pipeline pipe;

		rs2::frameset frameset;
		// try_wait_for_frames will keep repeating the last frame at the end of the file,
		  // so we need to exit the look in some other way!
		


		cv::VideoWriter* vidOutRGB = NULL;
		cv::VideoWriter* vidOutDepth = NULL;

		try
		{
			cfg.enable_device_from_file(inputFilename);
			pipe.start(cfg);

			auto playback = ctx.load_device(inputFilename);
			playback.set_real_time(false);
			std::vector<rs2::sensor> sensors = playback.query_sensors();
			std::mutex mutex;

			auto duration = playback.get_duration();
			int progress = 0;
			uint64_t posCurr = playback.get_position();


			rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise
			rs2::colorizer color_map(2);

			cv::Mat rgbI;

			int lastFrameNumber = 0;
			// try_wait_for_frames will keep repeating the last frame at the end of the file,
			// so we need to exit the look in some other way!
			while (pipe.try_wait_for_frames(&frameset, 1000))
			{
				int posP = static_cast<int>(posCurr * 100. / duration.count());
				if (posP > progress)
				{
					progress = posP;
					std::cout << posP << "%" << "\r" << std::flush;
				}

				int frameNumber = frameset[0].get_frame_number();

				if (frameNumber < lastFrameNumber) break;

				lastFrameNumber = frameNumber;

				rs2::frame depthRaw = frameset.get_depth_frame();

				rs2::frame rgb = frameset.get_color_frame();


				////////////////////////
				rs2::frame filtered = temp_filter.process(depthRaw); // Does not copy the frame, only adds a reference
				depthRaw = filtered;


				// Query frame size (width and height)
				const int w = depthRaw.as<rs2::video_frame>().get_width();
				const int h = depthRaw.as<rs2::video_frame>().get_height();

				const int w_rgb = rgb.as<rs2::video_frame>().get_width();
				const int h_rgb = rgb.as<rs2::video_frame>().get_height();


				rgbI = cvtToMAT( rgb, h_rgb, w_rgb);

				// read depth matrix
				cv::Mat depth(cv::Size(w, h), CV_16UC1, (void*)depthRaw.get_data(), cv::Mat::AUTO_STEP);

				depth.convertTo(depth, CV_8UC1, 1.0 / 255.0);

				if (!vidOutRGB) vidOutRGB = new cv::VideoWriter(outputVideoRGB, cv::VideoWriter::fourcc('X', '2', '6', '4'),
					depthRaw.as<rs2::video_frame>().get_profile().fps(), rgbI.size());

				vidOutRGB->write(rgbI);

				if (!vidOutDepth) vidOutDepth = new cv::VideoWriter(outputVideoDepth, cv::VideoWriter::fourcc('X', '2', '6', '4'),
					depthRaw.as<rs2::video_frame>().get_profile().fps(), rgbI.size());

				vidOutDepth->write(depth);

				auto posNext = playback.get_position();


				if (posNext < posCurr)
					break;

				posCurr = posNext;

				cv::waitKey(10);

			}

			vidOutDepth->release();
			vidOutRGB->release();
		}
		catch (rs2::error ex)
		{
			std::cout << ex.what() << "\n";
		}

		return 1;
	}

///////////////////////
// Store to  MP4 Format
void cameraCapturer::captureFromRealSenseToMP4( std::string outputDir)
{

		rs2::context                          ctx;        // Create librealsense context for managing devices
		std::vector<rs2::pipeline*> pipelines;

		

		// We'll keep track of the last frame of each stream available to make the presentation persistent
		std::map<int, rs2::frame> render_frames;

		// Capture serial numbers before opening streaming
				
		camera_serials.clear();
		
		float       alpha = 0.5f;               // Transparancy coefficient 
		direction   dir = direction::none;  // Alignment direction
		this->outputDir = outputDir;




		try
		{
			logData("RS-MP4 : Querying devices ");

			generatePipelines(ctx, pipelines, false);
		}
		catch (rs2::error ex)
		{
			logData("RS-MP4 : ERROR   " +std::string( ex.what()),1);

		}

		try
		{
	// Define two align objects. One will be used to align
	// to depth viewport and the other to color.
	// Creating align object is an expensive operation
	// that should not be performed in the main loop
			 // Declare object that handles camera pose calculations
			logData("RS-MP4 : Starting capture  of " + std::to_string(pipelines.size())  + " pipes ");
			frameIndex = 0;

			cv::Mat depths[4];
			cv::Mat rgbs[4];
			// Collect the new frames from all the connected devices
			std::vector<std::pair< std::string, cv::Mat> > localFrames;

			
			int64 nw = cv::getTickCount();

			// Main app loop
			while (!_finishCapture && frameIndex < cameraCapturer::getInstance()->maxFrames)
			{
				// Collect the new frames from all the connected devices
				std::vector<rs2::frame> new_frames;

				int64 nw = cv::getTickCount();

				// Read available frames
				int pipe_ind = 0;
				std::mutex addMtx;
				

				/// process all frames
				for (int i = 0; i< pipelines.size();i++)
				{
					rs2::frameset fs;

					if (pipelines[i]->poll_for_frames(&fs))
					{
						// pre-aligned
						auto frame = fs.get_color_frame();
						int h = frame.as<rs2::video_frame>().get_height();
						int w = frame.as<rs2::video_frame>().get_width();
						cv::Mat m = cvtToMAT(frame, h, w);
						auto serial = rs2::sensor_from_frame(frame)->get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
						if (camerasRec[serial]) camerasRec[serial]->setStream("ColorNotAligned", m);

						lastFrames[serial +std::string( "ColorNotAligned")] = m.clone();

						
							if (alignment[serial])
							{
								fs = aligners[serial]->process(fs);
							}
							//save in temporal list
							addMtx.lock();
							for (const rs2::frame& f : fs)
							{
								new_frames.emplace_back(f);
							}
							addMtx.unlock();
						
						// Compute frameRate
						double ms = (cv::getTickCount() - frameTime[serial]) / cv::getTickFrequency();
						frameRate[serial] = frameRate[serial] + 1.0 / ms;
						frameTime[serial] = cv::getTickCount();
						frameAccum[serial] = frameAccum[serial] + 1;

						if (frameIndex % 100 == 0)
						{
							std::cout << frameRate[serial] / frameAccum[serial] << "\n";
						}
					}
				}

				double timeForReading = (cv::getTickCount() - nw) / cv::getTickFrequency();
			
				//process this list
				if (new_frames.size() > 0)
				{
					nw = cv::getTickCount();
					// Convert the newly-arrived frames to render-friendly format
					for (const auto& frame : new_frames)
					{
						auto serial = rs2::sensor_from_frame(frame)->get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
						if (frame.as<rs2::video_frame>())
						{
							// Get the serial number of the current frame's device

							int h = frame.as<rs2::video_frame>().get_height();
							int w = frame.as<rs2::video_frame>().get_width();
							
							{
								cv::Mat m = cvtToMAT(frame, h, w);

								std::string outputFile = outputDir + startRecordingTime + "/" + startRecordingTime + "_" + frame.get_profile().stream_name() + "_" + serial + ".mp4";

								if (m.cols > 0)
								{
									
									if (isRecording &&  camerasRec[serial]) camerasRec[serial]->setStream(frame.get_profile().stream_name(), m);

									/// Copy last frame
									lckFrameRead.lock();
									lastFrames[serial + frame.get_profile().stream_name()] = m.clone();
									lckFrameRead.unlock();
									
								}
							}
						}
						else
							if (frame.as < rs2::motion_frame>())
							{
								auto motion = frame.as < rs2::motion_frame>();

								if ( motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
								{
									// Get the timestamp of the current frame
									double ts = motion.get_timestamp();
									// Get gyro measures
									rs2_vector gyro_data = motion.get_motion_data();
									// Call function that computes the angle of motion based on the retrieved measures
									//algo.process_gyro(gyro_data, ts);
									if (frameIndex % 100 == 0) std::cout << serial << "::Gyro x " << gyro_data.x << " y " << gyro_data.y << " z " << gyro_data.z << "\n";
									gyroscopes[serial] = cv::Vec3f(gyro_data.x, gyro_data.y, gyro_data.z);
								}

								// If casting succeeded and the arrived frame is from accelerometer stream
								if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
								{
									// Get accelerometer measures
									rs2_vector accel_data = motion.get_motion_data();
									// Call function that computes the angle of motion based on the retrieved measures
									//algo.process_accel(accel_data);

									if (frameIndex % 100 == 0) std::cout << serial << "::ACC x " << accel_data.x << " y " << accel_data.y << " z " << accel_data.z << "\n";
									accelerometers[serial] = cv::Vec3f(accel_data.x, accel_data.y, accel_data.z);
								}
								
								

							}

						// Apply the colorizer of the matching device and store the colorized frame
#ifdef _WIN32
						render_frames[frame.get_profile().unique_id()] = colorizers[serial].process(frame);
#endif
						
					}

					double timeForProcessing = (cv::getTickCount() - nw) / cv::getTickFrequency();

					if (frameIndex % 100 == 0)
					{
						std::cout << "RS-MP4 : frame captured at " << 1 / timeForProcessing << " fps. Frams" << frameIndex << "\n";
						std::cout << " Time for reading " << timeForReading* 1000 << "ms \n";
						std::cout << " Time for processing " << timeForProcessing * 1000 << "ms \n";
					}


					nw = cv::getTickCount();

					frameIndex++;

				}

				renderCapturers(lastFrames);
				
			}

			//Stop all Pipes
			for (auto pipe : pipelines)
			{
				pipe->stop(); //File will be closed at this point
			}
			// Force stop recording
			setRecordingState(false);

		}
		catch (rs2::error ex)
		{
			std::cout << ex.what() << "\n";
			logData("RS-MP4 : ERROR   " + std::string(ex.what()), 1);
		}
	}

void cameraCapturer::setRecordingState(bool recording)
{
		// already recording
		if (isRecording && recording) return;
		// already stopped
		if (!isRecording && !recording) return;

		std::string date =  return_current_time_and_date() ;
		
		// I receive a START Recording
		if (recording)
		{
			if (outputFormat == "MP4")
			{
				startRecordingTime = return_current_time_and_date();
				if (!dirExists(outputDir + startRecordingTime))
				{
					createDirectory(outputDir + startRecordingTime);
				}

				camerasRec.clear();

				// init all recorders
				rs2::context                          ctx;
				for (auto&& dev : ctx.query_devices())
				{
					std::string serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
					rsCameraRecorder* cs = new rsCameraRecorder(serial);
					cs->outputFile = outputDir + startRecordingTime + "/" + date + "-" + serial;
					cs->timeNow = cv::getTickCount();

					/// multi thread
					cs->th = std::thread(thRecord, cs);

					camerasRec[serial] = cs;
				}

				isRecording = true;
				frameIndex = 0;
			}
			else
			{
				isRecording = true;
			}
		}
		else
		{
			isRecording = false;

			// wait for all frames to be saved
			mtxSave.lock();
			
			//// finish all recorders
			std::map<std::string, rsCameraRecorder*>::iterator it = camerasRec.begin();
			// Iterate over the map using Iterator till end.
			while (it != camerasRec.end())
			{
				/// multi thread

				it->second->releaseCapture();
				
				it++;
			}

			camerasRec.clear();
			mtxSave.unlock();

		}
	}

///////////////////////
// Store to BAG Format
void cameraCapturer::generatePipelines(rs2::context& ctx, std::vector<rs2::pipeline*>&  pipelines, bool recordToBag)
{
	std::string date = return_current_time_and_date();
	// Capture serial numbers before opening streaming

	camera_serials.clear();
	aligners.clear();
	pipelines.clear();

	for (auto&& dev : ctx.query_devices())
	{
		std::string serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
		camera_serials.push_back(serial);
		alignment[serial] = defaultAlignmentState;

		aligners[serial] = new  rs2::align(RS2_STREAM_DEPTH);
		frameAccum[serial] = 0;
	}

	logData("RS : Enabling streams  ");

	// Start a streaming pipe per each connected device
	for (auto&& serial : camera_serials)
	{
		rs2::pipeline* pipe= new rs2::pipeline(ctx);
		rs2::config cfg;
		cfg.enable_device(serial);
		//cfg.enable_all_streams();

		// upper cameras. Higher resolution. S1 or S2
		if (serial == "f1120419" || serial == "f0350334" || serial == "00000000f1062125" || serial == "f1181925" || serial == "f1120833" || serial == "f1062125")
		{
			std::cout << " Camera " << serial << " enabling high quality streamings" << "\n";
			cfg.enable_stream(RS2_STREAM_DEPTH, 1024, 768, RS2_FORMAT_Z16, 30);
			cfg.enable_stream(RS2_STREAM_COLOR, HIGH_CAMERAS_WIDTH, HIGH_CAMERAS_HEIGHT, RS2_FORMAT_BGR8, 30);
		}
		else
			// lower cameras
		{
			std::cout << " Camera " << serial << " enabling low quality streamings" << "\n";
			cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
			cfg.enable_stream(RS2_STREAM_COLOR, LOW_CAMERAS_WIDTH, LOW_CAMERAS_HEIGHT, RS2_FORMAT_BGR8, 30);
		}

		// Add streams of gyro and accelerometer to configuration
		if (check_imu_is_supported(serial))
		{
			cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 100);
			cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 100);
		}

		// Enable default configuration
		if (recordToBag)
		{
			std::string fileName = outputDir + startRecordingTime + "/" + date + "_" + serial + ".bag";
			cfg.enable_record_to_file(fileName);
		}

		profiles[serial] = pipe->start(cfg);
		//// By default, auto exposure ON
		setAutoExposure(serial, true);

		pipelines.emplace_back(pipe);

		logData("--- RS-MP4 : Enable stream  OK" + serial);
		// Map from each device's serial number to a different colorizer
		colorizers[serial] = rs2::colorizer();

		temporal_filters[serial] = new 	rs2::temporal_filter();

		logData("--- RS-MP4 : Enable Filters  OK" + serial);


	}
}

void cameraCapturer::captureFromRealSenseToBAG( std::string outputDir)
{
		try
		{

			rs2::context                          ctx;        // Create librealsense context for managing devices

			std::vector<rs2::pipeline*>     pipelines;

			logData("RS-BAG : Querying devices ");

			

			// Start a streaming pipe per each connected device
			int i = 0;
		

			int frameNumber = 0;
		
			logData("RS-BAG :  Starting capture " );
			cv::Mat depth;
			cv::Mat image;

			bool startRecording = false;
		
			while (!_finishCapture)
			{
				std::map<std::string, cv::Mat > localFrames;

				int64 nw = cv::getTickCount();
				// Collect the new frames from all the connected devices
				if (isRecording)
				{
					if (!startRecording)
					{
						logData("RS-BAG : Initializing ");

						generatePipelines(ctx, pipelines, true);

						startRecording = true;

					}
				}
				else
				{
					if (startRecording)
					{
						// Stop all Pipes
						for (auto pipe : pipelines)
						{
							pipe->stop(); //File will be closed at this point
						}

						pipelines.clear();
						startRecording = false;
					}

				}
	
				for (auto pipe : pipelines)
				{
					rs2::frameset fs;
					if (pipe->poll_for_frames(&fs))
					{
						rs2::frameset data = pipe->wait_for_frames(); // Wait for next set of frames from the camera
						rs2::frame depthRaw = data.get_depth_frame();
						rs2::frame rgb = data.get_color_frame();

						std::string serial = pipe->get_active_profile().get_device().get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

						// Extract motion Info
						for (const auto& frame : fs)
						{
							if (frame.as < rs2::motion_frame>())
							{
								auto motion = frame.as < rs2::motion_frame>();

								if (motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
								{
									// Get the timestamp of the current frame
									double ts = motion.get_timestamp();
									// Get gyro measures
									rs2_vector gyro_data = motion.get_motion_data();
									// Call function that computes the angle of motion based on the retrieved measures
									//algo.process_gyro(gyro_data, ts);
									if (frameIndex % 100 == 0) std::cout << serial << "::Gyro x " << gyro_data.x << " y " << gyro_data.y << " z " << gyro_data.z << "\n";
									gyroscopes[serial] = cv::Vec3f(gyro_data.x, gyro_data.y, gyro_data.z);
								}

								// If casting succeeded and the arrived frame is from accelerometer stream
								if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
								{
									// Get accelerometer measures
									rs2_vector accel_data = motion.get_motion_data();
									// Call function that computes the angle of motion based on the retrieved measures
									//algo.process_accel(accel_data);

									if (frameIndex % 100 == 0) std::cout << serial << "::ACC x " << accel_data.x << " y " << accel_data.y << " z " << accel_data.z << "\n";
									accelerometers[serial] = cv::Vec3f(accel_data.x, accel_data.y, accel_data.z);
								}
							}
						}

						if (depthRaw != NULL)
						{
							// Query frame size (width and height)
							const int w = depthRaw.as<rs2::video_frame>().get_width();

							const int h = depthRaw.as<rs2::video_frame>().get_height();
							// read depth matrix

							depth = cvtToMAT(depthRaw, h, w);

							localFrames["depth" + serial] = depth;
						}


						if (rgb != NULL)
						{
							const int w = rgb.as<rs2::video_frame>().get_width();

							const int h = rgb.as<rs2::video_frame>().get_height();

							// Create OpenCV matrix of size (w,h) from the colorized depth data
							image = cvtToMAT(rgb, h, w);

							localFrames["rgb" + serial] = image;
						}
					}
				}

				if (localFrames.size() > 0)
				{
					double ms = (cv::getTickCount() - nw) / cv::getTickFrequency() * 1000;
					if (frameNumber % 100 == 0)  std::cout << "RS-BAG : frame captured at " + std::to_string(ms) << std::endl;
					
					renderCapturers(localFrames);

					frameNumber++;
				}
				
				
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
				

			}

			// Stop all Pipes
			for (auto pipe : pipelines)
			{
				pipe->stop(); //File will be closed at this point
			}

			
		}
		catch (std::exception ex)
		{
			logData("RS-BAG :  Error " + std::string(ex.what()),1);

		}
	}

void cameraCapturer::startCapturing()
{

		if (toUpperCase( outputFormat) == "BAG")
		{
			captureFromRealSenseToBAG(outputDir);
		}
		else
		{
			captureFromRealSenseToMP4(outputDir);
		}
	}

	void cameraCapturer::stopCapturing()
	{

	}

	cameraCapturer::cameraCapturer(int w, int h,int fps, std::string format, std::string dir)
	{
		width = w;
		height = h;
		outputFormat = format;
		outputDir = dir;
		
		_instance = this;
	}


	cameraCapturer* cameraCapturer::getInstance()
	{
		if (!_instance) _instance = new cameraCapturer(1,1,1,"MP4", "");
		return _instance;
	}


void mainLoop()
{

	if (cameraCapturer::getInstance()->outputFormat == "")
	{
		/// Waiting for response from Server

		std::cout << "Waiting notification from server " << "\n";

		while (true)
		{
			/////////////////////

			if (cameraCapturer::getInstance()->outputFormat != "")
				break;

			std::this_thread::sleep_for(std::chrono::milliseconds(10));			
		}

		std::cout << "Now start capturing " << "\n";
	}
	
	_instance->startCapturing();

}

void starCapturing(int width, int height,int fps, std::string outputFormat, std::string outputDir)
{
	_finishCapture = false;

	cameraCapturer::getInstance()->outputDir = outputDir;
	cameraCapturer::getInstance()->outputFormat = outputFormat;


	captureThread = std::thread(mainLoop);
	// Single thread recording
//	storeThread = std::thread(recording);

}

void stopCapturing()
{
	_finishCapture = true;

	captureThread.join();

// Single thread recording
//	storeThread.join();
	if (_instance)
	{
		delete _instance;

		_instance = NULL;
	}
}

//////////////////////////////////////

cv::Mat getLastDepthFrame()
{
	cv::Mat clone;

	mtxFrame.lock();
	clone = _lastDepthFrame.clone();
	mtxFrame.unlock();
	return clone;
}


cv::Mat getLastRGBFrame()
{
	return _lastRGBFrame;
}

void captureFromRealSense(int width, int height)
{
#ifdef REAL_SENSE
	// Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map(2);

	// Declare RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline pipe;
	rs2::config cfg;

	rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise

	// Use a configuration object to request only depth from the pipeline
	cfg.enable_stream(RS2_STREAM_DEPTH, width, 0, RS2_FORMAT_Z16, 30);
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

		mtxFrame.lock();
		_lastDepthFrame = depth.clone();
		_lastRGBFrame = image.clone();
		mtxFrame.unlock();

		std::this_thread::sleep_for(std::chrono::milliseconds(10));

	}
#endif
}

////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
float* depthData = NULL;
#ifdef REAL_SENSE
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
#endif

void captureFromStructure(int width, int height)
{
#ifdef REAL_SENSE
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
		return;
	}

	/* Loop forever. The SessionDelegate receives samples on a background thread
	   while streaming. */
	while (!_finishCapture) {
		std::this_thread::sleep_for(std::chrono::seconds(10));
	}
#endif
}

void starCapturing(int width, int height, std::string cameraModel)
{
	if (cameraModel == "L515")
	{
		captureThread = std::thread(captureFromRealSense, width, height);
	}
	else
		if (cameraModel == "D415")
		{
			captureThread = std::thread(captureFromRealSense, 1280, 720);
		}
		else
			if (cameraModel == "OCCIPITAL")
			{
				captureThread = std::thread(captureFromStructure, width, height);
			}


}

