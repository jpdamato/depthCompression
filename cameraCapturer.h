#pragma once

#include <opencv2/opencv.hpp>   // Include OpenCV API

#include <librealsense2/rs.hpp>

#include <thread>
#include <mutex>

#ifndef PI
const double PI = 3.14159265358979323846;
#endif

#ifdef _WIN32
const float globalFrameRate = 30; //A cuantos fraes graba
const bool defaultAlignmentState = true; // si usa alineacion o no
const int HIGH_CAMERAS_WIDTH = 1920; // Full HD resolucion por defecto de camara RGB
const int HIGH_CAMERAS_HEIGHT = 1080; // HD
const int LOW_CAMERAS_WIDTH = 1280; // HD
const int LOW_CAMERAS_HEIGHT = 720; // HD

#else
const float globalFrameRate = 25;
const bool defaultAlignmentState = false;
const int HIGH_CAMERAS_WIDTH = 1920; // HD
const int HIGH_CAMERAS_HEIGHT = 1080; // HD

const int LOW_CAMERAS_WIDTH = 1280; // HD
const int LOW_CAMERAS_HEIGHT = 720; // HD

#endif

//////////////////////////////
// Basic Data Types         //
//////////////////////////////

struct xfloat3 {
	float x, y, z;
	xfloat3 operator*(float t)
	{
		return { x * t, y * t, z * t };
	}

	xfloat3 operator-(float t)
	{
		return { x - t, y - t, z - t };
	}

	void operator*=(float t)
	{
		x = x * t;
		y = y * t;
		z = z * t;
	}

	void operator=(xfloat3 other)
	{
		x = other.x;
		y = other.y;
		z = other.z;
	}

	void add(float t1, float t2, float t3)
	{
		x += t1;
		y += t2;
		z += t3;
	}
};


class InputParser {
public:
	InputParser(int &argc, char **argv);
	/// @author iain
	std::string getCmdOption(const std::string &option) ;
	/// @author iain
	bool cmdOptionExists(const std::string &option) ;
private:
	std::vector <std::string> tokens;
};


class rotation_estimator
{
	// theta is the angle of camera rotation in x, y and z components
	xfloat3 theta;
	std::mutex theta_mtx;
	/* alpha indicates the part that gyro and accelerometer take in computation of theta; higher alpha gives more weight to gyro, but too high
	values cause drift; lower alpha gives more weight to accelerometer, which is more sensitive to disturbances */
	float alpha = 0.98;
	bool first = true;
	// Keeps the arrival time of previous gyro frame
	double last_ts_gyro = 0;
public:
	// Function to calculate the change in angle of motion based on data from gyro
	void process_gyro(rs2_vector gyro_data, double ts)
	{
		if (first) // On the first iteration, use only data from accelerometer to set the camera's initial position
		{
			last_ts_gyro = ts;
			return;
		}
		// Holds the change in angle, as calculated from gyro
		xfloat3 gyro_angle;

		// Initialize gyro_angle with data from gyro
		gyro_angle.x = gyro_data.x; // Pitch
		gyro_angle.y = gyro_data.y; // Yaw
		gyro_angle.z = gyro_data.z; // Roll

		// Compute the difference between arrival times of previous and current gyro frames
		double dt_gyro = (ts - last_ts_gyro) / 1000.0;
		last_ts_gyro = ts;

		// Change in angle equals gyro measures * time passed since last measurement
		gyro_angle = gyro_angle * dt_gyro;

		// Apply the calculated change of angle to the current angle (theta)
		std::lock_guard<std::mutex> lock(theta_mtx);
		theta.add(-gyro_angle.z, -gyro_angle.y, gyro_angle.x);
	}

	void process_accel(rs2_vector accel_data)
	{
		// Holds the angle as calculated from accelerometer data
		xfloat3 accel_angle;

		// Calculate rotation angle from accelerometer data
		accel_angle.z = atan2(accel_data.y, accel_data.z);
		accel_angle.x = atan2(accel_data.x, sqrt(accel_data.y * accel_data.y + accel_data.z * accel_data.z));

		// If it is the first iteration, set initial pose of camera according to accelerometer data (note the different handling for Y axis)
		std::lock_guard<std::mutex> lock(theta_mtx);
		if (first)
		{
			first = false;
			theta = accel_angle;
			// Since we can't infer the angle around Y axis using accelerometer data, we'll use PI as a convetion for the initial pose
			theta.y = PI;
		}
		else
		{
			/*
			Apply Complementary Filter:
				- high-pass filter = theta * alpha:  allows short-duration signals to pass through while filtering out signals
				  that are steady over time, is used to cancel out drift.
				- low-pass filter = accel * (1- alpha): lets through long term changes, filtering out short term fluctuations
			*/
			theta.x = theta.x * alpha + accel_angle.x * (1 - alpha);
			theta.z = theta.z * alpha + accel_angle.z * (1 - alpha);
		}
	}

	// Returns the current rotation angle
	xfloat3 get_theta()
	{
		std::lock_guard<std::mutex> lock(theta_mtx);
		return theta;
	}
};

class rsCameraRecorder
{
public:
	cv::Mat depth;
	cv::Mat rgb;
	cv::Mat rgbNotAligned;
	std::string outputFile;
	int64 timeNow;
	cv::VideoWriter* recorderDepth = NULL;
	cv::VideoWriter* recorderRGB = NULL;
	cv::VideoWriter* recorderRGBNotAligned = NULL;

	std::mutex setM;

	std::thread th;
	std::string serial;

	rsCameraRecorder(std::string sr)
	{
		this->serial = sr;
	}
		   
	bool finishCapture = false;

	void releaseCapture()
	{
		finishCapture = true;
		th.join();
		if (recorderDepth) recorderDepth->release();
		if (recorderRGB) recorderRGB->release();
		if (recorderRGBNotAligned) recorderRGBNotAligned->release();
		
	}

	void startCapturers(bool needsAlign)
	{
		//int wD = this->depth.cols;
		//int hD = this->depth.rows;
		
		//this->recorderDepth = new cv::VideoWriter(this->outputFile + "depth.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), frameRate, cv::Size(wD, hD));
		//std::cout << "Start Depth " << this->serial << " width " << wD << " height " << hD << "\n";
		int wRGB = this->rgb.cols;
		int hRGB = this->rgb.rows;

		if (wRGB == 0)
		{
			wRGB = this->depth.cols;
			hRGB = this->depth.rows;
		}

		if (needsAlign)
		{
			this->recorderRGB = new cv::VideoWriter(this->outputFile + "rgb.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), globalFrameRate, cv::Size(wRGB, hRGB));
			std::cout << "Start RGB " << this->serial << " width " << wRGB << " height " << hRGB << "\n";
		}

		wRGB = this->rgbNotAligned.cols;
		hRGB = this->rgbNotAligned.rows;
		this->recorderRGBNotAligned = new cv::VideoWriter(this->outputFile + "rgbNA.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), globalFrameRate, cv::Size(wRGB, hRGB));
		std::cout << "Start RGB not-aligned " << this->serial << " width " << wRGB << " height " << hRGB << "\n";
	}

	void setStream(std::string ss, cv::Mat& m)
	{
		setM.lock();
		if (ss == "Depth")
		{
			depth = m.clone();
		}
		else
			if (ss == "Color")
			{
				rgb = m.clone();
			}
			else
				if (ss == "ColorNotAligned")
				{
					rgbNotAligned = m.clone();
				}

		setM.unlock();
	}
};


class cameraCapturer
{
protected:
	
	int width, height;
	
	std::vector<std::string> camera_serials;
	rs2::context ctx;
	rs2::config cfg;
	std::map<std::string, rs2::align*> aligners;

	std::map<std::string, rs2::colorizer> colorizers; // Declare map from device serial number to colorizer (utility class to convert depth data RGB colorspace)
	std::map< std::string, rs2::temporal_filter*> temporal_filters;   // Temporal   - reduces temporal noise


public:
	int maxFrames = 1000;
	std::string environment = "dev"; //prod or dev
	bool isRecording = false;
	std::string outputFormat;
	std::string outputDir;
	std::string startRecordingTime;
	
	int frameIndex = 0;
	std::map < std::string, bool> alignment;
	std::map < std::string, cv::Vec3f> accelerometers;
	std::map < std::string, cv::Vec3f> gyroscopes;
	std::map < std::string, float> frameRate;
	std::map < std::string, int64> frameTime;
	std::map < std::string, int> frameAccum;
	std::map<std::string, rsCameraRecorder*>  camerasRec;
	std::map<std::string, cv::Mat>  lastFrames;
	std::map<std::string, rs2::pipeline_profile>  profiles;

	cameraCapturer(int w, int h,int fps, std::string format, std::string dir);
	void generatePipelines(rs2::context& ctx, std::vector<rs2::pipeline*>&   pipelines, bool recordToBag);
	int processBagFile(std::string inputFilename, std::string outputVideoRGB, std::string outputVideoDepth);

	void captureFromRealSenseToMP4( std::string outputDir);

	void captureFromRealSenseToBAG( std::string outputDir);

	void startCapturing();

	void stopCapturing();

	void setRecordingState(bool recording);

	cv::Mat getFrame(std::string cameraID, std::string frameType);

	float getFrameRate(std::string cameraID);

	cv::Vec3f getAccelerometerAngles(std::string cameraID);

	cv::Vec3f getGyroAngles(std::string cameraID);

	void setAutoExposure(std::string cameraID, bool state);

	void setAlignment(std::string cameraID, bool state);

	static cameraCapturer* getInstance();
};

void starCapturing(int width, int height, std::string cameraModel);

void logData(std::string s, int level = 1);
cv::Mat getLastDepthFrame();
cv::Mat getLastRGBFrame();
bool createDirectory(const std::string& name);
bool dirExists(const std::string& dirPath);
std::string return_current_time_and_date();
bool check_imu_is_supported();
void starCapturing(int width, int height, int fps, std::string outputFormat, std::string ouputFile);
void stopCapturing();
void exportCSV(std::string outfilename, std::vector<std::string> values, int time);