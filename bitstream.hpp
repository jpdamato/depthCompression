#include <stdlib.h>
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include<iostream>
#include<fstream>
#include <mutex>
#include <chrono>
#include <omp.h>

#include "zstd.h"
#include "u_ProcessTime.h"

using namespace std::chrono;


#define LOSSLESS_COMPRESSION 0
#define LZ4_COMPRESSION 1
#define ZSTD_COMPRESSION 2
#define SPLINE_COMPRESSION 3
#define LINEAR_COMPRESSION 4

#define MIN_SAMPLES_EQ 10

typedef unsigned short ushort;
typedef unsigned int uint;



static void exportCSV(std::string outfilename, std::vector<float>& values, int time)
{
	std::ofstream myfile(outfilename, std::ios::out | std::ios::app);
	myfile << time;
	for (int i = 0; i < values.size(); i++)
	{
		myfile << ";" << std::to_string( values[i]);
	}
	myfile << "\n";
	myfile.close();
}

static int get_file_size(std::string filename) // path to file
{
	FILE *p_file = NULL;
	p_file = fopen(filename.c_str(), "rb");
	fseek(p_file, 0, SEEK_END);
	int size = ftell(p_file);
	fclose(p_file);
	return size;
}

static uint as_uint(const float x) {
	return *(uint*)&x;
}

static  float as_float(const uint x) {
	return *(float*)&x;
}

static unsigned short as_ushort(char y, char x)
{
	uint16_t result = (((unsigned short)y << 8) & 0xF00) | x;

	return result;
}

static float half_to_float(const ushort x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
	const uint e = (x & 0x7C00) >> 10; // exponent
	const uint m = (x & 0x03FF) << 13; // mantissa
	const uint v = as_uint((float)m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
	return as_float((x & 0x8000) << 16 | (e != 0)*((e + 112) << 23 | m) | ((e == 0)&(m != 0))*((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
}
static ushort float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
	const uint b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
	const uint e = (b & 0x7F800000) >> 23; // exponent
	const uint m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
	return (b & 0x80000000) >> 16 | (e > 112)*((((e - 112) << 10) & 0x7C00) | m >> 13) | ((e < 113)&(e > 101))*((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}
///////////////////////////////////////////////////
////////////////////////////////////////////
static bool fitIt(
		float* x,
		float* y,
		const int &             order,
		float*     coeffs, int nelements)
	{

		size_t N = nelements;

		int n = order;
		int np1 = n + 1;
		int np2 = n + 2;
		int tnp1 = 2 * n + 1;
		float tmp;

		// X = vector that stores values of sigma(xi^2n)
		float X[10];
		for (int i = 0; i < tnp1; ++i) {
			X[i] = 0;
			for (int j = 0; j < N; ++j)
				X[i] += (float)pow(x[j], i);
		}

		// a = vector to store final coefficients.
		//std::vector<float> a(np1);
		float a[10];
		for (int i = 0; i < np1; ++i) a[i] = 0.0;

		// B = normal augmented matrix that stores the equations.
		//std::vector<std::vector<float> > B(np1, std::vector<float>(np2, 0));
		float B[10][10];

		for (int i = 0; i <= n; ++i)
			for (int j = 0; j <= n; ++j)
				B[i][j] = X[i + j];

		// Y = vector to store values of sigma(xi^n * yi)
		float Y[10];
		for (int i = 0; i < np1; ++i) {
			Y[i] = (float)0;
			for (int j = 0; j < N; ++j) {
				Y[i] += (float)pow(x[j], i)*y[j];
			}
		}

		// Load values of Y as last column of B
		for (int i = 0; i <= n; ++i)
			B[i][np1] = Y[i];

		n += 1;
		int nm1 = n - 1;

		// Pivotisation of the B matrix.
		for (int i = 0; i < n; ++i)
			for (int k = i + 1; k < n; ++k)
				if (B[i][i] < B[k][i])
					for (int j = 0; j <= n; ++j) {
						tmp = B[i][j];
						B[i][j] = B[k][j];
						B[k][j] = tmp;
					}

		// Performs the Gaussian elimination.
		// (1) Make all elements below the pivot equals to zero
		//     or eliminate the variable.
		for (int i = 0; i < nm1; ++i)
			for (int k = i + 1; k < n; ++k) {
				float t = B[k][i] / B[i][i];
				for (int j = 0; j <= n; ++j)
					B[k][j] -= t * B[i][j];         // (1)
			}

		// Back substitution.
		// (1) Set the variable as the rhs of last equation
		// (2) Subtract all lhs values except the target coefficient.
		// (3) Divide rhs by coefficient of variable being calculated.
		for (int i = nm1; i >= 0; --i)
		{
			a[i] = B[i][n];                   // (1)
			for (int j = 0; j < n; ++j)
				if (j != i)
					a[i] -= B[i][j] * a[j];       // (2)
			a[i] /= B[i][i];                  // (3)
		}


		for (size_t i = 0; i < np1; ++i)
			coeffs[i] = a[i];

		return true;
};
///////////////////////////////////////////////////
////////////////////////////////////////////
class depthRasterCompression
{
public:
	cv::Mat mt;
	std::vector<std::string> compressionModes = { "LOSSLESS", "LZ4", "ZSTD", "SPLINE" , "LINEAR" };
	std::string compressionName;

	depthRasterCompression()
	{
		compressionName = "OpenCV";
	}

	virtual void createFromImage(cv::Mat&m) 
	{
		this->mt = m.clone();
	}
	virtual   cv::Mat restoreAsImage()
	{
		return mt;
	}

	virtual size_t encode(cv::Mat _m, std::string fn)
	{
		this->mt = _m.clone();
		cv::imwrite(fn, mt);

		return get_file_size(fn);
	}

	virtual size_t saveToFile(std::string fn)
	{
		//using default writer
		cv::imwrite(fn,mt);

		return get_file_size(fn);
	}
	virtual cv::Mat decode(std::string fn)
	{
		mt = cv::imread(fn, -1);
		return mt;
	}

};

///////////////////////////////////////////////////
////////////////////////////////////////////
struct spline_data
{
public:
	unsigned short x0;
	unsigned short y0;	
	unsigned short values_count;
	std::vector<unsigned short> values;
	std::vector<char> cvalues;
	float coefs[4] ;

	int memSize() { return 6 + values.size() ; }

	int last_index;
	int last_value;
	int getValue(int index, int mode)
	{
		if (mode == LOSSLESS_COMPRESSION)
		{
			if (index == last_index + 1)
			{
				last_index++;
				last_value = last_value + cvalues[index];
				return last_value;
			}
			else
			{
				int value = coefs[0];

				for (int i = 0; i <= index; i++) value += cvalues[i];

				last_index = index;
				last_value = value;
				return value;
			}
		}
		else
			if (mode == LINEAR_COMPRESSION || values_count < MIN_SAMPLES_EQ )
			{
				double a = (double)index / values_count;
				int v0 = coefs[0];
				int v1 = coefs[1];
				return (int)(v1 * a + v0 * (1 - a));
			}
			else
				{
					int x = index;
					float v0 = coefs[0] + coefs[1] * x + coefs[2] * x * x + coefs[3] * (x) * (x) * (x);
					return (int)(v0);
				}
	}

	void fit(int mode )
	{
		coefs[0] = coefs[1] = coefs[2] = coefs[3] = 0;

		
		values_count = values.size();
		if (mode == LOSSLESS_COMPRESSION)
		{
			coefs[0] = values[0];
		}
		else
		if (mode == LINEAR_COMPRESSION || values.size() < MIN_SAMPLES_EQ)
		{
			coefs[0] = values[0];
			coefs[1] = values[values_count - 1];
		}
		else		
		{
			std::vector<float> xs;
			std::vector<float> ys;

			int subsample = 2;
			for (int i = 0; i < values.size(); i = i + subsample)
			{
				int value = values[i];
				xs.push_back(i);
				ys.push_back(value);
			}

			fitIt((float*)xs.data(), (float*)ys.data(), 3, coefs,xs.size());
		}
		
	}
};

class splineCompression : public depthRasterCompression
{
public:

	std::vector<spline_data> splines;
	int h, w;
	int quantizationMode = LOSSLESS_COMPRESSION;
	unsigned short* outBuffer = NULL;
	unsigned short* compBuffer = NULL;
	unsigned short* inBuffer = NULL;

	int zstd_compression_level = 1;
	int freePixelsRemoval = 1;
	splineCompression(int m)
	{
		quantizationMode = m;
		compressionName = "splineCompression";
	}

	std::vector<unsigned short> vectorized;

	void vectorizeSplines()
	{
		vectorized.clear();
		for (auto& sp : splines)
		{
			// SIGNAL
			vectorized.push_back(sp.values_count);
			vectorized.push_back(sp.x0);
			vectorized.push_back(sp.y0);
			
	
			if (quantizationMode == LOSSLESS_COMPRESSION)
			{
				vectorized.push_back(sp.coefs[0]);
				for (int i = 0; i < sp.values.size()-1; i= i+2) vectorized.push_back(as_ushort( sp.cvalues[i], sp.cvalues[i+1]));

				if (sp.values_count % 2 == 1) vectorized.push_back(as_ushort(sp.cvalues[sp.values_count-1], 0));

			}
			else
				if (quantizationMode == LINEAR_COMPRESSION)
				{
					vectorized.push_back(sp.coefs[0]);
					vectorized.push_back(sp.coefs[1]);
				}
				else
				{
					if (sp.values_count >= MIN_SAMPLES_EQ)
					{
						vectorized.push_back(sp.coefs[0]);
						vectorized.push_back(float_to_half(sp.coefs[1]));
						vectorized.push_back(float_to_half(sp.coefs[2]));
						vectorized.push_back(float_to_half(sp.coefs[3]));
					}
					else
					{
						vectorized.push_back(sp.coefs[0]);
						vectorized.push_back(sp.coefs[1]);
					}
				}


		}
	}

	virtual void createFromImage(cv::Mat&m)
	{
		this->mt = m.clone();
		std::cout << "----------------------------" << "\n";
		std::cout << "Compression Mode " << compressionModes[ quantizationMode ] << "\n";
		std::cout << "----------------------------" << "\n";

		std::mutex mtx;
		if (!outBuffer) 	outBuffer  = (unsigned short*)malloc(m.cols * m.rows * 2);

		h = m.rows;
		w = m.cols;
		
		std::vector< std::vector<spline_data>> all_splines;

		for (int i = 0; i < 12; i++) all_splines.push_back(std::vector<spline_data>());

		startProcess("createFromImage" + compressionModes[quantizationMode]);
		/// For all pixels
#pragma omp parallel for
		for (int y = 0; y < m.rows; y++)
		{
			int th_ID = omp_get_thread_num();

			for (int x = 0; x < m.cols; x++)
			{
				unsigned short value = m.at<unsigned short>(y, x);
				if (value < 300) continue;

				spline_data p;
				p.x0 = x;
				p.y0 = y;
				p.coefs[0] = value;

				while (true)
				{

					if (x >= m.cols) break;
					unsigned short value2 = m.at<unsigned short>(y, x);
					if (value < 300) break;

					if (abs(value - value2) < 128)
					{
						p.values.push_back(value2);
						p.cvalues.push_back(value2 - value);
						value = value2;
						x++;
					}
					else
					{
						x--;
						break;
					}


				}

				p.fit(quantizationMode);

			
				// discard lonely values
				if (quantizationMode == LOSSLESS_COMPRESSION)
				{
					all_splines[th_ID].push_back(p);
				}
				else
					if (p.values.size() >= freePixelsRemoval)
					{
						all_splines[th_ID].push_back(p);
					}
			}
		}

		endProcess("createFromImage" + compressionModes[quantizationMode]);
		splines.clear();


		for (int i = 0; i < all_splines.size(); i++)
			for (int j = 0; j < all_splines[i].size(); j++)
				splines.push_back(all_splines[i][j]);
		


	}

	virtual size_t encode(cv::Mat _m, std::string fn)
	{
		
		createFromImage(_m);

		unsigned int size = 0;
		startProcess("encode" + compressionModes[quantizationMode]);
		vectorizeSplines();
		/////////////////////////////////////////////////////
	  // Compress using LZ4
		char* srcBuffer = (char*)vectorized.data();
		size_t srcSize = vectorized.size() * 2;

		size_t const cBuffSize = ZSTD_compressBound(srcSize);

		size_t const outSize = ZSTD_compress(outBuffer, cBuffSize, srcBuffer, srcSize, zstd_compression_level);

		double compressRate = (float)(outSize) / (w*h * 2);
		
		endProcess("encode" + compressionModes[quantizationMode]);

		if (fn != "")
		{
			std::ofstream out(fn, std::ios::out | std::ios::binary);
			if (!out)
			{
				std::cout << "Cannot open output file\n";
				return 0;
			}

			// bin mask
			out.write((const char*)outBuffer, outSize);


			out.close();
		}

		return outSize;

	}
	
	virtual cv::Mat restoreAsImage()
	{
		cv::Mat m;

		int indexNZ = 0;
		m.create(h, w, CV_16UC1);
		m.setTo(0);

		int maxLength = 0;
#pragma omp parallel for
		for (int i = 0 ; i<splines.size(); i++)
		{
			spline_data sp = splines[i];
			unsigned short value = 0;

			for (int i = 0; i < sp.values_count; i++)
			{
				unsigned short y = sp.y0;
				unsigned short x = sp.x0 + i;
			
				value = sp.getValue(i,quantizationMode);

				m.at<unsigned short>(y, x) = value;
			}
		}

		std::cout << "mean length " << (double)maxLength / splines.size();
		return m;
	}
	
	virtual size_t saveToFile(std::string fn)	
	{
		unsigned int size = 0;

		vectorizeSplines();
		/////////////////////////////////////////////////////
	  // Compress using LZ4
		char* srcBuffer = (char*)vectorized.data();
		size_t srcSize = vectorized.size() * 2;

		size_t const cBuffSize = ZSTD_compressBound(srcSize);
				
		auto start = high_resolution_clock::now();

		size_t const outSize = ZSTD_compress(outBuffer, cBuffSize, srcBuffer, srcSize, zstd_compression_level);
		auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
		//std::cout << " TIME ZSTD " << duration.count() << "\n";

		double compressRate = (float)(outSize) / (w*h * 2);
		std::cout << "We successfully compressed some data! " << outSize << "Ratio: " << compressRate << "\n";

		if (fn != "")
		{
			std::ofstream out(fn, std::ios::out | std::ios::binary);
			if (!out)
			{
				std::cout << "Cannot open output file\n";
				return 0 ;
			}

			// bin mask
			out.write((const char*)outBuffer, outSize);
			

			out.close();
		}
		
		return get_file_size(fn);


	}

	virtual cv::Mat decode(std::string fn) 	
	{
		h = 768;
		w = 1024;
		size_t outSize = get_file_size(fn);

		/// Prepare DATA
		std::ifstream input(fn, std::ios::binary | std::ios::in);
		if (!input)
		{
			std::cout << "Cannot open input file\n";
			return cv::Mat();
		}
		
		size_t srcSize = 1024 * 768 * 2;
		
		if (!inBuffer)
		{
			inBuffer = (unsigned short*)malloc(srcSize);
			compBuffer = (unsigned short*)malloc(srcSize);
		}

		for (int i = 0; i < srcSize; i++) inBuffer[0];
		for (int i = 0; i < srcSize; i++) compBuffer[0];

		input.read((char*)compBuffer, outSize);

		////////////////////////////
		input.close();

		startProcess("decode" + compressionModes[quantizationMode]);
		// bin mask
		size_t decSz = ZSTD_decompress(inBuffer, srcSize, compBuffer, outSize);

		if (ZSTD_isError(decSz))
		{
			std::cout << ZSTD_getErrorName(decSz) << "\n";

		}

		std::vector<spline_data> readSP;
		
		int index = 0;
		while (index < decSz/2)
		{
			// SIGNAL
			spline_data sp;
			sp.values_count = inBuffer[index]; index++;
			sp.x0 = inBuffer[index]; index++;
			sp.y0 = inBuffer[index]; index++;

			if (quantizationMode == LOSSLESS_COMPRESSION)
			{
				for (int i = 0; i < sp.values_count; i++)
				{
					// BUG
					sp.values.push_back(inBuffer[index]);
					index++;
				}
			}
			else
				if (quantizationMode == LINEAR_COMPRESSION)
				{
					float coef0 = inBuffer[index]; index++;
					float coef1 = inBuffer[index]; index++;
					sp.coefs[0] = coef0;
					sp.coefs[1] = coef1;
					sp.coefs[2] = 0.0;
					sp.coefs[3] = 0.0;
				}
				else
				if (quantizationMode == SPLINE_COMPRESSION)
				{
					if (sp.values_count >= MIN_SAMPLES_EQ)
					{
						float coef0 = inBuffer[index]; index++;
						float coef1 = half_to_float(inBuffer[index]); index++;
						float coef2 = half_to_float( inBuffer[index]); index++;
						float coef3 = half_to_float(inBuffer[index]); index++;

						sp.coefs[0] = coef0;
						sp.coefs[1] = coef1;
						sp.coefs[2] = coef2;
						sp.coefs[3] = coef3;
					}
					else
					{
						float coef0 = inBuffer[index]; index++;
						float coef1 = inBuffer[index]; index++;

						sp.coefs[0] = coef0;
						sp.coefs[1] = coef1;
						sp.coefs[2] = 0;
						sp.coefs[3] = 0;
					}
				}

			readSP.push_back(sp);

		}

		this->splines.clear();
		splines.swap(readSP);
	
		cv::Mat _m = restoreAsImage();
		
		endProcess("decode" + compressionModes[quantizationMode]);
		return _m;
	}

	~splineCompression()
	{
		free(inBuffer);
		free(compBuffer);
	}


	void display(int row)
	{
		cv::Mat histImg;
		histImg.create(cv::Size(w, 200), CV_8UC3);

		histImg.setTo(0);
		cv::RNG rng(12345);

		int maxH = 18000;
		int hist_h = 200;
		for (auto& sp : splines)
		{
			if (sp.y0 != row) continue;
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

			if (sp.values.size() > 0)
			{
				for (int i = 0; i < sp.values.size(); i = i + 1)
				{

					int value = sp.getValue(i, 0);
					double y = (double)(value) / (maxH)*histImg.rows;
					int x = sp.x0 + i;


					/// Render Histogram
					cv::line(histImg, cv::Point(x, hist_h), cv::Point(x, hist_h - y), color, 1, 8, 0);
				}
			}
			else
			{
				for (int i = 0; i < sp.values_count; i = i + 1)
				{
					int x = sp.x0 + i;
					int y = sp.y0;

					double value = mt.at<unsigned short>(y,x);

					double z = (double)(value) / (maxH)*histImg.rows;
				
					/// Render Histogram
					line(histImg, cv::Point(x, hist_h), cv::Point(x, hist_h - z), color, 1, 8, 0);
				}
			}

			if (quantizationMode == SPLINE_COMPRESSION)
			{
				
				for (int i = 0; i < sp.values_count; i = i + 1)
				{
					int x = sp.x0 + i;

					double v0 = sp.getValue(i, quantizationMode);
					double v1 = sp.getValue(i+1, quantizationMode);

					double y0 = (double)(v0) / (maxH)*histImg.rows;
					double y1 = (double)(v1) / (maxH)*histImg.rows;

					/// Render Histogram
					line(histImg, cv::Point(x, hist_h-y0), cv::Point(x+1, hist_h - y1), cv::Scalar(0,255,255), 1, 8, 0);
				}
				

			}
			else
			{
				// Render estimation
				int x0 = sp.x0;
				int x1 = x0 + sp.values_count;
				int y0 = (double)sp.getValue(0, LINEAR_COMPRESSION) / (maxH)*histImg.rows;
				int y1 = (double)sp.getValue(sp.values_count - 1, LINEAR_COMPRESSION) / (maxH)*histImg.rows;
				line(histImg, cv::Point(x0, hist_h - y0), cv::Point(x1, hist_h - y1), cv::Scalar(0, 255, 0), 1, 8, 0);
			}
			

		}

		cv::imshow("histo"+ compressionModes[quantizationMode], histImg);
		//std::cout << "End HistDisplay" << std::endl;
	}
};
