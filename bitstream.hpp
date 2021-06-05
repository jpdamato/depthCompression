#include <stdlib.h>
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include<iostream>
#include<fstream>
#include <mutex>
#include <chrono>

#include "zstd.h"



#include "intSeqCompression.hpp"

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
template <class TYPE>
class PolynomialRegression {
public:

	PolynomialRegression();
	virtual ~PolynomialRegression() {};

	bool fitIt(
		float* x,
		float* y,
		const int &             order,
		float*     coeffs, int nelements);
};

template <class TYPE>
PolynomialRegression<TYPE>::PolynomialRegression() {};

template <class TYPE>
bool PolynomialRegression<TYPE>::fitIt(
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
	TYPE tmp;

	// X = vector that stores values of sigma(xi^2n)
	std::vector<TYPE> X(tnp1);
	for (int i = 0; i < tnp1; ++i) {
		X[i] = 0;
		for (int j = 0; j < N; ++j)
			X[i] += (TYPE)pow(x[j], i);
	}

	// a = vector to store final coefficients.
	std::vector<TYPE> a(np1);

	// B = normal augmented matrix that stores the equations.
	std::vector<std::vector<TYPE> > B(np1, std::vector<TYPE>(np2, 0));

	for (int i = 0; i <= n; ++i)
		for (int j = 0; j <= n; ++j)
			B[i][j] = X[i + j];

	// Y = vector to store values of sigma(xi^n * yi)
	std::vector<TYPE> Y(np1);
	for (int i = 0; i < np1; ++i) {
		Y[i] = (TYPE)0;
		for (int j = 0; j < N; ++j) {
			Y[i] += (TYPE)pow(x[j], i)*y[j];
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
			TYPE t = B[k][i] / B[i][i];
			for (int j = 0; j <= n; ++j)
				B[k][j] -= t * B[i][j];         // (1)
		}

	// Back substitution.
	// (1) Set the variable as the rhs of last equation
	// (2) Subtract all lhs values except the target coefficient.
	// (3) Divide rhs by coefficient of variable being calculated.
	for (int i = nm1; i >= 0; --i) {
		a[i] = B[i][n];                   // (1)
		for (int j = 0; j < n; ++j)
			if (j != i)
				a[i] -= B[i][j] * a[j];       // (2)
		a[i] /= B[i][i];                  // (3)
	}

	
	for (size_t i = 0; i < a.size(); ++i)
		coeffs[i] = a[i];

	return true;
}

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
	virtual size_t saveToFile(std::string fn)
	{
		//using default writer
		cv::imwrite(fn,mt);

		return get_file_size(fn);
	}
	virtual cv::Mat readFromFile(std::string fn)
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

			PolynomialRegression<float> pr;
			std::vector<float> cf;
			pr.fitIt((float*)xs.data(), (float*)ys.data(), 3, coefs,xs.size());
		}
		
	}
};

class splineCompression : public depthRasterCompression
{
public:

	std::vector<spline_data> splines;
	int h, w;
	int quantizationMode = LOSSLESS_COMPRESSION;
	unsigned short* outBuffer;

	int compression_level = 3;
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

		outBuffer  = (unsigned short*)malloc(m.cols * m.rows * 2);

		h = m.rows;
		w = m.cols;

		splines.clear();
		/// For all pixels
#pragma omp parallel for
		for (int y = 0; y < m.rows; y++)
		{
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

				mtx.lock();
				// discard lonely values
				if (quantizationMode == LOSSLESS_COMPRESSION)
				{
					splines.push_back(p);
				}
				else
					if (p.values.size() > 1)
					{
						splines.push_back(p);
					}
				mtx.unlock();
			}
		}

	}
	
	virtual cv::Mat restoreAsImage()
	{
		cv::Mat m;

		int indexNZ = 0;
		m.create(h, w, CV_16UC1);
		m.setTo(0);


		int maxLength = 0;
		for (auto& sp : splines)
		{
			unsigned short value = 0;

			maxLength += sp.values.size();
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

		size_t const outSize = ZSTD_compress(outBuffer, cBuffSize, srcBuffer, srcSize, compression_level);
		auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
		std::cout << " TIME ZSTD " << duration.count() << "\n";

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

	virtual cv::Mat readFromFile(std::string fn) 	
	{

		size_t outSize = get_file_size(fn);

		std::ifstream input(fn, std::ios::binary | std::ios::in);
		if (!input)
		{
			std::cout << "Cannot open input file\n";
			return cv::Mat();
		}
		
		unsigned short* compBuffer, *inBuffer ;

		size_t srcSize = 1024 * 768 * 2;
		

		inBuffer = (unsigned short*)malloc(srcSize);		
		compBuffer = (unsigned short*)malloc(outSize);

		for (int i = 0; i < srcSize; i++) inBuffer[0];
		for (int i = 0; i < outSize; i++) compBuffer[0];

		input.read((char*)compBuffer, outSize);

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
		////////////////////////////
		input.close();

		free(inBuffer);
		free(compBuffer);

		return restoreAsImage();
	}


	double compressionRate()
	{
		unsigned int size = 0;

		vectorizeSplines();

		/////////////////////////////////////////////////////
	  // Compress using LZ4
		char* srcBuffer = (char*)vectorized.data();
		size_t srcSize = vectorized.size() * 2;

		size_t const cBuffSize = ZSTD_compressBound(srcSize);

		char* outBuffer = (char*)malloc(srcSize);
		size_t const outSize = ZSTD_compress(outBuffer, cBuffSize, srcBuffer, srcSize, compression_level);

		double compressRate = (float)(outSize ) / (w*h * 2);
		std::cout << "We successfully compressed some data! " << outSize  << "Ratio: " << compressRate << "\n";

		return compressRate;
		
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

///////////////////////////////////////////////////
////////////////////////////////////////////
class bitStream : public depthRasterCompression 
{
private:
  char *data; // where bitstream is contained
  std::vector < std::vector<short>> nonZeroValuesRows;
  int size;   // size of the bitstream

  int w, h;
public:
	int algorithm = ZSTD_COMPRESSION;
	int _minThresold = 300;
	int _maxThresold = 25000;

	int pixelsError = 0;
	int quantization = 1;

	cv::Mat origImage;
	cv::Mat computedImage;

	std::vector<short> nonZeroValues;
	std::vector<short> resampled;


	bitStream() 
	{
		compressionName = "ZSTD";
	}

  // returns data as a byte array
  char *toByteArray() {
    return data;
  }
  /**************************/
  bool getbit(char x, int y) {
    return (x >> (7 - y)) & 1;
  }
  int chbit(int x, int i, bool v) {
    if(v) return x | (1 << (7 - i));
    return x & ~(1 << (7 - i));
  }
  /**************************/

  // opens an existing byte array as bitstream
  void openBytes(char *bytes, int _size) {
    data = bytes;
    size = _size;
  }
  // creates a new bit stream
  void open(int _size) {

	if (!data)  data = (char*)malloc(_size);
	else
	{
		for (int i = 0; i < _size; i++) data[i] = 0;
	}

	nonZeroValues.clear();
    size = _size;
  }
  // closes bit stream (frees memory)
  void close() {
    free(data);
  }


  bool startSequence()
  {

  }

  // convert from image
  void createFromImage(cv::Mat&m)
  {
	  int size = m.cols * m.rows;
	  this->open(size);
	  this->w = m.cols;
	  this->h = m.rows;

	  origImage = m.clone();
	
	  pixelsError = 0;

	  nonZeroValuesRows.resize(h);

	  for (int x = 0; x < w; x++)
		  for (int y = 0; y < h; y++)
		  {
			  unsigned short befValue;
			  if (x == 0) befValue = 0;
			  else befValue = m.at<unsigned short>(y, x - 1);

			  unsigned short value = m.at<unsigned short>(y, x);
			  int index = y * w + x;

			  if (value < _minThresold) 
			  {
				  m.at<unsigned short>(y, x) = 0;
				  value = 0; 
				  pixelsError++;
			  }

			  if (value > _maxThresold)
			  {
				  value = 0;
				  m.at<unsigned short>(y, x) = 0;
				  pixelsError++;
			  }
			  
			  data[index / 8] = chbit(data[index / 8], index % 8, value == 0);
			 // save non zero
			  if (value > 0)
			  {
				  nonZeroValues.push_back(value);
				  nonZeroValuesRows[y].push_back(value);
			  }
			  // if next
		  }
	  
	  int32Encoder::encode(nonZeroValues, resampled, quantization);
	 
  }

  cv::Mat getBackGroundAsImage()
  {
	  cv::Mat m;

	  m.create(h, w, CV_8UC3);
	  for (int x = 0; x < w; x++)
		  for (int y = 0; y < h; y++)
		  {
			  int index = y * w + x;
			  bool value = getbit(data[index / 8], index % 8);
			  if (value)
				  m.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			  else
				  m.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
		  }

	  return m;
	  
  }

  cv::Mat restoreAsImage()
  {
	  cv::Mat m;

	  int indexNZ = 0;
	  m.create(h, w, CV_16UC1);
	  m.setTo(0);

	  nonZeroValues.clear();
	  int32Encoder::decode(resampled, nonZeroValues, quantization);
	

	  for (int x = 0; x < w; x++)
		  for (int y = 0; y < h; y++)
		  {
			  int index = y * w + x;

			  unsigned short befValue;
			  if (x == 0) befValue = 0;
			  else befValue = origImage.at<unsigned short>(y, x - 1);

			 
			  bool value = getbit(data[index / 8], index % 8);
			  //es background
			  if (value)
			  {
				//  m.at<unsigned short>(y, x) = 0;
			  }
			  else
			  {
				  if (indexNZ < nonZeroValues.size())
				  {
					  m.at<unsigned short>(y, x) = nonZeroValues[indexNZ];
					  
					  indexNZ++;
				  }
			  }
		  }


	  computedImage = m.clone();

		  
	  return m;

  }

  int error()
  {
	  int accum = 0;
	  for (int x = 0; x < w; x++)
		  for (int y = 0; y < h; y++)
		  {
			  unsigned short v0 = origImage.at<unsigned short>(y, x);
			  unsigned short v1 = computedImage.at<unsigned short>(y, x);

			  if (v0 == 0 && v1 == 0) continue;
			  
			  if (v0 < _minThresold)continue;
			  if (v0 > _maxThresold)continue;
			  
			  accum += (v0 - v1);
		  }

	  return accum;
  }
  
  
  double compressionRate()
  {
	  return saveToFileZSTD("");
  }
  ////////////////////////////////////////////////////////////
  // Save using ZTSTD
  double saveToFileZSTD(std::string filename)
  {
	  
	  /////////////////////////////////////////////////////
	  // Compress using LZ4
	  char* srcBuffer = (char*)resampled.data();
	  size_t srcSize = resampled.size() * 2;

	  char* srcBuffer2 = (char*)data;
	  size_t srcSize2 = size / 8;

	  size_t const cBuffSize = ZSTD_compressBound(srcSize);
	  size_t const cBuffSize2 = ZSTD_compressBound(srcSize2);

	  char* outBuffer = (char*)malloc(srcSize);
	  char* outBuffer2 = (char*)malloc(srcSize2);
	  char* testBuffer = (char*)malloc(srcSize);
		  
	  /* Compress.
	 * If you are doing many compressions, you may want to reuse the context.
	 * See the multiple_simple_compression.c example.
	 */
	  size_t const outSize = ZSTD_compress(outBuffer, cBuffSize, srcBuffer, srcSize, 9);
	  size_t const outSize2 = ZSTD_compress(outBuffer2, cBuffSize2, srcBuffer2, srcSize2, 9);

	  double compressRate = (float)(outSize + outSize2) / (w*h*2);
	  std::cout << "We successfully compressed some data! " << outSize+ outSize2 << "Ratio: " << compressRate << "\n";
	 
	  //////////////

	  ZSTD_decompress(testBuffer, srcSize, outBuffer, outSize);

	  if (filename != "")
	  {
		  std::ofstream out(filename, std::ios::out | std::ios::binary);
		  if (!out)
		  {
			  std::cout << "Cannot open output file\n";
			  return 0;
		  }

		  // bin mask
		  out.write((const char*)outBuffer, outSize);
		  // non zero values
		  out.write((const char*)outBuffer2, outSize2);

		  out.close();
	  }
	

	  /* Validation */
	  if (memcmp(srcBuffer, testBuffer, srcSize) != 0)
		  std::cout << "Validation failed.  *src and *new_src are not identical." << "\n";

	  free(outBuffer);   /* no longer useful */
	  free(outBuffer2);   /* no longer useful */
	  free(testBuffer);   /* no longer useful */

	

	  return compressRate;
  }

  /////////////////////////////////////////////
  // SAve to File using LZ4
#ifdef LZ4
  void saveToFileLZ4(std::string filename)
  {
	  std::ofstream out(filename, std::ios::out | std::ios::binary);
	  if (!out)
	  {
		  std::cout << "Cannot open output file\n";
		  return ;
	  }

	  /////////////////////////////////////////////////////
	  // Compress using LZ4
	  char* src = (char*)nonZeroValues.data();
	  int src_size = nonZeroValues.size() * 2;
	  
	  // LZ4 provides a function that will tell you the maximum size of compressed output based on input data via LZ4_compressBound().
	  const int max_dst_size = LZ4_compressBound(src_size);
	  // We will use that size for our destination boundary when allocating space.
	  char* compressed_data = (char*)malloc((size_t)max_dst_size);
	 
	  int compressed_data_size = LZ4_compress_default(src, compressed_data, src_size, max_dst_size);
	  if (compressed_data_size > 0)
		  std::cout << "We successfully compressed some data! " << compressed_data_size << "Ratio: " << (float)compressed_data_size / src_size << "\n";
	 
	  // Not only does a positive return_value mean success, the value returned == the number of bytes required.
	  // You can use this to realloc() *compress_data to free up memory, if desired.  We'll do so just to demonstrate the concept.
	  compressed_data = (char *)realloc(compressed_data, (size_t)compressed_data_size);
	  if (compressed_data == NULL)
		  std::cout << "Failed to re-alloc memory for compressed_data.  Sad :(" <<"\n";

	  char* src2 = data;
	  int src_size2 = size / 8;

	  const int max_dst_size2 = LZ4_compressBound(src_size2);
	  // We will use that size for our destination boundary when allocating space.
	  char* compressed_data2 = (char*)malloc((size_t)max_dst_size2);

	  int compressed_data_size2 = LZ4_compress_default(src2, compressed_data2, src_size, max_dst_size2);
	  if (compressed_data_size2 > 0)
		  std::cout << "We successfully compressed some data! " << compressed_data_size2 << "Ratio: " << (float)compressed_data_size2 / src_size2 << "\n";

	  // Not only does a positive return_value mean success, the value returned == the number of bytes required.
	  // You can use this to realloc() *compress_data to free up memory, if desired.  We'll do so just to demonstrate the concept.
	  compressed_data2 = (char *)realloc(compressed_data2, (size_t)compressed_data_size2);
	  

	  // bin mask
	  out.write((const char*)compressed_data2, compressed_data_size2);
	  // non zero values
	  out.write((const char*)compressed_data, compressed_data_size);

	  free(compressed_data);   /* no longer useful */
	  free(compressed_data2);   /* no longer useful */

	  out.close();
  }

#endif


  virtual size_t saveToFile(std::string filename)
  {
	  saveToFileZSTD(filename);


	  return get_file_size(filename);
  }

  cv::Mat readFromFile(std::string filename)
  {
	  std::ifstream input(filename, std::ios::binary);
	  if (!input)
	  {
		  std::cout << "Cannot open input file\n";
		  return cv::Mat();
	  }
	  int nonZeroValuesCount = nonZeroValues.size() ;
	  nonZeroValues.clear();

	  for (int i = 0; i < nonZeroValuesCount; i++) nonZeroValues.push_back(0);

	  input.read(data, size / 8);
		// bin mask
	  input.read((char*)nonZeroValues.data(), nonZeroValuesCount*2);
	
	  input.close();

	  return restoreAsImage();
  }


  // writes to bitstream
  void write(int ind, int bits, int dat) {
    ind += bits;
    while(dat) {
      data[ind / 8] = chbit(data[ind / 8], ind % 8, dat & 1);
      dat /= 2;
      ind--;
    }
  }
  // reads from bitstream
  int read(int ind, int bits) {
    int dat = 0;
    for(int i = ind; i < ind + bits; i++) {
      dat = dat * 2 + getbit(data[i / 8], i % 8);
    }
    return dat;
  }
};
