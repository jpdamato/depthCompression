
#include<iostream>
#include<fstream>
#include <mutex>
#include <chrono>
#include <omp.h>

#include <opencv2/opencv.hpp>   // Include OpenCV API

#ifdef ZSTD
#include "zstd.h"
#endif
#include "u_ProcessTime.h"


using namespace std::chrono;


#define LOSSLESS_COMPRESSION 0
#define LZ4_COMPRESSION 1
#define ZSTD_COMPRESSION 2
#define SPLINE_COMPRESSION 3
#define LINEAR_COMPRESSION 4
#define SPLINE5_COMPRESSION 5
#define MIN_EXPECTED_ERROR 300

#define MIN_SAMPLES_EQ 24


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
		float X[20];
		for (int i = 0; i < tnp1; ++i) {
			X[i] = 0;
			for (int j = 0; j < N; ++j)
				X[i] += (float)pow(x[j], i);
		}

		// a = vector to store final coefficients.
		//std::vector<float> a(np1);
		float a[20];
		for (int i = 0; i < np1; ++i) a[i] = 0.0;

		// B = normal augmented matrix that stores the equations.
		//std::vector<std::vector<float> > B(np1, std::vector<float>(np2, 0));
		float B[20][20];

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
	std::vector<std::string> compressionModes = { "LOSSLESS", "LZ4", "ZSTD", "SPLINE" , "LINEAR", "FIVE" };
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

	virtual void computeMetrics()
	{

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
class spline_data
{
public:
	bool valid = 0;
	unsigned short x0;
	unsigned short y0;	
	bool visited = false;
	unsigned short values_count;
	std::vector<unsigned short> values;
	std::vector<char> cvalues;
	std::vector<char> residual;
	float coefs[6] ;
	double _error = 0;


	int memSize() { return 6 + values.size() ; }

	int last_index;
	int last_value;
	int getValue(int index, int mode)
	{
		if (mode == LOSSLESS_COMPRESSION)
		{

			if (values.size() > index) return values[index];

			// Progressive
			if (index == last_index + 1)
			{
				last_index++;
				last_value = last_value + cvalues[index] ;
				return last_value;
			}
			else
			{
				int value = coefs[0];

				for (int i = 0; i <= index; i++) value += cvalues[i] ;

				last_index = index;
				last_value = value;
				return value;
			}
		}
		else
			if (mode == LINEAR_COMPRESSION || values_count < MIN_SAMPLES_EQ )
			{
				int x = index;
				float v0 = coefs[0] + coefs[1] * x + coefs[2] * x * x + coefs[3] * (x) * (x) * (x);
				
				return (int)(v0);
			}
			else
				if (mode == SPLINE_COMPRESSION)
				{
					int x = index;
					float v0 = coefs[0] + coefs[1] * x + coefs[2] * x * x + coefs[3] * (x) * (x) * (x);
					return (int)(v0);
				}
				else
					if (mode == SPLINE5_COMPRESSION)
					{
						int x = index;
						float v0 = coefs[0] + coefs[1] * x + coefs[2] * x * x + coefs[3] * (x) * (x) * (x)+coefs[4] * x * x * x * x + coefs[5] * x * x * x * x * x;;
						return (int)(v0);
					}


	}

	

	double evaluate(int xv)
	{
		xv = xv - x0;
		return  coefs[0] + coefs[1] * xv + coefs[2] * xv * xv + coefs[3] * xv * xv * xv;
	}

	void fit(int mode, int quantization )
	{
		coefs[0] = coefs[1] = coefs[2] = coefs[3] = coefs[4] = coefs[5] = 0;

		values_count = values.size();
		if (values.size() == 0) return;

		if (mode == LOSSLESS_COMPRESSION)
		{
			coefs[0] = values[0];
		}
		else
		if (mode == LINEAR_COMPRESSION || values.size() < MIN_SAMPLES_EQ)
		{

			coefs[0] = values[0];

			coefs[1] = (float)(1.0f*values[values_count-1] - values[0])/ values_count;

			
		}
		else		
		{
			std::vector<float> xs;
			std::vector<float> ys;

			int subsample = 1;
			if (values.size() > 100) subsample = 4;
			else if (values.size() > 20) subsample = 2;

			for (int i = 0; i < values.size(); i = i + subsample)
			{
				int value = values[i];
				xs.push_back(i);
				ys.push_back(value);
			}

			if (mode == SPLINE_COMPRESSION) fitIt((float*)xs.data(), (float*)ys.data(), 3, coefs,xs.size());
			else fitIt((float*)xs.data(), (float*)ys.data(), 5, coefs, xs.size());
		}

		computeResidual(mode, quantization);
		
	}

	void computeResidual(int mode,int QUANTIZATION)
	{
		_error = 0;
		values_count = values.size();
		//residual.clear();
		if (values.size() > 0)
		{
			
			residual.clear();

		
			////////////////////////////////////////////
			for (int i = 0; i < values_count; i++)
			{
				//cvalues.push_back( (values[i] - values[i-1]) / QUANTIZATION);

				short diff = (values[i] - getValue(i, mode));

				residual.push_back(diff / QUANTIZATION);

				_error += abs(diff / QUANTIZATION);
				//residual.push_back(evaluate(i) - values[i]);
			}
		}

		_error = _error / values_count;
	}

	double error()
	{
		return _error;
	}

	
};


///////////////////////////////////////////////////
////////////////////////////////////////////

class memManager
{
public:
	std::vector<spline_data*> available;
	std::vector<spline_data*> allocated;
	std::mutex mtx;
	spline_data* getNew()
	{
		if (available.size() == 0)
		{
			spline_data* sp = new spline_data();
			sp->values_count = 0;
			mtx.lock();
			allocated.push_back(sp);
			mtx.unlock();
			return sp;
		}
		else
		{
			mtx.lock();
			spline_data* sp = available[available.size()-1];
			sp->values_count = 0;
			sp->valid = 1;
			sp->visited = false;
			sp->values.clear();
			sp->cvalues.clear();
			available.pop_back();
			allocated.push_back(sp);
			mtx.unlock();
			return sp;

		}
	}

	void releaseAll()
	{
		for (auto sp : allocated)
		{
			available.push_back(sp);
			sp->values_count = 0;
		}

		allocated.clear();
	}

	void init(int count)
	{
		for (int i = 0; i < count; i++)
		{
			available.push_back(new spline_data());
		}
	}
};


class splineCompression : public depthRasterCompression
{
public:

	memManager memMgr;
	std::vector<spline_data*> splines;
	int h, w;
	unsigned short encodedMode = LOSSLESS_COMPRESSION;
	unsigned short* outBuffer = NULL;
	unsigned short* compBuffer = NULL;
	unsigned short* inBuffer = NULL;
	unsigned short* vectorized = NULL;
	int vectorized_count = 0;
	int quantization = 16;
	bool improveMethod = false;

	int scale = 1;

	int offset = 0;

	// ZIP  Compression Level 1 = low .. 9 = high
	int zstd_compression_level = 9;
	// Accept this amount of continuos pixels
	int lonelyPixelsRemoval = 1;
	// Retry if a pixel failed
	int checkNeighRetry = 1;

	bool saveResidual = false;
#ifdef ZSTD
	ZSTD_CCtx* ctx = NULL;
#endif
	// Constructor
	splineCompression(int m)
	{
		encodedMode = m;
		compressionName = "splineCompression";
		memMgr.init(100000);
#ifdef ZSTD
		ctx = ZSTD_createCCtx();
#endif
	}

	
	///////////////////////////////
	void vectorizeSplines()
	{
		allocateMem();
		vectorized_count = 0;
		

		for (auto& sp : splines)
		{
			// SIGNAL
			vectorized[vectorized_count] = sp->values_count; vectorized_count++;
			vectorized[vectorized_count] = sp->x0; vectorized_count++;
			vectorized[vectorized_count] = sp->y0; vectorized_count++;
		
			if (encodedMode == LOSSLESS_COMPRESSION)
			{
				vectorized[vectorized_count]  = sp->coefs[0] ; vectorized_count++;
				if (sp->values_count == 1)
				{
					/////
				}
				else
				{
					for (int i = 0; i < sp->values_count - 1; i = i + 2)
					{
						vectorized[vectorized_count] = as_ushort(sp->cvalues[i]/scale, sp->cvalues[i + 1] / scale) ; vectorized_count++;
					}

					if (sp->values_count % 2 == 1)
					{
						vectorized[vectorized_count] = as_ushort(sp->cvalues[sp->values_count - 1] / scale, 0);  vectorized_count++;
					}
				}

			}
			else
				if (encodedMode == LINEAR_COMPRESSION)
				{
					vectorized[vectorized_count] = sp->coefs[0]; vectorized_count++;
					vectorized[vectorized_count] = float_to_half(sp->coefs[1]); vectorized_count++;
				}
				else
				{
					if (sp->values_count < MIN_SAMPLES_EQ)
					{
						vectorized[vectorized_count] = sp->coefs[0]; vectorized_count++;
						vectorized[vectorized_count] = float_to_half(sp->coefs[1]); vectorized_count++;
					}
					else
					if (encodedMode == SPLINE_COMPRESSION)
					{
						vectorized[vectorized_count] = sp->coefs[0]; vectorized_count++;
						vectorized[vectorized_count] = float_to_half(sp->coefs[1]); vectorized_count++;
						vectorized[vectorized_count] = float_to_half(sp->coefs[2]); vectorized_count++;
						vectorized[vectorized_count] = float_to_half(sp->coefs[3]); vectorized_count++;
					}
					else
					{
						vectorized[vectorized_count] = sp->coefs[0]; vectorized_count++;
						vectorized[vectorized_count] = float_to_half(sp->coefs[1]); vectorized_count++;
						vectorized[vectorized_count] = float_to_half(sp->coefs[2]); vectorized_count++;
						vectorized[vectorized_count] = float_to_half(sp->coefs[3]); vectorized_count++;
						vectorized[vectorized_count] = float_to_half(sp->coefs[4]*1000); vectorized_count++;
						vectorized[vectorized_count] = float_to_half(sp->coefs[5]*10000); vectorized_count++;
					}
				}


			if (saveResidual && sp->error() > 0)
			{
				for (int i = 0; i < sp->values_count - 1; i = i + 2)
				{
					vectorized[vectorized_count] = as_ushort(sp->residual[i], sp->residual[i + 1]); vectorized_count++;
				}
			}
		}
	}

	void allocateMem()
	{
		if (!outBuffer) 	outBuffer = new unsigned short[w * h * 2];

		if (!inBuffer)
		{
			inBuffer = new unsigned short[w * h * 2];
			compBuffer = new unsigned short[w * h * 2];
		}

		if (!vectorized) vectorized = new unsigned short[w * h * 2];

		// clear buffers
#pragma omp parallel for
		for (int i = 0; i < h * w * 2; i++) 
		{ 
			inBuffer[i] = 0;  
			compBuffer[i] = 0;  
			outBuffer[i] = 0;		
			vectorized[i] = 0;
		}


	}

	
	double similitud(spline_data* p0, spline_data* p1, int mode)
	{
		int dif = p0->x0 + p0->values_count - p1->x0;

		if (dif > 3) return 10.0;

		double d = p0->getValue(p1->x0 - p0->x0, mode) - p1->getValue(0,mode);

		return d / 10.0;
	}

	void merge(spline_data* p0, spline_data* p1)
	{
		if (p0->values_count == 0) return;
		if (p1->values_count == 0) return;

		// fill holes
		int dif = p0->x0 + p0->values_count - p1->x0;
		
		for (int i = 0; i < dif; i++) p0->values.push_back(p0->values[p0->values_count -1]);

		for (int i = 0; i < p1->values_count; i++) p0->values.push_back(p1->values[i]);

		p0->values_count = p0->values.size();

		p0->fit(encodedMode, quantization);
		p1->valid = 0;
		p1->values.clear();
		p1->values_count = 0;


	}

	bool split(spline_data* p, spline_data* p0, spline_data* p1, int mode)
	{
		p->valid = 1;
		p0->valid = 0;
		p1->valid = 0;

		int MAX_DIF = 200;

		int step = 2;
		int maxG = 0;
		// find maxGradient
		double worstError = 1000000;
		// add values while error is low
		int counter = 1;
		int best_middle = 0;
		
		for (int middle = 2; middle < p->values.size() - 10; middle = middle + 2)
		{
			p0->values.clear();
			p1->values.clear();
		
			for (int i = 0; i < p->values.size(); i = i + 1)
			{
				unsigned short v0 = p->values[i]; // orig value
				if (i < middle) { p0->values.push_back(v0); }
				else { p1->values.push_back(v0); }

			}
			p1->fit(mode, quantization);
			p0->fit(mode, quantization);

			//double errorP = p->error();
			double error0 = p0->error();
			double error1 = p1->error();


			if ((error0 * 0.5 + error1 * 0.5 + 4000/ middle) < worstError)
			{
				worstError = error0 * 0.5 + error1 * 0.5 + 4000 / middle; // MAX(error0, error1);
				best_middle = middle;
			}

		}


		if (worstError < p->error() )
			{

			p0->values.clear();
			p1->values.clear();

			for (int i = 0; i < p->values.size(); i = i + 1)
			{
				unsigned short v0 = p->values[i]; // orig value
				if (i < best_middle) { p0->values.push_back(v0); }
				else { p1->values.push_back(v0); }

			}

			p1->fit(mode, quantization);
			p0->fit(mode, quantization);

				p0->x0 = p->x0;
				p0->y0 = p->y0;
				p1->x0 = p->x0 + p0->values_count;
				p1->y0 = p->y0;

				p->valid = 0;
				p0->valid = 1;
				p1->valid = 1;
				p0->visited = true; // not process again

				return true;
			}
		
		return false;
	}

	// Make proposal then split
	std::vector<spline_data*> encodeRowIter(int y, int cols, unsigned short* pixels, int mode, int itercount)
	{
		// Take a first approach
		std::vector<spline_data*> ps;

	//	if (y != 250) return ps;
		int window = 5;
		/////////////////////////////////
		for (int x = 0; x < cols; x++)
		{
			unsigned short value = pixels[y * w + x];
			if (value < 300) continue;

			spline_data* p = memMgr.getNew();
			p->x0 = x;
			p->y0 = y;
			p->coefs[0] = value;
			
			// check consecutive pixels
			while (true)
			{
				if (x >= cols) break;
				// take a sample
				unsigned short value2 = pixels[y * w + x];
				
				// may be it is noise
				if (value2 < 300)
				{
					break;
				}
				
				p->values.push_back(value2);
				p->values_count = p->values.size();
				x++;
			}

			p->fit(mode, quantization);

			if (p->values_count >= lonelyPixelsRemoval)
				{
					p->valid = 1;
					ps.push_back(p);
				}

		}
		/// split planes
		if (mode != LOSSLESS_COMPRESSION)
		{
			for (int iter = 0; iter < itercount; iter++)
			{
				std::vector<spline_data*> newps;

				for (auto p : ps) p->visited = false;

				for (int i = 0 ; i<ps.size() ; i++)
				{
					spline_data* p = ps[i];
					if (!p->valid) continue;
					if (p->visited) continue;

					if (p->error() < MIN_EXPECTED_ERROR)
					{
						newps.push_back(p);
						continue;
					}

					if (p->values_count >  MIN_SAMPLES_EQ)
					{
						spline_data* p0 = memMgr.getNew();
						spline_data* p1 = memMgr.getNew();

						if (split(p, p0, p1, mode))
						{
							newps.push_back(p0);
							newps.push_back(p1);
						}
						else
						{
							newps.push_back(p);
						}
					}
					else
					{
						newps.push_back(p);
					}

				}

				ps.clear();
				ps.swap(newps);
			}
		}
		return ps;
	
	}

	// Generate a first approach
	std::vector<spline_data*> encodeRow(int y, int cols, unsigned short* pixels)
	{
		std::vector<spline_data*> ps;

		int window = 5;

		for (int x = 0; x < cols; x++)
		{
			unsigned short value = pixels[y * w + x];
			if (value < 300) continue;

			spline_data* p = memMgr.getNew();
			p->x0 = x;
			p->y0 = y;
			p->coefs[0] = value;
			int ret = checkNeighRetry;
			// check consecutive pixels
			while (true)
			{
				if (x >= cols) break;
				// take a sample
				unsigned short value2 = pixels[y * w + x];
				unsigned short value3 = pixels[y * w + x + 1];
				// may be it is noise
				if (value2 < 300)
				{
					break;
				}

				if (abs(value - value2) < 128  || abs(value - value3) < 128 )
				{
					p->values.push_back(value2);
					value = value2;
					p->values_count = p->values.size();
					x++;
				}
				else
				{
					x--;
					break;
				}
			}

			p->fit(encodedMode, quantization);

			// discard lonely values
			if (encodedMode == LOSSLESS_COMPRESSION)
			{
				p->valid = 1;
				ps.push_back(p);
			}
			else
				if (p->values_count < lonelyPixelsRemoval)
				{
					p->valid = 0;
				}
				else
				{
					p->valid = 1;
					ps.push_back(p);
				}

		}
		/*
		if (ps.size() == 0) return ps;
		//// Merge

		if (quantizationMode != LOSSLESS_COMPRESSION)
		{
			
			for (int iter = 0; iter < 3; iter++)
			{
				for (int i = 0; i < ps.size() - 1; i++)
				{
					if (!ps[i]->valid) continue;
					if (similitud(ps[i], ps[i + 1], quantizationMode) < 5.0)
					{
						merge(ps[i], ps[i + 1]);
					}
				}
			}
		}
		*/
		return ps;
	}


	virtual void createFromImage(cv::Mat&m)
	{
		std::mutex mtx;
		this->mt = m.clone();
		std::cout << "----------------------------" << "\n";
		std::cout << "Compression Mode " << compressionModes[ encodedMode ] << "\n";
		std::cout << "----------------------------" << "\n";

		h = m.rows;
		w = m.cols;

		
		allocateMem();
		startProcess("createFromImage" + compressionModes[encodedMode]);
		unsigned short* pixels = (unsigned short*)m.data;

		// Release Mem
		memMgr.releaseAll();

		/// For all pixels
#pragma omp parallel for
		for (int y = 0; y < m.rows; y++)
		{
			// Old method. 
			if (improveMethod)
			{
				encodeRowIter(y, m.cols, pixels, encodedMode, 10);
			}
			else
			{
				encodeRow(y, m.cols, pixels);
			}
			//
					
		}

		splines.clear();
		for (auto& sp : memMgr.allocated)
		{
			if (sp->valid)
			{
				splines.push_back(sp);
			}
		}

		endProcess("createFromImage" + compressionModes[encodedMode]);

	}

	virtual void computeMetrics()
	{
		double maxError = 0;
		double minError = 1000;
		double accumError = 0;
		for (auto p : splines)
		{
			double err = p->error();
			if (err> maxError)
			{
				maxError = p->error();
			}

			if (err < minError)
			{
				minError = err;
			}

			accumError += err;

		}
		std::cout << "-------------------------------------" << "\n";
		std::cout << " Splines " << splines.size() << "\n";
		std::cout << "   Min Error " << minError << "\n";
		std::cout << "   Max Error " << maxError << "\n";
		std::cout << "	 Mean Error " << accumError/splines.size() << "\n";


		std::cout << "-------------------------------------" << "\n";
	}

	virtual size_t encode(cv::Mat& _m, std::string fn)
	{
		
		createFromImage(_m);

		unsigned int size = 0;
		startProcess("encode" + compressionModes[encodedMode]);
		vectorizeSplines();
		/////////////////////////////////////////////////////
	  // Compress using LZ4
		char* srcBuffer = (char*)vectorized;
		size_t srcSize = vectorized_count * 2;

		size_t outSize = srcSize;
		endProcess("encode" + compressionModes[encodedMode]);


		startProcess("zip" + compressionModes[encodedMode]);

		if (zstd_compression_level > 0)
		{
#ifdef ZSTD
			size_t const cBuffSize = ZSTD_compressBound(srcSize);

			outSize = ZSTD_compressCCtx(ctx, outBuffer, cBuffSize, srcBuffer, srcSize, zstd_compression_level);
#endif
		}
		else
		{
			outBuffer = (unsigned short*)srcBuffer;
		}
		endProcess("zip" + compressionModes[encodedMode]);

	

		if (fn != "")
		{
			std::ofstream out(fn, std::ios::out | std::ios::binary);
			if (!out)
			{
				std::cout << "Cannot open output file\n";
				return 0;
			}

			out.write((const char*)&w, 2);
			out.write((const char*)&h, 2);
			out.write((const char*)&encodedMode, 2);
			out.write((const char*)&zstd_compression_level, 2);			
			
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

		unsigned short* pixels = (unsigned short*)m.data;

		int maxLength = 0;
#pragma omp parallel for
		for (int i = 0 ; i<splines.size(); i++)
		{
			spline_data* sp = splines[i];
			unsigned short value = 0;

			for (int i = 0; i < sp->values_count; i++)
			{
				unsigned short y = sp->y0;
				unsigned short x = sp->x0 + i;

				value = sp->getValue(i, encodedMode) ;

				if (this->saveResidual && sp->error() > 0)
				{
					pixels[y * w + x] = value + sp->residual[i] * quantization;
				}
				else
				{
					pixels[y * w + x] = value;
				}

				
			}
		}
				
		return m;
	}
	
	virtual size_t saveToFile(std::string fn)	
	{
		unsigned int size = 0;

		vectorizeSplines();
		/////////////////////////////////////////////////////
	  // Compress using LZ4
		char* srcBuffer = (char*)vectorized;
		size_t srcSize = vectorized_count * 2;

		auto start = high_resolution_clock::now();

#ifdef ZSTD
		size_t const cBuffSize = ZSTD_compressBound(srcSize);
		size_t const outSize = ZSTD_compressCCtx(ctx,outBuffer, cBuffSize, srcBuffer, srcSize, zstd_compression_level);
#else
		size_t const outSize = srcSize;
#endif
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
	
		size_t outSize = get_file_size(fn)-8;

		/// Prepare DATA
		std::ifstream input(fn, std::ios::binary | std::ios::in);
		if (!input)
		{
			std::cout << "Cannot open input file\n";
			return cv::Mat();
		}
		
		
		input.read((char*)&w, 2);
		input.read((char*)&h, 2);
		input.read((char*)&encodedMode, 2);
		input.read((char*)&zstd_compression_level, 2);		
			   
		size_t srcSize = w * h * 2;

		std::cout << "Decode parameters " << w << "x" << h << " mode " << compressionModes[encodedMode] << " zip compression "<< zstd_compression_level<<"\n";
		
		allocateMem();


		input.read((char*)compBuffer, outSize);
		
		////////////////////////////
		input.close();

		size_t decSz;

		startProcess("decode" + compressionModes[encodedMode]);
		if (zstd_compression_level > 0)
		{
			// bin mask
#ifdef ZSTD
			decSz = ZSTD_decompress(inBuffer, srcSize, compBuffer, outSize);

			if (ZSTD_isError(decSz))
			{
				std::cout << ZSTD_getErrorName(decSz) << "\n";

			}
#endif
		}
		else
		{
			inBuffer = compBuffer;
			decSz = outSize;

		}

		std::vector<spline_data*> readSP;
		
		int index = 0;
		while (index < decSz/2)
		{
			// SIGNAL
			spline_data* sp = memMgr.getNew();
			sp->values_count = inBuffer[index]; index++;
			sp->x0 = inBuffer[index]; index++;
			sp->y0 = inBuffer[index]; index++;

			if (encodedMode == LOSSLESS_COMPRESSION)
			{
				for (int i = 0; i < sp->values_count; i++)
				{
					// BUG
					sp->values.push_back(inBuffer[index]);
					index++;
				}
			}
			else
				if (encodedMode == LINEAR_COMPRESSION)
				{
					float coef0 = inBuffer[index]; index++;
					
					float coef1 = half_to_float(inBuffer[index]); index++;

					sp->coefs[0] = coef0;
					sp->coefs[1] = coef1;
					sp->coefs[2] = 0.0;
					sp->coefs[3] = 0.0;
				}
				else
					if (sp->values_count < MIN_SAMPLES_EQ)
					{
						float coef0 = inBuffer[index]; index++;
						float coef1 = half_to_float(inBuffer[index]); index++;

						sp->coefs[0] = coef0;
						sp->coefs[1] = coef1;
						sp->coefs[2] = 0;
						sp->coefs[3] = 0;
					}
					else
				if (encodedMode == SPLINE_COMPRESSION)
				{
						float coef0 = inBuffer[index]; index++;
						float coef1 = half_to_float(inBuffer[index]); index++;
						float coef2 = half_to_float( inBuffer[index]); index++;
						float coef3 = half_to_float(inBuffer[index]); index++;

						sp->coefs[0] = coef0;
						sp->coefs[1] = coef1;
						sp->coefs[2] = coef2;
						sp->coefs[3] = coef3;
				
				}
				else
				{
					sp->coefs[0] = inBuffer[index]; index++;
					sp->coefs[1] = half_to_float(inBuffer[index]); index++;
					sp->coefs[2] = half_to_float(inBuffer[index]); index++;
					sp->coefs[3] = half_to_float(inBuffer[index]); index++;
					sp->coefs[4] = half_to_float(inBuffer[index])/1000; index++;
					sp->coefs[5] = half_to_float(inBuffer[index])/10000; index++;
				}

			readSP.push_back(sp);

		}

		this->splines.clear();
		splines.swap(readSP);
	
		cv::Mat _m = restoreAsImage();
		
		endProcess("decode" + compressionModes[encodedMode]);
		return _m;
	}

	~splineCompression()
	{
		free(inBuffer);
		free(compBuffer);
	}


	void display(int row, int mode, int iter)
	{
		cv::Mat histImg;
		histImg.create(cv::Size(w, 200), CV_8UC3);

		histImg.setTo(255);
		cv::RNG rng(12345);

		int maxH = 45000;
		int hist_h = 200;
		unsigned short* pixels = (unsigned short*)mt.data;
		std::vector<spline_data*> ps =  encodeRowIter(row, this->mt.cols, pixels, mode, iter);

		double gerror = 0;
		double max_error = 0;

		for (auto& sp : ps)
		{
			if (sp->y0 != row) continue;
			if (!sp->valid) continue;
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

			gerror += sp->error();

			max_error = MAX(max_error, sp->error());

			if (sp->values_count > 0)
			{
				for (int i = 0; i < sp->values_count; i = i + 1)
				{

					int value = sp->getValue(i, 0);
					double y = (double)(value) / (maxH)*histImg.rows;
					int x = sp->x0 + i;
					/// Render Histogram
					cv::line(histImg, cv::Point(x, hist_h), cv::Point(x, hist_h - y), color, 1, 8, 0);
				}
			}
			else
			{
				for (int i = 0; i < sp->values_count; i = i + 1)
				{
					int x = sp->x0 + i;
					int y = sp->y0;

					double value = mt.at<unsigned short>(y,x);

					double z = (double)(value) / (maxH)*histImg.rows;
				
					/// Render Histogram
					line(histImg, cv::Point(x, hist_h), cv::Point(x, hist_h - z), color, 1, 8, 0);
				}
			}

			if (mode == SPLINE_COMPRESSION || mode == SPLINE5_COMPRESSION)
			{
				
				for (int i = 0; i < sp->values_count - 1; i = i + 1)
				{
					int x = sp->x0 + i;

					double v0 = sp->getValue(i, mode);
					double v1 = sp->getValue(i+1, mode);

					double y0 = (double)(v0) / (maxH)*histImg.rows;
					double y1 = (double)(v1) / (maxH)*histImg.rows;

					/// Render Histogram
					line(histImg, cv::Point(x, hist_h-y0), cv::Point(x+1, hist_h - y1), cv::Scalar(0,255,255), 1, 8, 0);
				}
				

			}
			else
			{
				// Render estimation
				int x0 = sp->x0;
				int x1 = x0 + sp->values_count;
				int y0 = (double)sp->getValue(0, LINEAR_COMPRESSION) / (maxH)*histImg.rows;
				int y1 = (double)sp->getValue(sp->values_count - 1, LINEAR_COMPRESSION) / (maxH)*histImg.rows;
				line(histImg, cv::Point(x0, hist_h - y0), cv::Point(x1, hist_h - y1), cv::Scalar(0, 255, 0), 1, 8, 0);
			}
			
			
		}

		cv::putText(histImg, "error:" + std::to_string(gerror / ps.size()), cv::Point(20, 50), 1, 1.5, cv::Scalar(255, 255, 255));

		cv::putText(histImg, "error:" + std::to_string(max_error), cv::Point(20, 80), 1, 1.5, cv::Scalar(255, 255, 255));

		cv::imshow("histo" + std::to_string(row) , histImg);
		//std::cout << "End HistDisplay" << std::endl;
	}
};
