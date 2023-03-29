#include "gpu/clHeader.h"


#ifdef OPENCL_GPU

#else

#define CLK_LOCAL_MEM_FENCE 1
bool definedTheadId = false;
int _threadID = 0;
void setThreadID(int threadID)
{
	if (threadID >= 0)
	{
		_threadID = threadID;
		definedTheadId = true;
	}
	else
	{
		definedTheadId = false;
	}

}

int openclPeekAtLastError() { return 0; }

int get_group_id(int dim) { return 0; }
int get_local_id(int dim) 
{ 
	if (definedTheadId)
		return _threadID;

	return omp_get_thread_num(); 
}

int get_global_size(int dim) { return 0; }
int get_local_size(int dim) { return 0; }

void barrier(int barrier) {}
int cudaPeekAtLastError() { return 0; }
void check_error(int X) { return; }
int cuda_gridsize(int v)
{
	return v;
}

void opencl_push_array(float *x_gpu, float *x, size_t n) { }

void opencl_pull_array(float *x_gpu, float *x, size_t n) { }

float opencl_mag_array(float *x_gpu, size_t n) { return 0.0f; }


#endif


#if defined OPENCL_GPU


#else


int thID = -1;

void set_threadId(int th)
{
	thID = th;
}

int get_global_id(int dim)
{
	if (thID >= 0)
		return thID;
	else
		return omp_get_thread_num();
}

void __syncthreads() {}
float atomic_add(float x, float y) { return x + y; }

#include <mutex>
#include <omp.h>
std::mutex base;

void atomic_increment(int &value)
{

	int  th = omp_get_thread_num();

	base.lock();
	{
		value++;
	}
	base.unlock();

}


#endif


unsigned int as_uintF(const float x)
{
	return *(unsigned int*)&x;
}

float as_floatU(const uint x) {
	return *(float*)&x;
}

unsigned short as_ushortF(char y, char x)
{
	unsigned short result = (((unsigned short)y << 8) & 0xF00) | x;

	return result;
}


float cl_half_to_float(unsigned short x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
	const uint e = (x & 0x7C00) >> 10; // exponent
	const uint m = (x & 0x03FF) << 13; // mantissa
	const uint v = as_uintF((float)m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
	return as_floatU((x & 0x8000) << 16 | (e != 0)*((e + 112) << 23 | m) | ((e == 0)&(m != 0))*((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
}
unsigned short cl_float_to_half(float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
	const uint b = as_uintF(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
	const uint e = (b & 0x7F800000) >> 23; // exponent
	const uint m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
	return (b & 0x80000000) >> 16 | (e > 112)*((((e - 112) << 10) & 0x7C00) | m >> 13) | ((e < 113)&(e > 101))*((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}


void spline_clear(spline_ptr p)
{
	p->valid = 0;
	p->values_count = 0;
	p->effective = 0;
	p->x0 = 0;
	p->y0 = 0;
	p->res_counter = 0;
	p->cvalues_count = 0;
	p->_error = 0;
	p->values_count = 0;
	p->new_ps = 0;
	p->last_index = 0;
	p->last_value = 0;
	p->visited = 0;

	for (int i = 0; i < 6; i++) p->coefs[i] = 0;
#ifdef GPU
	for (int i = 0; i < 5; i++) p->values[i] = 0;
	for (int i = 0; i < 5; i++) p->cvalues[i] = 0;
	for (int i = 0; i < 5; i++) p->residual[i] = 0;
#else
	for (int i = 0; i < SPLINE_VECTOR_SIZE; i++) p->values[i] = 0;
	for (int i = 0; i < SPLINE_VECTOR_SIZE; i++) p->cvalues[i] = 0;
	for (int i = 0; i < SPLINE_VECTOR_SIZE; i++) p->residual[i] = 0;
#endif


}

int spline_getValue(spline_ptr p, int index, int mode)
{
	if (mode == CL_LOSSLESS_COMPRESSION)
	{

		if (p->values_count > index) return p->values[index];

		// Progressive
		if (index == p->last_index + 1)
		{
			p->last_index++;
			p->last_value = p->last_value + p->cvalues[index];
			return p->last_value;
		}
		else
		{
			int value = p->coefs[0];

			for (int i = 0; i <= index; i++) value += p->cvalues[i];

			p->last_index = index;
			p->last_value = value;
			return value;
		}
	}
	else
		if (mode == CL_LINEAR_COMPRESSION || p->values_count < CL_MIN_SAMPLES_EQ)
		{
			int x = index;
			float v0 = p->coefs[0] + p->coefs[1] * x + p->coefs[2] * x * x + p->coefs[3] * (x) * (x) * (x);

			return (int)(v0);
		}
		else
			if (mode == CL_SPLINE_COMPRESSION)
			{
				int x = index;
				float v0 = p->coefs[0] + p->coefs[1] * x + p->coefs[2] * x * x + p->coefs[3] * (x) * (x) * (x);
				return (int)(v0);
			}
			else
				if (mode == CL_SPLINE5_COMPRESSION)
				{
					int x = index;
					float v0 = p->coefs[0] + p->coefs[1] * x + p->coefs[2] * x * x + p->coefs[3] * (x) * (x) * (x)+p->coefs[4] * x * x * x * x + p->coefs[5] * x * x * x * x * x;;
					return (int)(v0);
				}

	return 0;
}

float spline_error(spline_ptr p)
{
	return p->_error;
}


spline_ptr mmger_getNew(mmgr_ptr mmgr, spline_ptr allocated_splines)
{
	// only one thread should come here
	int th_id = get_global_id(0);
	int id = mmgr->allocated_ids[th_id];

	mmgr->allocated_ids[th_id]++;

	if (id >= mmgr->total_size)
	{
		//printf("Not enough mem ");
		return NULL;
	}
	///  Clear data
	spline_clear(&allocated_splines[id]);

	return &allocated_splines[id];
}

void mmger_releaseAll(mmgr_ptr mmgr, spline_ptr allocated_splines, int threadCount)
{

#pragma omp parallel for
	for (int i = 0; i < mmgr->total_size; i++)
	{
		spline_clear(&allocated_splines[i]);
	}

	for (int i = 0; i < threadCount; i++) mmgr->allocated_ids[i] = 0;
}

void mmger_init(mmgr_ptr mmgr, int count, int threadCount)
{
#if defined OPENCL_GPU
#else
	mmgr->allocated_splines = new cl_spline_data[count * CL_ALLOCATION_PER_THREAD];

	mmgr->total_size = count * CL_ALLOCATION_PER_THREAD;

	mmgr->activeThreads = count;

	for (int i = 0; i < threadCount; i++) mmgr->allocated_ids[i] = i * CL_ALLOCATION_PER_THREAD;

	for (int i = 0; i < count * CL_ALLOCATION_PER_THREAD; i++) spline_clear(&mmgr->allocated_splines[i]);
#endif
}



#ifdef OPENCL_GPU 
kernel
#endif
void cl_mem_init(mmgr_ptr mmgr , spline_ptr splines, int rows)
{
	int th_id = get_global_id(0);

	if (th_id > rows) return;

	int baseID = th_id * CL_ALLOCATION_PER_THREAD;
	mmgr->allocated_ids[th_id] = baseID; 

	for (int i = 0; i < CL_ALLOCATION_PER_THREAD; i++)
	{
		spline_clear(&splines[baseID + i]);
	}
}


#ifdef OPENCL_GPU 
kernel
#endif
void fill_kernel(int N, float ALPHA, Float_ptr X, int INCX)
{
	int i = get_global_id(0); //(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) X[i*INCX] = ALPHA;
}

#ifdef OPENCL_GPU 
kernel
#endif
void copy_kernel(int N, Float_ptr X, int OFFX, int INCX, Float_ptr Y, int OFFY, int INCY)
{
	int i = get_global_id(0); //(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

#ifdef OPENCL_GPU 
kernel
#endif
void mul_kernel(int N, Float_ptr X, int INCX, Float_ptr Y, int INCY)
{
	int i = get_global_id(0); //(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) Y[i*INCY] *= X[i*INCX];
}


#ifdef OPENCL_GPU 
kernel
#endif
void cl_test(int w, Short_PTR pixels, int mode, int itercount, mmgr_ptr  mMgr, int improved)
{
	int th_id = get_global_id(0);
	mMgr->allocated_ids[th_id] = th_id * CL_ALLOCATION_PER_THREAD;
	// iterate through image and test amount of threads
	for (int x = 0; x < w; x = x + 1)
	{
		unsigned short value = pixels[th_id * w + x];
		if (value < 300) continue;

		// only one thread should come here
		int id = mMgr->allocated_ids[th_id];
		mMgr->allocated_ids[th_id]++;
	}

}

/*

CPU INVOCATIONS

*/


int float_abs_compare(const void * a, const void * b)
{
	float fa = *(const float*)a;
	if (fa < 0) fa = -fa;
	float fb = *(const float*)b;
	if (fb < 0) fb = -fb;
	return (fa > fb) - (fa < fb);
}

///////////////////////////////////////////////////
////////////////////////////////////////////

///////////////////////////////////////////////////
////////////////////////////////////////////
int cl_fitIt(
	pFloat_ptr x,
	pFloat_ptr y,
	int             order,
	Float_ptr     coeffs, int nelements)
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



void computeResidual(spline_ptr p, int mode, int QUANTIZATION)
{
	p->_error = 0;

	//residual.clear();
	if (p->values_count > 0)
	{

		if (mode == CL_LOSSLESS_COMPRESSION)
		{
			////////////////////////////////////////////
			for (int i = 0; i < p->values_count; i++)
			{
				// LossLess
				if (i > 0)
				{
					p->cvalues[p->cvalues_count] = (p->values[i] - p->values[i - 1]) / QUANTIZATION;
					p->cvalues_count++;
				}
				else
				{
					p->cvalues[p->cvalues_count] = 0;
					p->cvalues_count++;
				}

			}
		}
		else
		{
			////////////////////////////////////////////
			for (int i = 0; i < p->values_count; i++)
			{
				short diff = (p->values[i] - spline_getValue(p, i, mode));

				p->residual[p->res_counter] = (diff / QUANTIZATION); p->res_counter++;

				p->_error += abs(diff / QUANTIZATION);
				//residual.push_back(evaluate(i) - values[i]);
			}
		}
	}

	p->_error = p->_error / p->values_count;
}


void fit(spline_ptr p, int mode)
{
	p->coefs[0] = p->coefs[1] = p->coefs[2] = p->coefs[3] = p->coefs[4] = p->coefs[5] = 0;

	if (p->values_count == 0) return;

	if (mode == CL_LOSSLESS_COMPRESSION)
	{
		p->coefs[0] = p->values[0];
	}
	else
		if (mode == CL_LINEAR_COMPRESSION || p->values_count < CL_MIN_SAMPLES_EQ)
		{

			p->coefs[0] = p->values[0];

			p->coefs[1] = (float)(1.0f*p->values[p->values_count - 1] - p->values[0]) / p->values_count;


		}
		else
		{
			float xs[SPLINE_VECTOR_SIZE];
			float ys[SPLINE_VECTOR_SIZE];
			int xs_counter = 0;

			int subsample = 1;
			if (p->values_count > 100) subsample = 4;
			else if (p->values_count > 20) subsample = 2;

			for (int i = 0; i < p->values_count; i = i + subsample)
			{
				int value = p->values[i];
				xs[xs_counter] = i;
				ys[xs_counter] = value;
				xs_counter++;
			}

			if (mode == CL_SPLINE_COMPRESSION) cl_fitIt(xs, ys, 3, p->coefs, xs_counter);
			else cl_fitIt(xs, ys, 5, p->coefs, xs_counter);
		}

	computeResidual(p, mode, CL_quantization);

}

bool split(spline_ptr p, spline_ptr p0, spline_ptr p1, int mode)
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

	for (int middle = 2; middle < p->values_count - 10; middle = middle + 2)
	{
		spline_clear(p0);
		spline_clear(p1);

		for (int i = 0; i < p->values_count; i = i + 1)
		{
			unsigned short v0 = p->values[i]; // orig value
			if (i < middle) { p0->values[p0->values_count] = v0;  p0->values_count++; }
			else { p1->values[p1->values_count] = v0;  p1->values_count++; }

		}
		fit(p0, mode);
		fit(p1, mode);

		//double errorP = p->error();
		float error0 = spline_error(p0);
		float error1 = spline_error(p1);


		if ((error0 * 0.5 + error1 * 0.5 + 4000 / middle) < worstError)
		{
			worstError = error0 * 0.5 + error1 * 0.5 + 4000 / middle; // MAX(error0, error1);
			best_middle = middle;
		}

	}


	if (worstError < spline_error(p))
	{
		spline_clear(p0);
		spline_clear(p1);

		for (int i = 0; i < p->values_count; i = i + 1)
		{
			unsigned short v0 = p->values[i]; // orig value
			if (i < best_middle) { p0->values[p0->values_count] = v0;  p0->values_count++; }
			else { p1->values[p1->values_count] = v0;  p1->values_count++; }

		}

		fit(p1, mode);
		fit(p0, mode);

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

#ifdef OPENCL_GPU 
kernel
#endif
void cl_encodeRowIter(int w, int rows, Short_PTR pixels, int mode, int itercount, mmgr_ptr  mMgr, spline_ptr allocated_splines, int improved)
{
	int thID = get_local_id(0);

	if (thID > rows) return;

	mMgr->allocated_ids[thID] = thID * CL_ALLOCATION_PER_THREAD;

	// Take a first approach
	//spline_ptr ps[1024];
	int ps_counter = 0;

	//	if (y != 250) return ps;
	int window = 5;
	/////////////////////////////////

	for (int x = 0; x < w; x++)
	{
		unsigned short value = pixels[thID * w + x];
		if (value < 300) continue;

		spline_ptr p = mmger_getNew(mMgr, allocated_splines);

		if (p == NULL)
		{
			break;
		}
		p->x0 = x;
		p->y0 = thID;
		p->coefs[0] = value;

		// check consecutive pixels
		while (true)
		{
			if (x >= w) break;
			// take a sample
			unsigned short value2 = pixels[thID * w + x];
			unsigned short value3 = pixels[thID * w + x + 1];
			// may be it is noise
			if (value2 < 300)
			{
				break;
			}

			if (abs(value - value2) < 128 || abs(value - value3) < 128)
			{
				p->values[p->values_count] = value2; p->values_count++;
				if (p->values_count >= SPLINE_VECTOR_SIZE)
				{
					//	printf("error");
					break;
				}
				value = value2;

				x += 1;
			}
			else
			{
				x--;
				break;
			}


		}

		// discard lonely values

		if (p->values_count >= CL_lonelyPixelsRemoval)
		{
			fit(p, mode);
			p->valid = 1;
			p->effective = 1;

		}
		else
		{
			p->valid = 0;
			/// store gradients missing
		}

	}
	/// split planes	
	/*
	if (improved)
	{
		spline_ptr new_ps[1024];
		int np_count = 0;

		for (int i = 0; i < ps_counter; i++) ps[i]->visited = false;

		for (int iter = 0; iter < itercount; iter++)
		{
			bool changed = false;

			for (int i = 0; i < ps_counter; i++)
			{
				spline_ptr p = ps[i];
				if (!p->valid) continue;

				if (p->visited || spline_error(p) < CL_MIN_EXPECTED_ERROR)
				{
					p->visited = true;
					new_ps[np_count] = p;	np_count++;
					continue;
				}

				if (p->values_count > CL_MIN_SAMPLES_EQ)
				{
					spline_ptr p0 = mmger_getNew(mMgr, allocated_splines);
					spline_ptr p1 = mmger_getNew(mMgr, allocated_splines);

					if (split(p, p0, p1, mode))
					{
						changed = true;
						new_ps[np_count] = p0;	np_count++;
						new_ps[np_count] = p1;	np_count++;
						p->valid = 0;
					}
					else
					{
						new_ps[np_count] = p1;	np_count++;
					}
				}
				else
				{
					new_ps[np_count] = p;	np_count++;
				}

			}

			if (!changed) break;

			for (int i = 0; i < np_count; i++)
			{
				ps[i] = new_ps[i];
			}
			ps_counter = np_count;

		}

		// Mask as final
		for (int i = 0; i < np_count; i++)
		{
			new_ps[i]->effective = true;
		}
	}
	else
	
	{

		// Mask as final
		for (int i = 0; i < ps_counter; i++)
		{
		//	ps[i]->effective = true;
		}
	}
	*/

}




///////////////////////////////
///////////////////////////////
#ifdef OPENCL_GPU 
kernel
#endif
void cl_vectorizeSplines(spline_ptr  splines, int splinesCount, Short_PTR vectorized, mmgr_ptr  mMgr, int encodedMode, int scale, int saveResidual)
{
	int thID = get_local_id(0);

	int vectorized_count = thID * splinesCount;

	mMgr->validSplines = 0;

	vectorized_count += 1;

	{
		int startIndex = thID * CL_ALLOCATION_PER_THREAD;
		int endIndex = mMgr->allocated_ids[thID];
		for (int i = startIndex; i < endIndex; i++)
		{
			spline_ptr sp = &splines[i];
			if (!sp->effective) continue;
			if (sp->values_count == 0) continue;
			// SIGNAL
			mMgr->validSplines++;
			vectorized[vectorized_count] = sp->values_count; vectorized_count++;
			vectorized[vectorized_count] = sp->x0; vectorized_count++;
			vectorized[vectorized_count] = sp->y0; vectorized_count++;

			if (encodedMode == CL_LOSSLESS_COMPRESSION)
			{
				vectorized[vectorized_count] = sp->coefs[0]; vectorized_count++;
				if (sp->values_count == 1)
				{
					/////
				}
				else
				{
					for (int i = 0; i < sp->cvalues_count - 1; i = i + 2)
					{
						vectorized[vectorized_count] = as_ushortF(sp->cvalues[i] / scale, sp->cvalues[i + 1] / scale); vectorized_count++;
					}

					if (sp->cvalues_count % 2 == 1)
					{
						vectorized[vectorized_count] = as_ushortF(sp->cvalues[sp->values_count - 1] / scale, 0);  vectorized_count++;
					}
				}

			}
			else
				if (encodedMode == CL_LINEAR_COMPRESSION)
				{
					vectorized[vectorized_count] = sp->coefs[0]; vectorized_count++;
					vectorized[vectorized_count] = cl_float_to_half(sp->coefs[1]); vectorized_count++;
				}
				else
				{
					if (sp->values_count < CL_MIN_SAMPLES_EQ)
					{
						vectorized[vectorized_count] = sp->coefs[0]; vectorized_count++;
						vectorized[vectorized_count] = cl_float_to_half(sp->coefs[1]); vectorized_count++;
					}
					else
						if (encodedMode == CL_SPLINE_COMPRESSION)
						{
							vectorized[vectorized_count] = sp->coefs[0]; vectorized_count++;
							vectorized[vectorized_count] = cl_float_to_half(sp->coefs[1]); vectorized_count++;
							vectorized[vectorized_count] = cl_float_to_half(sp->coefs[2]); vectorized_count++;
							vectorized[vectorized_count] = cl_float_to_half(sp->coefs[3]); vectorized_count++;
						}
						else
						{
							vectorized[vectorized_count] = sp->coefs[0]; vectorized_count++;
							vectorized[vectorized_count] = cl_float_to_half(sp->coefs[1]); vectorized_count++;
							vectorized[vectorized_count] = cl_float_to_half(sp->coefs[2]); vectorized_count++;
							vectorized[vectorized_count] = cl_float_to_half(sp->coefs[3]); vectorized_count++;
							vectorized[vectorized_count] = cl_float_to_half(sp->coefs[4] * 1000); vectorized_count++;
							vectorized[vectorized_count] = cl_float_to_half(sp->coefs[5] * 10000); vectorized_count++;
						}
				}


			if (encodedMode != CL_LOSSLESS_COMPRESSION)
			{
				if (saveResidual && spline_error(sp) > 0)
				{
					for (int i = 0; i < sp->values_count - 1; i = i + 2)
					{
						vectorized[vectorized_count] = as_ushortF(sp->residual[i], sp->residual[i + 1]); vectorized_count++;
					}
				}
			}
		}
	}
	vectorized[thID * splinesCount] = vectorized_count - thID * splinesCount;
	
	
}

