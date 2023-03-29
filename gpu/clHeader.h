#ifndef FIT_DEPTH
#define FIT_DEPTH

#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

using namespace std;

#define SPLINE_VECTOR_SIZE 256
#define CL_LOSSLESS_COMPRESSION 0
#define CL_LZ4_COMPRESSION 1
#define CL_ZSTD_COMPRESSION 2
#define CL_SPLINE_COMPRESSION 3
#define CL_LINEAR_COMPRESSION 4
#define CL_SPLINE5_COMPRESSION 5
#define CL_MIN_EXPECTED_ERROR 128

#define CL_MIN_SAMPLES_EQ 24
#define CL_lonelyPixelsRemoval 2
#define CL_quantization 8

#define CL_ALLOCATION_PER_THREAD 120
#define CL_TOTAL_ALLOCATION 60000
#define CL_CLOSE_MASK 0xFFFE

#if defined OPENCL_GPU

typedef global int* Int_ptr;
typedef global char* Char_ptr;
typedef global uchar* Uchar_ptr;
typedef global short* Short_PTR;
typedef global unsigned int* Uint_ptr;
typedef global float* Float_ptr;
typedef private float* pFloat_ptr;

typedef enum {
	LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

#else
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned int* Uint_ptr;
typedef int* Int_ptr;
typedef char* Char_ptr;
typedef uchar* Uchar_ptr;
typedef short* Short_PTR;
typedef float* Float_ptr;
typedef float* pFloat_ptr;
#endif 



struct cl_spline_data
{
	int valid;
	int effective;
	unsigned short x0;
	unsigned short y0;
	int visited;
	int cvalues_count;
	int res_counter;
	unsigned short values_count;
	unsigned short values[SPLINE_VECTOR_SIZE];
	char cvalues[SPLINE_VECTOR_SIZE];
	char residual[SPLINE_VECTOR_SIZE];
	float coefs[6];
	float _error;
	int new_ps;
	int last_index;
	int last_value;
};

#if defined OPENCL_GPU
typedef global struct cl_spline_data* spline_ptr;
typedef global struct cl_spline_data** spline_ptr_ptr;
#else
typedef struct cl_spline_data* spline_ptr;
typedef struct cl_spline_data** spline_ptr_ptr;
#endif

struct cl_memManager
{
	
	spline_ptr allocated_splines;
	int allocated_ids[2048];
	int allocated_size;
	int total_size;
	int final_OuputSize;
	int validSplines;
	int activeThreads;

};


#if defined OPENCL_GPU
typedef global struct cl_memManager* mmgr_ptr;
#else
typedef struct cl_memManager* mmgr_ptr;
#endif


/// external
void setThreadID(int threadID);

void cl_encodeRowIter( int w, int rows, Short_PTR pixels, int mode, int itercount, mmgr_ptr  mMgr, spline_ptr allocated_splines,  int improved);
void cl_vectorizeSplines(spline_ptr  splines, int splinesCount, Short_PTR vectorized, mmgr_ptr  mMgr, int encodedMode, int scale, int saveResidual);
void mmger_init(mmgr_ptr mmgr, int count, int threadCount);
void mmger_releaseAll(mmgr_ptr mmgr, spline_ptr allocated_splines, int threadCount);
void cl_test(int w, Short_PTR pixels, int mode, int itercount, mmgr_ptr  mMgr, int improved);
void cl_mem_init(mmgr_ptr mmgr, spline_ptr splines, int rows);
#endif