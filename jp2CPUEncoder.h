
#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/shape/shape_distance.hpp>

#include "thirdparty/jp2/format_defs.h"
#include "thirdparty/jp2/opj_string.h"
#include "thirdparty/jp2/openjpeg.h"

typedef enum opj_prec_mode {
	OPJ_PREC_MODE_CLIP,
	OPJ_PREC_MODE_SCALE
} opj_precision_mode;

typedef struct opj_prec {
	OPJ_UINT32         prec;
	opj_precision_mode mode;
} opj_precision;

typedef struct opj_decompress_params {
	/** core library parameters */
	opj_dparameters_t core;

	/** input file name */
	char *infile;
	/** output file name */
	char outfile[OPJ_PATH_LEN];
	/** input file format 0: J2K, 1: JP2, 2: JPT */
	int decod_format;
	/** output file format 0: PGX, 1: PxM, 2: BMP */
	int cod_format;
	/** index file name */
	char indexfilename[OPJ_PATH_LEN];

	/** Decoding area left boundary */
	OPJ_UINT32 DA_x0;
	/** Decoding area right boundary */
	OPJ_UINT32 DA_x1;
	/** Decoding area up boundary */
	OPJ_UINT32 DA_y0;
	/** Decoding area bottom boundary */
	OPJ_UINT32 DA_y1;
	/** Verbose mode */
	OPJ_BOOL m_verbose;

	/** tile number of the decoded tile */
	OPJ_UINT32 tile_index;
	/** Nb of tile to decode */
	OPJ_UINT32 nb_tile_to_decode;

	opj_precision* precision;
	OPJ_UINT32     nb_precision;

	/* force output colorspace to RGB */
	int force_rgb;
	/* upsample components according to their dx/dy values */
	int upsample;
	/* split output components to different files */
	int split_pnm;
	/** number of threads */
	int num_threads;
	/* Quiet */
	int quiet;
	/** number of components to decode */
	OPJ_UINT32 numcomps;
	/** indices of components to decode */
	OPJ_UINT32* comps_indices;
} opj_decompress_parameters;


/* -------------------------------------------------------------------------- */
#define JP2_RFC3745_MAGIC "\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a"
#define JP2_MAGIC "\x0d\x0a\x87\x0a"
/* position 45: "\xff\x52" */
#define J2K_CODESTREAM_MAGIC "\xff\x4f\xff\x51"

cv::Mat readJP2File(std::string jp2File);
void writeJP2File(std::string outFile, cv::Mat& frame);