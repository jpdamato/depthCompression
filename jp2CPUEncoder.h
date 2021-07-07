#pragma once

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

/////////////////////////////////////////////////////
// J2K reader

int infile_format(const char *fname)
{
	FILE *reader;
	const char *s, *magic_s;
	int ext_format, magic_format;
	unsigned char buf[12];
	OPJ_SIZE_T l_nb_read;
	reader = fopen(fname, "rb");

	if (reader == NULL) {
		return -2;
	}

	memset(buf, 0, 12);
	l_nb_read = fread(buf, 1, 12, reader);
	fclose(reader);
	if (l_nb_read != 12) {
		return -1;
	}

	ext_format = JP2_CFMT;

	if (ext_format == JPT_CFMT) {
		return JPT_CFMT;
	}

	if (memcmp(buf, JP2_RFC3745_MAGIC, 12) == 0 || memcmp(buf, JP2_MAGIC, 4) == 0) {
		magic_format = JP2_CFMT;
		magic_s = ".jp2";
	}
	else if (memcmp(buf, J2K_CODESTREAM_MAGIC, 4) == 0) {
		magic_format = J2K_CFMT;
		magic_s = ".j2k or .jpc or .j2c";
	}
	else {
		return -1;
	}

	if (magic_format == ext_format) {
		return ext_format;
	}

	s = fname + strlen(fname) - 4;

	fputs("\n===========================================\n", stderr);
	fprintf(stderr, "The extension of this file is incorrect.\n"
		"FOUND %s. SHOULD BE %s\n", s, magic_s);
	fputs("===========================================\n", stderr);

	return magic_format;
}
void set_default_parameters(opj_decompress_parameters* parameters)
{
	if (parameters) {
		memset(parameters, 0, sizeof(opj_decompress_parameters));

		/* default decoding parameters (command line specific) */
		parameters->decod_format = -1;
		parameters->cod_format = -1;

		/* default decoding parameters (core) */
		opj_set_default_decoder_parameters(&(parameters->core));
	}
}
cv::Mat readJP2File(std::string jp2File)
{
	cv::Mat res;
	int64 startRendertime = cv::getTickCount();
	std::cout << "Start decoding jp2File" << "\n";

	opj_decompress_parameters parameters;           /* decompression parameters */

	OPJ_INT32 num_images, imageno;
	int failed = 0;
	OPJ_FLOAT64 t, tCumulative = 0;
	OPJ_UINT32 numDecompressedImages = 0;
	OPJ_UINT32 cp_reduce;

	/* set decoding parameters to default values */
	set_default_parameters(&parameters);

	opj_image_t* image = NULL;
	opj_stream_t *l_stream = NULL;              /* Stream */
	opj_codec_t* l_codec = NULL;                /* Handle to a decompressor */
	opj_codestream_index_t* cstr_index = NULL;

	/* read the input file and put it in memory */
	   /* ---------------------------------------- */
	parameters.infile = (char*)jp2File.c_str();
	parameters.decod_format = infile_format(parameters.infile);
	l_stream = opj_stream_create_default_file_stream(parameters.infile, 1);
	if (!l_stream) {
		std::cout << "ERROR : failed to create stream from the file" << "\n";
		failed = 1;
	}

	/* decode the JPEG2000 stream */
		/* ---------------------- */

	 /* JPEG 2000 compressed image data */
	/* Get a decoder handle */
	l_codec = opj_create_decompress(OPJ_CODEC_JP2);


	/* Setup the decoder decoding parameters using user parameters */
	if (!opj_setup_decoder(l_codec, &(parameters.core)) && !failed) {
		std::cout << "ERROR : failed to setup the decoder" << "\n";

		opj_stream_destroy(l_stream);
		opj_destroy_codec(l_codec);
		failed = 1;
	}

	/* Read the main header of the codestream and if necessary the JP2 boxes*/
	if (!opj_read_header(l_stream, l_codec, &image) && !failed) {
		std::cout << "ERROR -> opj_decompress: failed to read the header\n";
		opj_stream_destroy(l_stream);
		opj_destroy_codec(l_codec);
		opj_image_destroy(image);
		failed = 1;
	}

	if (parameters.numcomps && !failed) {
		if (!opj_set_decoded_components(l_codec,
			parameters.numcomps,
			parameters.comps_indices,
			OPJ_FALSE)) {
			std::cout << "ERROR -> opj_decompress: failed to set the component indices!\n";
			opj_destroy_codec(l_codec);
			opj_stream_destroy(l_stream);
			opj_image_destroy(image);
			failed = 1;
		}
	}



	if (!parameters.nb_tile_to_decode && !failed)
	{
		/* Get the decoded image */
		if (!(opj_decode(l_codec, l_stream, image) &&
			opj_end_decompress(l_codec, l_stream))) {
			std::cout << "ERROR -> opj_decompress: failed to decode image!\n";
			opj_destroy_codec(l_codec);
			opj_stream_destroy(l_stream);
			opj_image_destroy(image);
			failed = 1;

		}

		int w = (int)image->comps[0].w;
		int h = (int)image->comps[0].h;


		int adjustR;
		if (image->numcomps == 3)
		{
			res = cv::Mat(w, h, CV_8UC3);
			uchar* dstData = (uchar*)res.data;
			for (int i = 0; i < w * h; i++)
			{
				for (int c = 0; c < image->numcomps; c++)
				{
					int r;
					r = image->comps[c].data[i];
					r += (image->comps[c].sgnd ? 1 << (image->comps[c].prec - 1) : 0);
					dstData[i*image->numcomps + c] = r;
				}
			}

			cv::cvtColor(res, res, CV_RGB2BGR);
		}
		else
			if (image->numcomps == 1)
			{
				res = cv::Mat(w, h, CV_16UC1);
				short* dstData = (short*)res.data;
				for (int i = 0; i < w * h; i++)
				{
					int r;
					r = image->comps[0].data[w * h - ((i) / (w)+1) * w + (i) % (w)];
					r += (image->comps[0].sgnd ? 1 << (image->comps[0].prec - 1) : 0);
					dstData[i] = r;

				}
			}



	}

	opj_image_destroy(image);
	image = NULL;
	//cv::resize(res, res, cv::Size(), 1080.0 / res.rows, 1080.0 / res.rows);
	//cv::imshow("input", res);
	//cv::waitKey(-1);
	double msecs = abs(startRendertime - cv::getTickCount()) / cv::getTickFrequency();

	std::cout << " FINISHED OK .. Required " << msecs << " secs " << "\n";
	return res;
}



/////////////////////////////////////////////////////
// J2K Writer
void writeJP2File(std::string outFile, cv::Mat& frame)
{
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);

	cv::imwrite(outFile, frame, compression_params);

	return;

	////////////////////
	opj_cparameters_t parameters;   /* compression parameters */

	opj_stream_t *l_stream = 00;
	opj_codec_t* l_codec = 00;
	opj_image_t *image = NULL;

	int num_threads = 4;

	int ret = 0;
	
	l_codec = opj_create_compress(OPJ_CODEC_J2K);
	
	if (!opj_setup_encoder(l_codec, &parameters, image)) {
		fprintf(stderr, "failed to encode image: opj_setup_encoder\n");
		opj_destroy_codec(l_codec);
		opj_image_destroy(image);	
		return;
	}

	if (num_threads >= 1 &&
		!opj_codec_set_threads(l_codec, num_threads)) {
		fprintf(stderr, "failed to set number of threads\n");
		opj_destroy_codec(l_codec);
		opj_image_destroy(image);
		ret = 1;
		return;
	}

	/* encode the image */
	bool bSuccess = opj_start_compress(l_codec, image, l_stream);
	if (!bSuccess) {
		fprintf(stderr, "failed to encode image: opj_start_compress\n");
	}

	bSuccess = bSuccess && opj_encode(l_codec, l_stream);
	if (!bSuccess) 
	{
			fprintf(stderr, "failed to encode image: opj_encode\n");
	}
	
	bSuccess = bSuccess && opj_end_compress(l_codec, l_stream);
	if (!bSuccess) 
	{
		fprintf(stderr, "failed to encode image: opj_end_compress\n");
	}

	if (!bSuccess) 
	{
		opj_stream_destroy(l_stream);
		opj_destroy_codec(l_codec);
		opj_image_destroy(image);
		fprintf(stderr, "failed to encode image\n");
		remove(parameters.outfile);
		return;
	}

	fprintf(stdout, "[INFO] Generated outfile %s\n", parameters.outfile);
	/* close and free the byte stream */
	opj_stream_destroy(l_stream);

	/* free remaining compression structures */
	opj_destroy_codec(l_codec);

	/* free image data */
	opj_image_destroy(image);
}



