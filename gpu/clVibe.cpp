#include "clVibe.h"
#include "cl_utils.h"

#include "gpu_param.h"
#include "../u_ProcessTime.h"

const std::string vibeCLsrc = "#define GPU 1\n\n#if defined GPU\ntypedef global int* Int_ptr; \ntypedef global char* Char_ptr; \ntypedef global uchar* Uchar_ptr; \ntypedef global short* Short_PTR; \ntypedef global unsigned int* Uint_ptr; \n#else\ntypedef unsigned int* Uint_ptr; \ntypedef int* Int_ptr; \ntypedef char* Char_ptr; \ntypedef uchar* Uchar_ptr; \ntypedef short* Short_PTR; \n#endif\n\n#define randomSize 2048\n//////////////////////////////\n#define randSubSample 8\n#define nchannels 1\n//////////////////////////////\n#define N 20\n#define R 32\n#define R2 400\n#define R2f 400.0\n#define Rf 20.0\n#define cMin 2\n// SHADOW Parameters\n#define SH_MAX_CORRELATION 0.8\n#define SH_MIN_ALFA 0.4\n#define SH_MAX_ALFA 1.0\n#define SH_STDC 0.15\n\n#ifndef GPU\n#include <omp.h>\nint thID = -1;\nvoid set_threadId(int th)\n{\n\tthID = th;\n}\nint get_global_id(int dim)\n{\n\tif (thID>0)\n\t\treturn thID;\n\telse\n\t\treturn omp_get_thread_num();\n}\n#endif\n\n\n\n\n#ifdef GPU\nkernel\n#endif\nvoid kernelInitialization(Uchar_ptr samples, Uchar_ptr input, int width, int height, Uchar_ptr blinkingMap)\n{\n\tlong big = width*height;\n\n\tint index = get_global_id(0);\n\tint x = index % width;\n\tint y = index / width;\n\n\tif ((x<width) && (y<height)) {\n\t\tfor (int subS = 0; subS<N; subS++) {\n\t\t\tlong base = big * subS;\n\t\t\tbase += y * width;\n\t\t\tbase += x;\n\t\t\tsamples[base] = input[y * width + x];\n\n\t\t\tblinkingMap[y*width + x] = 0;\n\t\t}\n\n\t}\n\n}\n\n\n\n#ifdef GPU\nkernel\n#endif\nvoid erode(Uchar_ptr input, Uchar_ptr output, int width, int height, int radius) {\n\t//Closing(3), filter BLOBs area < 15, Fill holes\n\tint index = get_global_id(0);\n\t\n\tif (index>= width * height) return;\n\tint minI = 255;\n\t\n\tint i = index % width;\n\tint j = index / width;\n\n\tfor (int n = -radius; n < radius; n++)\n\t\tfor (int m = -radius; m < radius; m++)\n\t\t\tif ((j + n >= 0) && (j + n < height) && (i + m >= 0) && (i + m < width))\n\t\t\t{\n\t\t\t\tint colorI = input[(j + n)*width + (i + m) + 0];\n\t\t\t\tminI = min(colorI, minI);\n\t\t\t};\n\toutput[index] = minI;\n}\n#ifdef GPU\nkernel\n#endif\nvoid dilate(Uchar_ptr input, Uchar_ptr output, int width, int height, int radius) {\n\t//Closing(3), filter BLOBs area < 15, Fill holes\n\tint maxI = 0;\n\tint index = get_global_id(0);\n\tint i = index % width;\n\tint j = index / width;\n\n\tfor (int n = -radius; n < radius; n++)\n\t\tfor (int m = -radius; m < radius; m++)\n\t\t\tif ((j + n >= 0) && (j + n < height) && (i + m >= 0) && (i + m < width))\n\t\t\t{\n\t\t\t\tint colorI = input[(j + n)*width + (i + m) + 0];\n\t\t\t\tmaxI = max(colorI, maxI);\n\t\t\t};\n\toutput[index] = maxI;\n}\n\n#ifdef GPU\nkernel\n#endif\nvoid postProcess(Uchar_ptr input, Uchar_ptr output, int width, int height) {\n\t//Closing(3), filter BLOBs area < 15, Fill holes\n\terode(input, output, width, height, 3);\n\tdilate(input, output, width, height, 3);\n\n}\n\n#ifdef GPU\nkernel\n#endif\nvoid copyBuffer(Uchar_ptr orig, Uchar_ptr dest, int width) {\n\tint index = get_global_id(0);\n\tint x = index % width;\n\tint y = index / width;\n\n\tdest[y*width + x] = orig[y*width + x];\n}\n\n\n#ifdef GPU\nkernel\n#endif\nvoid blinking(Uchar_ptr bf, Uchar_ptr segmentPrevia, Uchar_ptr segmentActual, Uchar_ptr blinkingMap, Uchar_ptr Update\n\t, int width, int height) {\n\tint index = get_global_id(0);\n\tint x = index % width;\n\tint y = index / width;\n\n\n\tif (!((x == 0) || (y == 0) || (x == width - 1) || (y == height - 1))) {\n\n\t\t//Actualizar nivel de blinkeo\n\t\tif (bf[y * width + x] == 0 //Clasificado como Background\n\t\t\t&&\n\t\t\tsegmentPrevia[y * width + x] != segmentActual[y * width + x] //Blinkeo\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t //Inner border of BG?\n\t\t\t&& ((bf[(y - 1) * width + x - 1] != 0) || (bf[y * width + x - 1] != 0) || (bf[(y - 1) * width + x] != 0) ||\n\t\t\t\t(bf[y * width + x + 1] != 0) || (bf[(y + 1) * width + x] != 0) || (bf[(y + 1) * width + x + 1] != 0) ||\n\t\t\t\t(bf[(y - 1) * width + x + 1] != 0) || (bf[(y + 1) * width + x - 1] != 0))) {\n\n\t\t\tif (blinkingMap[y * width + x] <= 135)\n\t\t\t\tblinkingMap[y * width + x] += 15;\n\n\t\t}\n\t\telse {\n\t\t\tif (blinkingMap[y * width + x] > 0)\n\t\t\t\tblinkingMap[y * width + x] -= 1;\n\t\t}\n\n\t\t//Actualizar modelo si no esta blinkeando y fue clasificado como Background\n\t\tif ((blinkingMap[y * width + x] < 30) && (bf[y * width + x] == 0)) {\n\t\t\t//ACTUALIZAR EL MODELO ! probReemplzo\n\t\t\tUpdate[y * width + x] = 0;\n\t\t\t//PROPAGAR !\n\t\t}\n\t\telse {\n\t\t\t//NO ACTUALIZAR EL MODELO\n\t\t\tUpdate[y * width + x] = 255;\n\t\t}\n\t}\n\telse {\t//Borde !\n\t\tif (bf[y * width + x] == 0) {\n\t\t\t//ACTUALIZAR EL MODELO, CUIDADO CON LOS BORDES !\n\t\t\tUpdate[y * width + x] = 0;\n\t\t}\n\t\telse {\n\t\t\t//NO ACTUALIZAR EL MODELO\n\t\t\tUpdate[y * width + x] = 255;\n\t\t}\n\t}\n\n}\n\n#ifdef GPU\nkernel\n#endif\nvoid computeShadowMask(Uchar_ptr inputData, Uchar_ptr meanInputData, Uchar_ptr _bgrModelData, int width, int height,int radius)\n{\n\tint index = get_global_id(0);\n\tint i = index % width;\n\tint j = index / width;\n\n\tfloat C = 0;\n\tfloat Varianzai = 0;\n\tfloat Varianzaf = 0;\n\tfloat Varianzac = 0;\n\tfloat mediac = 0;\n\tfloat stdc = 0;\n\tfloat alfa = 0;\n\tint vecinos = 0;\n\tfloat colori = 0;\n\tfloat colorf = 0;\n\tint vecinosSh = 0;\n\n\tint limsupY = height - 1;\n\tint liminfY = 1;\n\tint limsupX = width - 1;\n\tint liminfX = 1;\n\n\tif (_bgrModelData[j * width + i + 0] != 0)\n\t{\n\t\tC = 0;\n\t\tVarianzai = 0;\n\t\tVarianzaf = 0;\n\t\tVarianzac = 0;\n\t\tmediac = 0;\n\t\tstdc = 0;\n\t\talfa = 0;\n\t\tvecinos = 0;\n\t\tcolori = 0;\n\t\tcolorf = 0; //outputData[j, i, 0] = 128;\n\t\tfor (int n = -radius; n < radius; n++)\n\t\t\tfor (int m = -radius; m < radius; m++)\n\t\t\t\tif ((j + n >= liminfY) && (j + n < limsupY) && (i + m >= liminfX) && (i + m < limsupX))\n\t\t\t\t{\n\t\t\t\t\tcolori = inputData[(j + n)*width + (i + m) + 0];\n\t\t\t\t\tcolorf = meanInputData[(j + n)*width + (i + m) + 0];\n\t\t\t\t\tC += colori * colorf;\n\t\t\t\t\tVarianzai += colori * colori;\n\t\t\t\t\tVarianzaf += colorf * colorf;\n\t\t\t\t\tVarianzac += (colori / colorf) * (colori / colorf);\n\t\t\t\t\tmediac += (colori / colorf);\n\t\t\t\t\tvecinos++;\n\t\t\t\t\tif (_bgrModelData[(j + n)*width + (i + m) + 0] != 0)\n\t\t\t\t\t\tvecinosSh++;\n\t\t\t\t};\n\n\t\tC = (C / (vecinos + 0.001)) / (sqrt(Varianzai / (vecinos + 0.001)) * sqrt(Varianzaf / (vecinos + 0.001)));\n\t\tVarianzac = Varianzac / vecinos; mediac = mediac / vecinos;\n\t\tstdc = sqrt(Varianzac - mediac * mediac);\n\t\tcolori = inputData[(j)*width + (i)+0];\n\t\tcolorf = meanInputData[(j)*width + (i)+0];\n\t\talfa = colori / colorf;\n\t\tif ((C >= SH_MAX_CORRELATION) && (Varianzai < Varianzaf) && (vecinosSh>0))\n\t\t\tif ((stdc < SH_STDC) && (alfa >= SH_MIN_ALFA) && (alfa < SH_MAX_ALFA))\n\t\t\t{\n\t\t\t\t_bgrModelData[j * width + i + 0] = 1; // Aumentar para visualizar\n\t\t\t}\n\t}; //endif mascara\n\n}; //end area\n\n\nint randomfunc2(Int_ptr randomValues, int numIteracion, int x, int y, int* iv)\n{\n\t(*iv)++;\n\treturn  randomValues[(numIteracion* x * 11 + (*iv) * 5) % 2048] + randomValues[(numIteracion *  y * 32 + (*iv) * 3) % 1024];\n\n}\n\n#define RADIUS_N 7\n\n\n#ifdef GPU\nkernel\n#endif\nvoid detect(Uchar_ptr samples, Uchar_ptr input, Uchar_ptr bf, int width, int height)\n{\n\tint big = width*height;\n\n\tint index = get_global_id(0);\n\t\n\tif (index >= width * height) return;\n\t\n\tint x = index % width;\n\tint y = index / width;\n\n\tint count = 0;\n\tint ind = 0;\n\tint I0i = input[index];\n\n\twhile ((count<cMin) && (ind<N)) \n\t{\n\n\t\tint base = big * ind + index;\n\t\t\t\t\n\t\tint S0i = samples[base];\n\n\t\tif (abs(I0i - S0i) < R)\n\t\t{\n\t\t\tcount++;\n\t\t}\n\n\t\tind++;\n\t}\n\tif (count >= cMin)\n\t\tbf[index] = 0;//BACKGROUND\n\telse\n\t\tbf[index] = 255;//FOREGROUND\n\n\n}\n\n#ifdef GPU\nkernel\n#endif\nvoid update_model(Uchar_ptr input, Uchar_ptr samples, Uchar_ptr bf, int width, int height,\n\tint numIteracion, Int_ptr randomValues)\n{\n\tint index = get_global_id(0);\n\t\n\tif (index >= width * height) return;\n\tint x = index % width;\n\tint y = index / width;\n\n\tint big = width*height;\n\tint X_off; // [9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };\n\tint Y_off;// [9] = { -1, -1, -1, 1, 1, 1, 0, 0, 0 };\n\tint invocation = 0;\n\n     \n    if (numIteracion<N)\n\t{\n\t    int base =  width*height * numIteracion + index;\n\t\tsamples[base] = input[index];\n\t\treturn;\n\t}\n\t\n\t\t\n\t//Si fue considerado background\n\tif (bf[index] == 0)\n\t{\n \n        int random = randomfunc2(randomValues, numIteracion, x, y, &invocation) % randSubSample;\n\t\n\t\tif (random == 0)\n\t\t{\n\t\t\tint randSubS = numIteracion % N;\n\n\t\t\tint base = width*height * randSubS + index;\n\t\t\tsamples[base] =  input[index];\n\t\t\t\n\t\t}\n\n\t\t//Propagar si esta fuera del borde\n\t\tif ((x>0) && (y>0) && (x<width - 1) && (y<height - 1))\n\t\t{\n\t\t\t//Propagar a algun vecino!\n\t\t\tint random = randomfunc2(randomValues, numIteracion, x, y, &invocation) % randSubSample;\n\n\t\t\tif (random == 0)\n\t\t\t{\n\t\t\t\t//Random de los N samples\n\t\t\t\tint randSubP = (numIteracion + 1) % N;\n\t\t\t\t\n\n\t\t\t\t//Random {X,Y}\n\t\t\t\tint indexR = randomfunc2(randomValues, numIteracion, x, y, &invocation) % (RADIUS_N* RADIUS_N);\n\n\t\t\t\tY_off = (indexR / RADIUS_N) - (RADIUS_N/2);\n\t\t\t\tint Yr = y + Y_off;\n\n\t\t\t\tindexR = randomfunc2(randomValues, numIteracion, x, y, &invocation) % (RADIUS_N* RADIUS_N);\n\t\t\t\t\n\t\t\t\tX_off == (indexR % RADIUS_N) - (RADIUS_N / 2);\n\n\t\t\t\tint Xr = x + X_off;\n\n\t\t\t\tif ((Xr >= 0) && (Yr >= 0) && (Xr<width) && (Yr<height))\n\t\t\t\t{\n\t\t\t\t\tint baseP = big * randSubP + Yr * width + Xr;\n\t\t\t\t\tif ((baseP>=0) && (baseP <  big * N))\n\t\t\t\t\t{\n\t\t\t\t\t  samples[baseP] = input[Yr * width + Xr];\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\n\n\t}\n\n\n}\n\n";


void pixelBWCount(cv::Mat& src, int &count_white, int &count_black)
{
	count_black = 0;
	count_white = 0;
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {

			// change this to to 'src.atuchar>(y,x) == 255' 
			// if your img has only 1 channel
			if (src.at<uchar>(y, x) == 255) {
				count_white++;
			}
			else if (src.at<uchar>(y, x) == 0) {
				count_black++;
			}
		}
	}

}

void loadDataMatToUchar(uchar *data, cv::Mat &image, int nchannels)
{
	int width = image.cols;
	int height = image.rows;
	//#pragma omp parallel for
	for (int y = 0; y<height; y++)
	{
		for (int x = 0; x<width; x++)
		{
			data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 0] = image.data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 0];
			if (nchannels == 3) {
				data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 1] = image.data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 1];
				data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 2] = image.data[(long)y * (long)width * (long)nchannels + (long)x*nchannels + 2];
			}
		}
	}
}

cl_bgs::cl_bgs(string cl_prog, int localWG)
{
	numFrame = 0;
}


cl_bgs::cl_bgs()
{
	numFrame = 0;
}


int cl_bgs::init(std::string cl_prog, int localWG, int platformProcessingIndex , int deviceProcessingIndex )
{
	int error = CL_SUCCESS;
	vibeProgram = clUtils::loadCLSources(cl_prog, localWG, platformProcessingIndex, deviceProcessingIndex,&error);
	clCodeLoaded = 1;
	return error;
}
//void execute(cl::Context &context,cl::Program &program,cl::CommandQueue &queue);
void cl_bgs::executeRGB(cl::Context &context, cl::Program &program, cl::CommandQueue &queue)
{}
void cl_bgs::operate(cv::Mat& inputFrame, cv::Mat& bfFrame, bool fillContours)
{
	startProcess((char*)"clVibe");
	clUtils::setActiveProgram(vibeProgram);
	///Next image
	if (inputFrame.channels() == 3)
	{
		cvtColor(inputFrame, inputFrame, CV_BGR2GRAY);
	}
	loadDataMatToUchar(input, inputFrame, NCHANNELS);
		
	gpuBuffer* gsamples = gpuMemHandler::getBuffer(hsamples,true,CL_MEM_READ_WRITE,1);
	gpuBuffer* ginput = gpuMemHandler::getBuffer(input, true, CL_MEM_READ_WRITE, 1);
	gpuBuffer* gbf = gpuMemHandler::getBuffer(background, true, CL_MEM_READ_WRITE, 1);
	gpuBuffer* gRand = gpuMemHandler::getBuffer(buffRandom, true, CL_MEM_READ_WRITE, 1);
	gpuBuffer* gTemp = gpuMemHandler::getBuffer(tempBuffer, true, CL_MEM_READ_WRITE, 1);

	int error = gpuMemHandler::WriteBuffer(ginput, true, 0, true);

	clUtils::assertCL(error);

	size_t input_size = inputFrame.rows * inputFrame.cols;

	kernelCall("detect", clUtils::opencl_gridSize(input_size, 256), 256, { hsamples  , input , background , inputFrame.cols, inputFrame.rows }, {background });
	
	
	if (numFrame % updateStep == 0)
	{
		kernelCall("update_model", clUtils::opencl_gridSize(input_size, 256), 256, { input  , hsamples,background ,  inputFrame.cols, inputFrame.rows, numFrame,buffRandom }, {  });
	}

	
	//kernelCall("dilate", clUtils::opencl_gridSize(input_size, 256), 256, { background  , tempBuffer ,  inputFrame.cols, inputFrame.rows,3 }, {});
	//kernelCall("dilate", clUtils::opencl_gridSize(input_size, 256), 256, { tempBuffer  , background ,  inputFrame.cols, inputFrame.rows,3 }, {});


	//kernelCall("erode", clUtils::opencl_gridSize(input_size, 256), 256, { background  , tempBuffer ,  inputFrame.cols, inputFrame.rows,3 }, {});
	//kernelCall("erode", clUtils::opencl_gridSize(input_size, 256), 256, { tempBuffer  , background ,  inputFrame.cols, inputFrame.rows,3 }, {});

	// Finally, get data	
	error = gpuMemHandler::ReadBuffer(gbf, true, 0, true);

	clUtils::assertCL(error);

	///Pasar de GPU a CPU
	bfFrame.data = background;

	endProcess((char*)"clVibe");
	numFrame++;

}
void cl_bgs::computeShadowMask(cv::Mat& input, cv::Mat &meanInput, cv::Mat& bFrame)
{}
void cl_bgs::initialize(std::string srcCL,  cv::Mat &inputFrame, int platformProcessingIndex, int deviceProcessingIndex)
{
	int error = 0;

	if (srcCL == "")
	{
		std::string src(vibeCLsrc);
		clUtils::setCLSource(src);
		vibeProgram = clUtils::buildProgram(clUtils::getDefaultContext(), clUtils::getDefaultDevice(), &error);
	}
	else
	{

		vibeProgram = clUtils::loadCLSources(srcCL, 256, platformProcessingIndex, deviceProcessingIndex, &error);
	}

	clUtils::assertCL(error);

	if (inputFrame.channels() == 3)
		cvtColor(inputFrame, inputFrame, CV_BGR2GRAY);


	buffRandom = new int[randomSize];
	for (int i = 0; i<randomSize; i++)
	{
		int r = rand();
		buffRandom[i] = r % randomSize;
	}

	width = inputFrame.cols; cl_utils::width = width;
	height = inputFrame.rows; cl_utils::height = height;
	//    nchannels = inputFrame.channels();
	//	visualize::nchannels=nchannels;
	// N=20;
	//	visualize::N=N;
	//long lwidth = width;long lheight=height;

	//Por las dudas si truncaba
	wh = width * height;
	whchann = wh * NCHANNELS;
	whNchann = whchann * N;


	cout << "Num channels " << NCHANNELS << endl;

	input = new uchar[whchann];
	hsamples = new uchar[whNchann];
	for (int i = 0; i < whNchann;i++) hsamples[i] = 0;

	background = new uchar[whchann]; // (width + 2)*(height + 2)];
	tempBuffer = new uchar[whchann];
	loadDataMatToUchar(input, inputFrame, NCHANNELS);

	numFrame = 0;
}
cv::Mat cl_bgs::getMeanBack(cv::Mat& inputFrame)
{



	return meanFrame;
}
cv::Mat cl_bgs::getSample(cv::Mat& inputFrame, int index)
{

	cv::Mat mF(inputFrame.rows, inputFrame.cols, CV_8UC1);

	int offset = width * height * index;

	mF.data = &hsamples[ offset];
	
	return mF;
}
void cl_bgs::getBack(cv::Mat& bfFrame)
{
	gpuBuffer* gbf = gpuMemHandler::getBuffer(background);
	gpuMemHandler::ReadBuffer(gbf, true, 0, true);
	bfFrame.data = background;

}