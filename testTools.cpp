
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>  
#include <algorithm> 
#include <fstream>
#include <vector>
#include <map>
#include "testTools.h"
#include "bitstream.hpp"
#ifdef ZSTD
#include "jp2CPUEncoder.h"
#endif


splineCompression* splLinear = NULL;
splineCompression* splBiQubic = NULL;
splineCompression* splLossLess = NULL;
splineCompression* spl5 = NULL;


bool splineSort(spline_data* sp0, spline_data* sp1)
{
	return ((sp0->y0 * 2000 + sp0->x0) < (sp1->y0 * 2000 + sp1->x0));
}


std::vector<std::string> getAllFilesInDir(const std::string& dirPath, std::vector<std::string> extensions)
{
	std::vector<std::string> listOfFiles;

	//cout << "Open dir: "<< dirPath << endl;
	// Create a vector of string

	try
	{
		// Check if given path exists and points to a directory
		if (std::experimental::filesystem::exists(dirPath) && std::experimental::filesystem::is_directory(dirPath))
		{
			// Create a Recursive Directory Iterator object and points to the starting of directory
			std::experimental::filesystem::recursive_directory_iterator iter(dirPath);

			// Create a Recursive Directory Iterator object pointing to end.
			std::experimental::filesystem::recursive_directory_iterator end;

			int index = 0;
			// Iterate till end
			while (iter != end)
			{
				std::string ext = iter->path().extension().string();
				bool isAValidExtension = false;
				for (auto e : extensions)
				{
					if (ext == e)
					{
						isAValidExtension = true;
						break;
					}
				}

				if (isAValidExtension)
				{
					listOfFiles.push_back(iter->path().string());
				}

				index++;
				std::error_code ec;
				// Increment the iterator to point to next entry in recursive iteration

				iter.increment(ec);
				if (ec)
				{
					std::cerr << "Error While Accessing : " << iter->path().string() << " :: " << ec.message() << '\n';
				}
			}
		}
	}
	catch (std::system_error& e)
	{
		std::cerr << "Exception :: " << e.what();
	}

	return listOfFiles;
}




void histDisplay(std::vector<double>& histogram, const char* name)
{
	//std::cout<<"Start HistDisplay" << std::endl;

	if (histogram.size() < 10) return;
	std::vector<double> hist;
	cv::Mat histImg;
	histImg.create(cv::Size(600, 150), CV_8UC3);

	hist.clear();

	for (int i = 0; i < histogram.size(); i = i + 1)
	{
		hist.push_back(histogram[i]);
	}
	// draw the histograms
	int hist_w = histogram.size(); int hist_h = histImg.rows;
	int bin_w = 1;

	histImg.setTo(cv::Scalar(255, 255, 255));
	// find the maximum intensity element from histogram
	double max = hist[0];
	double min = 10000.0;
	for (int i = 1; i < histogram.size(); i++)
	{
		if (max < hist[i])
		{
			max = hist[i];
		}

		if (min > hist[i])
		{
			min = hist[i];
		}
	}

	double minReal = min;

	min = 0.0;
	// normalize the histogram between 0 and histImage.rows
	for (int i = 0; i < histogram.size(); i = i + 1)
	{
		hist[i] = ((double)(hist[i] - min) / (max - min))*histImg.rows;
	}


	// draw the intensity line for histogram
	for (int i = 0; i < histogram.size(); i = i + 1)
	{
		int x = (i * histImg.cols) / histogram.size();

		line(histImg, cv::Point(x, hist_h), cv::Point(x, hist_h - hist[i]), cv::Scalar(0, 0, 0), 1, 8, 0);
	}

	double pos0 = (0 - min) / (max - min)*histImg.rows;
	line(histImg, cv::Point(0, pos0), cv::Point(histImg.cols - 1, pos0), cv::Scalar(0, 0, 255), 1, 8, 0);

	cv::putText(histImg, "max:" + std::to_string(max), cv::Point(20, 20), 1, 0.8, cv::Scalar(255, 0, 0));
	cv::putText(histImg, "min:" + std::to_string(minReal), cv::Point(20, 40), 1, 0.8, cv::Scalar(255, 0, 0));


	// display histogram
//	namedWindow(name, CV_WINDOW_AUTOSIZE);

	cv::imshow(name, histImg);
	//std::cout << "End HistDisplay" << std::endl;
}
/////////////////////////////////////////////////
// Extract metrics comparing to J2K compression
void compareToJ2K(std::string outDir)
{
	if (!splLinear)  splLinear = new splineCompression(LINEAR_COMPRESSION);
	if (!splBiQubic)  splBiQubic = new splineCompression(SPLINE_COMPRESSION);
	if (!splLossLess) splLossLess = new splineCompression(LOSSLESS_COMPRESSION);
	if (!spl5)  spl5 = new splineCompression(SPLINE5_COMPRESSION);

	std::vector<double> r0;
	std::vector<double> r1;

	for (int i = 0; i < 150; i++)
	{
		std::string fRaw = outDir + std::to_string(i) + ".pgm";
		std::string fJ2K85 = outDir + std::to_string(i) + ".q85.pgm";
		std::string fJ2K95 = outDir + std::to_string(i) + ".q95.pgm";
		std::string fCubic = outDir + "Cubic\\" + std::to_string(i) + ".bin";
		std::string fLinear = outDir + "Linear\\" + std::to_string(i) + ".bin";
		std::string fFive = outDir + "Five\\" + std::to_string(i) + ".bin";

		cv::Mat mRaw = cv::imread(fRaw, -1);
		cv::Mat mJ2K85 = cv::imread(fJ2K85, -1);
		cv::Mat mJ2K95 = cv::imread(fJ2K95, -1);
		cv::Mat mCubic = splBiQubic->decode(fCubic);
		cv::Mat mFive = spl5->decode(fFive);

		cv::Mat mLinear = splLinear->decode(fLinear);

		// Update the window with new data


		cv::Mat im_color, im_gray;
		mRaw.convertTo(im_gray, CV_8UC1, 1.0 / 160.0);
		cv::applyColorMap(im_gray, im_color, cv::COLORMAP_HSV);

		cv::Mat im_colorR2, im_grayR2;
		mCubic.convertTo(im_grayR2, CV_8UC1, 1.0 / 160.0);
		cv::applyColorMap(im_grayR2, im_colorR2, cv::COLORMAP_HSV);

		cv::Mat im_colorR3, im_grayR3;
		mJ2K95.convertTo(im_grayR3, CV_8UC1, 1.0 / 160.0);
		cv::applyColorMap(im_grayR3, im_colorR3, cv::COLORMAP_HSV);

		double pp = qm::psnr(mRaw, mCubic, 1);
		double ppL = qm::psnr(mRaw, mLinear, 1);
		double pp5 = qm::psnr(mRaw, mFive, 1);

		double pp85 = qm::psnr(mRaw, mJ2K85, 1);
		double pp95 = qm::psnr(mRaw, mJ2K95, 1);

		cv::putText(im_colorR2, std::to_string(pp), cv::Point(20, 20), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_colorR2, std::to_string(ppL), cv::Point(20, 50), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_colorR2, std::to_string(pp5), cv::Point(20, 80), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_colorR2, "J2k:" + std::to_string(pp85), cv::Point(20, 110), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_colorR2, "J2k:" + std::to_string(pp95), cv::Point(20, 140), 1, 1, cv::Scalar(255, 255, 255));




		cv::imshow("RAW", im_color);
		//	cv::imshow("J2K", mJ2K);
		cv::imshow("CUBIC", im_colorR2);
		cv::imshow("J2K95", im_colorR3);

		r0.push_back(pp);
		r1.push_back(pp95);

		histDisplay(r0, "hist0");
		histDisplay(r1, "hist1");

		cv::waitKey(50);


	}

	showProcessTime();

	cv::waitKey(-1);


}




void compressionMetrics(cv::Mat& depth, std::string outDir, int frame, bool residual, bool verbose, int minElementsCount)
{

	if (!splLinear) 
	{
		splLinear = new splineCompression(LINEAR_COMPRESSION); splLinear->lonelyPixelsRemoval = minElementsCount;}
	if (!splBiQubic)  
	{ splBiQubic = new splineCompression(SPLINE_COMPRESSION); splBiQubic->lonelyPixelsRemoval = minElementsCount;
	}	if (!splLossLess) 
	{ splLossLess = new splineCompression(LOSSLESS_COMPRESSION); splLossLess->lonelyPixelsRemoval = minElementsCount;
	}
	if (!spl5)  
	{ 
		spl5 = new splineCompression(SPLINE5_COMPRESSION); spl5->lonelyPixelsRemoval = minElementsCount;
	}

	uint32_t width = depth.cols;
	uint32_t height = depth.rows;
	size_t orig_size = width * height * 2;

	std::vector<float> valuesToExport;

	if (!dirExists(outDir + "ZSTD//")) createDirectory(outDir + "ZSTD//");
	if (!dirExists(outDir + "Linear//")) createDirectory(outDir + "Linear//");
	if (!dirExists(outDir + "Cubic//")) createDirectory(outDir + "Cubic//");
	if (!dirExists(outDir + "Five//")) createDirectory(outDir + "Five//");
	if (!dirExists(outDir + "PNG//")) createDirectory(outDir + "PNG//");
	if (!dirExists(outDir + "J2K//")) createDirectory(outDir + "J2K//");
	if (!dirExists(outDir + "RAW//")) createDirectory(outDir + "RAW//");



	splBiQubic->saveResidual = residual;
	if (residual) splBiQubic->zstd_compression_level = 1;
	else   splBiQubic->zstd_compression_level = 0;

	splLinear->saveResidual = residual;
	if (residual) splLinear->zstd_compression_level = 1;
	else   splLinear->zstd_compression_level = 0;

	spl5->saveResidual = residual;
	if (residual) spl5->zstd_compression_level = 1;
	else   spl5->zstd_compression_level = 0;

	splLossLess->zstd_compression_level = 1;

	// ZSTD
	auto start = high_resolution_clock::now();
	double compZ = (double)splLossLess->encode(depth, outDir + "ZSTD//" + std::to_string(frame) + ".bin") / (orig_size);

	auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedZ = duration.count();

	// LINEAR
	start = high_resolution_clock::now();
	double compL = (double)splLinear->encode(depth, outDir + "Linear//" + std::to_string(frame) + ".bin") / (orig_size);

	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedL = duration.count();

	std::cout << "Quantization  " << splLinear->quantization << "\n";


	// BIQUBIC
	start = high_resolution_clock::now();
	double compQ = (double)splBiQubic->encode(depth, outDir + "Cubic//" + std::to_string(frame) + ".bin") / (orig_size);

	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedQ = duration.count();
	//bitS->readFromFile("d://temp//l515//ZSTD.bin");


	// FIVE
	start = high_resolution_clock::now();
	double compQ5 = (double)spl5->encode(depth, outDir + "Five//" + std::to_string(frame) + ".bin") / (orig_size);

	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedQ5 = duration.count();
	//PNG
	start = high_resolution_clock::now();
	cv::imwrite(outDir + "PNG//" + std::to_string(frame) + ".png", depth);
	double compPNG = (double)get_file_size(outDir + "PNG//" + std::to_string(frame) + ".png") / (orig_size);
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);

	double elapsedPNG = duration.count();

	double compJ2KL = 0;
#ifdef ZSTD
	//J2K LossLess
	start = high_resolution_clock::now();
	writeJP2File(outDir + "J2K//" + std::to_string(frame) + ".jp2", depth);
	compJ2KL = (double)get_file_size(outDir + "J2K//" + std::to_string(frame) + ".jp2") / (orig_size);
#endif	
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double elapsedJ2K = duration.count();

	start = high_resolution_clock::now();
	cv::Mat depthRestored = splLossLess->restoreAsImage(true);
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double decodeZ = duration.count();


	cv::Mat depthRestoredLinear, depthRestoredCubic, depthRestored5, depth2;
	double decodeLinear, decodeCubic, decode5, decodePNG;

	//if (residual)
	{
		start = high_resolution_clock::now();
		depthRestoredLinear = splLinear->restoreAsImage();//  splLinear->decode(outDir + "Linear//" + std::to_string(frame) + ".bin");
		duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
		decodeLinear = duration.count();

		start = high_resolution_clock::now();
		depthRestoredCubic = splBiQubic->restoreAsImage();//   splBiQubic->decode(outDir + "Cubic//" + std::to_string(frame) + ".bin");
		duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
		decodeCubic = duration.count();

		start = high_resolution_clock::now();
		depthRestored5 = spl5->restoreAsImage();// spl5->decode(outDir + "Five//" + std::to_string(frame) + ".bin");
		duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
		decode5 = duration.count();
	}

	start = high_resolution_clock::now();
	depth2 = cv::imread(outDir + "PNG//" + std::to_string(frame) + ".png", -1);
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	decodePNG = duration.count();

	start = high_resolution_clock::now();
	cv::Mat depthJP2 = cv::imread(outDir + "J2K//" + std::to_string(frame) + ".jp2", -1);
	duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	double decodeJP2 = duration.count();


	std::string fn = outDir + "RAW//" + std::to_string(frame) + ".raw";
	std::ofstream out(fn, std::ios::out | std::ios::binary);
	// bin mask
	out.write((const char*)depth.data, orig_size);
	out.close();


	////////////////////////////////////////////////////////////////////////
	fn = outDir + "RAW//" + std::to_string(frame) + ".pgm";
	cv::imwrite(fn, depth);

	// Compute Error
	double error = qm::psnr(depth, depthRestored, 1);
	double errorL = qm::psnr(depth, depthRestoredLinear, 1);
	double errorB = qm::psnr(depth, depthRestoredCubic, 1);
	double error5 = qm::psnr(depth, depthRestored5, 1);


	if (verbose)
	{
		std::cout << "-------------------------------------------" << "\n";
		if (residual) std::cout << "   RESIDUAL  " << "\n";
		std::cout << " LINEAR e: " << elapsedL << " ms. d:" << decodeLinear<<  " psnr " << errorL << " rate " << compL << "\n";
		std::cout << " CUBIC e:" << elapsedQ << " ms. d:" << decodeCubic << " psnr " << errorB <<   " rate " << compQ << "\n";
		std::cout << " LOSSLESS e:" << elapsedZ << " ms. d:" << decodeZ << " psnr " << error << " rate " << compZ << "\n";
		std::cout << " PNG e:"  << elapsedPNG << " ms. d:" << decodePNG << " psnr " << error << " rate " << compPNG << "\n";
		std::cout << " J2K e:" << elapsedJ2K << " ms. d:" << decodeJP2 << " psnr " << error << " rate " << compJ2KL << "\n";

		
		std::cout << "-------------------------------------------" << "\n";
	}


	valuesToExport.push_back(compZ);
	valuesToExport.push_back(error);
	valuesToExport.push_back(elapsedZ);
	valuesToExport.push_back(decodeZ);

	valuesToExport.push_back(compL);
	valuesToExport.push_back(errorL);
	valuesToExport.push_back(elapsedL);
	valuesToExport.push_back(decodeLinear);

	valuesToExport.push_back(compQ);
	valuesToExport.push_back(errorB);
	valuesToExport.push_back(elapsedQ);
	valuesToExport.push_back(decodeCubic);

	valuesToExport.push_back(compPNG);
	valuesToExport.push_back(0);
	valuesToExport.push_back(elapsedPNG);
	valuesToExport.push_back(decodePNG);

	valuesToExport.push_back(compQ5);
	valuesToExport.push_back(error5);
	valuesToExport.push_back(elapsedQ5);
	valuesToExport.push_back(decode5);

	valuesToExport.push_back(compJ2KL);
	valuesToExport.push_back(0);
	valuesToExport.push_back(elapsedJ2K);
	valuesToExport.push_back(decodeJP2);

	exportCSV(outDir + "data.csv", valuesToExport, frame);

	splBiQubic->memMgr.releaseAll();
	splLinear->memMgr.releaseAll();
	splLossLess->memMgr.releaseAll();
	spl5->memMgr.releaseAll();
}

cv::Mat calculateResidualAbs(cv::Mat& m, cv::Mat& depthRestored, int quantization, size_t outSizeC)
{

	char* diff = new char[m.cols * m.rows];
	uchar* output = new uchar[m.cols * m.rows * 2];
	cv::Mat depthRestoredPlus;
	//////////////////////////////////////////////////
	/// Compute Residual
	unsigned short* pix0 = (unsigned short*)m.data;
	unsigned short* pix1 = (unsigned short*)depthRestored.data;
	double minV = 1000;
	double maxV = -1000;

	int quantizationLevel = pow(2, quantization);

	for (int y = 0; y < m.rows; y++)
		for (int x = 0; x < m.cols; x++)
		{
			diff[y * m.cols + x] = (pix0[y * m.cols + x] - pix1[y * m.cols + x]) / (quantizationLevel);

		}

#ifdef ZSTD
	size_t cBuffSize = ZSTD_compressBound(m.cols * m.rows * 2);

	size_t outSize = ZSTD_compress(output, cBuffSize, diff, m.cols * m.rows, 9);
#else
	size_t cBuffSize = m.cols * m.rows * 2;
	size_t outSize = cBuffSize;
#endif
	/////////////////////////////////////////////////////////////
	depthRestoredPlus = m.clone();
	depthRestoredPlus.setTo(0);

	unsigned short* rpix1 = (unsigned short*)depthRestored.data;
	unsigned short* rpix2 = (unsigned short*)depthRestoredPlus.data;

	for (int y = 0; y < m.rows; y++)
		for (int x = 0; x < m.cols; x++)
		{
			rpix2[y * m.cols + x] = diff[y * m.cols + x] * quantizationLevel + rpix1[y * m.cols + x];

		}

	double errorPlus = qm::psnr(m, depthRestoredPlus, 1);
	std::cout << "Compression error + Residual ABS .  Error2 " << errorPlus << " Size." << outSizeC << " residual size " << +outSize << "\n";

	return depthRestoredPlus;
}



void createScripts(std::string outDir, int frame)
{

	std::string fn = outDir + "scripts_lossless_encodeJ2K.bat";
	std::string sc0 = "fast_j2k.exe -i " + std::to_string(frame) + ".pgm -o " + std::to_string(frame) + ".lossless.jp2 -a REV -l 7 -c 32";
	std::ofstream myfile(fn, std::ios::out | std::ios::app);
	myfile << sc0 << "\n";
	myfile.close();

	// Lossy encode 85 Q
	fn = outDir + "scripts_lossy_encodeJ2K_85.bat";
	std::string sc1 = "fast_j2k.exe -i " + std::to_string(frame) + ".pgm -o " + std::to_string(frame) + ".q85.jp2 -a IRREV -l 7 -c 32 -q 85";
	std::ofstream myfile2(fn, std::ios::out | std::ios::app);
	myfile2 << sc1 << "\n";
	myfile2.close();


	// Lossy encode 95 Q
	fn = outDir + "scripts_lossy_encodeJ2K_95.bat";
	std::string sc2 = "fast_j2k.exe -i " + std::to_string(frame) + ".pgm -o " + std::to_string(frame) + ".q95.jp2 -a IRREV -l 7 -c 32 -q 95";
	std::ofstream myfile3(fn, std::ios::out | std::ios::app);
	myfile3 << sc2 << "\n";
	myfile3.close();

	// Lossy decode 85 Q
	fn = outDir + "scripts_lossy_decodeJ2K_85.bat";
	std::string dec0 = "fast_j2k.exe -i " + std::to_string(frame) + ".q85.jp2 -o " + std::to_string(frame) + ".q85.pgm";
	std::ofstream myfile4(fn, std::ios::out | std::ios::app);
	myfile4 << dec0 << "\n";
	myfile4.close();

	// Lossy decode 95 Q
	fn = outDir + "scripts_lossy_decodeJ2K_95.bat";
	std::string dec1 = "fast_j2k.exe -i " + std::to_string(frame) + ".q95.jp2 -o " + std::to_string(frame) + ".q95.pgm";
	std::ofstream myfile5(fn, std::ios::out | std::ios::app);
	myfile5 << dec1 << "\n";
	myfile5.close();






	myfile << sc1 << "\n";
	myfile << sc2 << "\n";

	myfile << dec0 << "\n";
	myfile << dec1 << "\n";



}


int test1(std::string inputDir, std::string iformat)
{
	int mode = SPLINE_COMPRESSION;
	splineCompression* splLinear = new splineCompression(mode);

	unsigned char* outputC = new unsigned char[1000 * 1000 * 2];

	splLinear->saveResidual = true;

	splineCompression* splNoDictionary = new splineCompression(mode);
	splNoDictionary->saveResidual = true;

	int scanrow = 250;
	int i = 0;

	int iterCount = 30;

	int quantization = 7;

	while (i < 100)
	{
		std::string s = inputDir + std::to_string(i) + iformat;
		std::cout << s << "\n";
		cv::Mat m = cv::imread(s, -1);

		if (m.cols == 0) {
			i++;
			continue;
		}

		cv::Mat color_mat;
		constexpr double scaling = static_cast<double>(std::numeric_limits<uint8_t>::max()) / static_cast<double>(std::numeric_limits<uint16_t>::max());
		m.convertTo(color_mat, CV_8U, scaling);


		std::vector<float> valuesToExport;


		auto start = high_resolution_clock::now();

		startProcess("encodeDIC");

		size_t outSizeC = splLinear->encode(m, "D:\\SDKs\\zstd-1.4\\zstd\\build\\VS2010\\bin\\x64_Release\\test" + std::to_string(i) + "D.bin");

		double timeDic = endProcess("encodeDIC");

		auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
		double elapsedZ = duration.count();


		startProcess("encodeNODIC");

		size_t outSizeNoC = splNoDictionary->encode(m, "D:\\SDKs\\zstd-1.4\\zstd\\build\\VS2010\\bin\\x64_Release\\test" + std::to_string(i) + "N.bin");

		double timeNoDic = endProcess("encodeNODIC");


		valuesToExport.push_back(outSizeC);
		valuesToExport.push_back(timeDic);

		valuesToExport.push_back(outSizeNoC);
		valuesToExport.push_back(timeNoDic);

		////////////// Render
		splLinear->display(scanrow, mode, iterCount);



		cv::Mat depthRestored = splLinear->restoreAsImage();

		startProcess("savePNG");

		cv::imwrite("out.png", m);

		endProcess("savePNG");

		double error = qm::psnr(m, depthRestored, 1);

		std::cout << "-------------------------------------" << "\n";
		if (splLinear->saveResidual) std::cout << " ***** SAVING RESIDUAL ***** " << "\n";
		std::cout << "Quantization  " << splLinear->quantization << "\n";
		std::cout << " PSNR " << error << " time " << elapsedZ << "\n";
		std::cout << " Compression size " << outSizeC << "\n";

		std::cout << "-------------------------------------" << "\n";
		splLinear->computeMetrics();

		cv::line(m, cv::Point(0, scanrow), cv::Point(depthRestored.cols - 1, scanrow), cv::Scalar(255, 255, 255), 3);

		exportCSV("d:/temp/with dictionary.csv", valuesToExport, i);


		cv::imshow("orig", color_mat);
		cv::imshow("restored", depthRestored);


		showProcessTime();

		int key = cv::waitKey(1);

		if (key == '1') scanrow = (scanrow + 1) % m.rows;
		if (key == '2') scanrow = (scanrow + 10) % m.rows;
		if (key == '3') scanrow = (scanrow + 100) % m.rows;

		if (key == 'n') i++;
		if (key == 'c') splLinear->zstd_compression_level = (splLinear->zstd_compression_level + 1) % 10;
		if (key == '8') iterCount = (iterCount + 1) % 20;
		if (key == '9') quantization = (quantization + 1) % 10;
		if (key == 't') splLinear->saveResidual = !splLinear->saveResidual;


		splLinear->memMgr.releaseAll();
		i++;
	}
	return 1;
}


///////////////////////////////////
// Test 0 : simple file processing
int test0(std::string inputDir, int minElementsCount, bool doPreProcess)
{
	std::cout << "Starting compression TEST" << "\n";
	cv::Mat depth = cv::imread("e://temp//l515//test_rgb0.png", -1);

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5), cv::Point(2, 2));
	cv::dilate(depth, depth, element);

	cv::erode(depth, depth, element);
	 
	if (!splLinear) {
		splLinear = new splineCompression(LINEAR_COMPRESSION); splLinear->lonelyPixelsRemoval = minElementsCount;
	}
	if (!splBiQubic) { splBiQubic = new splineCompression(SPLINE_COMPRESSION); splBiQubic->lonelyPixelsRemoval = minElementsCount;
	}
	if (!splLossLess) { splLossLess = new splineCompression(LOSSLESS_COMPRESSION); splLossLess->lonelyPixelsRemoval = minElementsCount;
	}
	if (!spl5) { spl5 = new splineCompression(SPLINE5_COMPRESSION); spl5->lonelyPixelsRemoval = minElementsCount;
	}

	// Apply threshold
	int iter = 0;
	int scanrow = 400;
	size_t orig_size = 1024 * 768 * 2;
	std::vector<std::string> files = listFilesInDir(inputDir, ".png");
	if (files.size() == 0)files = listFilesInDir(inputDir, ".ppm");
	if (files.size() == 0)files = listFilesInDir(inputDir, ".pgm");

	while (iter < files.size())
	{
		cv::Mat depth = cv::imread(files[iter], -1);

		if (doPreProcess)
		{
			//cv::resize(depth, depth, cv::Size(), 2.0, 2.0);
			//cv::blur(depth, depth, cv::Size(3, 3));
			
			cv::Mat temp;

			
			cv::GaussianBlur(depth, temp, cv::Size(0, 0), 3);
			cv::addWeighted(depth, 1.25, temp, -0.25, 0, depth);

			depth = depth * 0.25;
		}


		// With Residual
		compressionMetrics(depth, "e://temp//l515//residual//", 0, true, true,1);

		// non Residual
		compressionMetrics(depth, "e://temp//l515//normal//", 0, false, true,1);

		cv::Mat depthRestored = splLossLess->restoreAsImage(true);
		cv::Mat depthRestoredLinear = splLinear->restoreAsImage(true);
		cv::Mat depthRestoredCubic = splBiQubic->restoreAsImage(true);


		splLinear->display(scanrow, LINEAR_COMPRESSION, 1);

		splBiQubic->display(scanrow, SPLINE_COMPRESSION, 1);

		double pp = qm::psnr(depth, depthRestored, 1);
		double ppL = qm::psnr(depth, depthRestoredLinear, 1);
		double pp85 = qm::psnr(depth, depthRestoredCubic, 1);


		cv::Mat im_color, im_gray;
		depthRestored.convertTo(im_gray, CV_8UC1, 1.0 / 160.0);
		cv::applyColorMap(im_gray, im_color, cv::COLORMAP_HSV);

		cv::putText(im_color, std::to_string(pp), cv::Point(20, 20), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_color, std::to_string(ppL), cv::Point(20, 50), 1, 1, cv::Scalar(255, 255, 255));
		cv::putText(im_color, std::to_string(pp85), cv::Point(20, 80), 1, 1, cv::Scalar(255, 255, 255));


		cv::line(im_color, cv::Point(0, scanrow), cv::Point(depthRestored.cols - 1, scanrow), cv::Scalar(255, 255, 255), 3);
		cv::imshow("M1", im_color);
		cv::imshow("M2", depthRestoredLinear);
		cv::imshow("M3", depthRestoredCubic);
		cv::imshow("x2", depth);

		int key = cv::waitKey(-1);

		if (key == '1') scanrow = (scanrow + 1) % depth.rows;
		if (key == '2') scanrow = (scanrow + 10) % depth.rows;
		if (key == '3') scanrow = (scanrow + 100) % depth.rows;
		if (key == '4') scanrow--;
		else iter++;

		if (iter > 150) break;
	}
	return 1;
}

std::vector<std::string> listFilesInDir(std::string path, std::string extension)
{
	std::vector<std::string> ls;
	for (const auto & entry : std::experimental::filesystem::directory_iterator(path))
	{
		std::string ext = entry.path().extension().string();

		if (ext != extension) continue;

		ls.push_back(entry.path().string());
	//	std::cout << entry.path() << std::endl;
	}

	return ls;
}