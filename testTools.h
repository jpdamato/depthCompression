#pragma once


#include <experimental/filesystem>
#include <omp.h>

#include "quality_metrics_OpenCV.h"
#include "cameraCapturer.h"

#include <chrono>

std::vector<std::string> getAllFilesInDir(const std::string& dirPath, std::vector<std::string> extensions);
void compressionMetrics(cv::Mat& depth, std::string outDir, int frame, bool residual, bool verbose, int minElementsCount);
cv::Mat calculateResidualAbs(cv::Mat& m, cv::Mat& depthRestored, int quantization, size_t outSizeC);
int test0(std::string inputDir,int minElementsCount, bool doPreProcess);
void createScripts(std::string outDir, int frame);
int test1(std::string inputDir, std::string iformat);
std::vector<std::string> listFilesInDir(std::string path, std::string extension);
void compareToJ2K(std::string outDir);
