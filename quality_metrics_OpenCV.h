#pragma once
#include <opencv2/opencv.hpp>

namespace qm
{
	/**
	 *	Compute the PSNR between 2 images
	 */
	double psnr(cv::Mat & img_src, cv::Mat & img_compressed, int block_size);
	double ssim(cv::Mat & img_src, cv::Mat & img_compressed, int block_size, bool show_progress = false);
	double absDif(cv::Mat & img_src, cv::Mat & img_compressed, int blockSizez);
}