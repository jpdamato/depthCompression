#include <opencv2/opencv.hpp>
#include <iostream>

#include "quality_metrics_OpenCV.h"

using namespace std;
using namespace cv;

// quality-metric
namespace qm
{
	#define C1 (float) (0.01 * 255 * 0.01  * 255)
	#define C2 (float) (0.03 * 255 * 0.03  * 255)


	// sigma on block_size
	double sigma(Mat & m, int i, int j, int block_size)
	{
		double sd = 0;

		Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
		Mat m_squared(block_size, block_size, CV_64F);

		multiply(m_tmp, m_tmp, m_squared);

		// E(x)
		double avg = mean(m_tmp)[0];
		// E(x²)
		double avg_2 = mean(m_squared)[0];


		sd = sqrt(avg_2 - avg * avg);

		return sd;
	}

	// Covariance
	double cov(Mat & m1, Mat & m2, int i, int j, int block_size)
	{
		Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
		Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
		Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));


		multiply(m1_tmp, m2_tmp, m3);

		double avg_ro 	= mean(m3)[0]; // E(XY)
		double avg_r 	= mean(m1_tmp)[0]; // E(X)
		double avg_o 	= mean(m2_tmp)[0]; // E(Y)


		double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)

		return sd_ro;
	}

	// Mean squared error
	double eqm(Mat & img1, Mat & img2)
	{
		int i, j;
		double eqm = 0;
		int height = img1.rows;
		int width = img1.cols;

		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++)
				eqm += (img1.at<double>(i, j) - img2.at<double>(i, j)) * (img1.at<double>(i, j) - img2.at<double>(i, j));

		eqm /= height * width;

		return eqm;
	}



	/**
	 *	Compute the PSNR between 2 images
	 */
	double psnr(Mat & img_src, Mat & img_compressed, int block_size)
	{
		int D = 15000;
		
		if (img_src.cols == 0) return 0;
		if (img_compressed.cols == 0) return 0;


		if (img_src.type() != CV_64F)
		{
			cv::Mat isrc, icomp;
			img_src.convertTo(isrc, CV_64F);
			img_compressed.convertTo(icomp, CV_64F);
			return (10 * log10((D*D) / eqm(isrc, icomp)));
		}
		else
		{
			return (10 * log10((D*D) / eqm(img_src, img_compressed)));
		}
	}


	double absDif(Mat & img_src, Mat & img_compressed, int blockSizez)
	{
		cv::Mat isrc, icomp, idst;

		if (img_src.type() != CV_64F)
		{

			img_src.convertTo(isrc, CV_64F);
			img_compressed.convertTo(icomp, CV_64F);

		}
		else
		{
			isrc = img_src;
			icomp = img_compressed;
		}


		cv::absdiff(isrc, icomp, idst);

		return cv::sum(idst)[0] / (idst.rows * idst.cols);
	}

	/**
	 * Compute the SSIM between 2 images
	 */
	double ssim(Mat & img_src, Mat & img_compressed, int block_size, bool show_progress )
	{
		cv::Mat isrc, icomp;

		if (img_src.type() != CV_64F)
		{
			
			img_src.convertTo(isrc, CV_64F);
			img_compressed.convertTo(icomp, CV_64F);
			
		}
		else
		{
			isrc = img_src;
			icomp = img_compressed;
		}

		double ssim = 0;

		int nbBlockPerHeight 	= img_src.rows / block_size;
		int nbBlockPerWidth 	= img_src.cols / block_size;

		for (int k = 0; k < nbBlockPerHeight; k++)
		{
			for (int l = 0; l < nbBlockPerWidth; l++)
			{
				int m = k * block_size;
				int n = l * block_size;

				double avg_o 	= mean(img_src(Range(k, k + block_size), Range(l, l + block_size)))[0];
				double avg_r 	= mean(img_compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];
				double sigma_o 	= sigma(img_src, m, n, block_size);
				double sigma_r 	= sigma(img_compressed, m, n, block_size);
				double sigma_ro	= cov(img_src, img_compressed, m, n, block_size);

				ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));
				
			}
			// Progress
			if (show_progress)
				cout << "\r>>SSIM [" << (int) ((( (double)k) / nbBlockPerHeight) * 100) << "%]";
		}
		ssim /= nbBlockPerHeight * nbBlockPerWidth;

		if (show_progress)
		{
			cout << "\r>>SSIM [100%]" << endl;
			cout << "SSIM : " << ssim << endl;
		}

		return ssim;
	}

}