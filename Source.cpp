#include <iostream>
#include <opencv2/opencv.hpp>
#include "colorConversion.h"
#include "segmentation.h"
using namespace std;
using namespace cv;
using namespace colorconversion;
using namespace segmentation;
//using namespace imageprocessing ;

void  fillHole(const  Mat srcBw, Mat & dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type()); // Extend the image  
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
	cv::floodFill(Temp, Point(0, 0), Scalar(255)); // Populate the area  
	Mat cutImg; // Crop the stretched image  
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
	dstBw = srcBw | (~cutImg);
}

int main()
{



	cv::Mat ihls_image;
	cv::Mat input_image = imread("bienbao.jpg");
	colorconversion::convert_rgb_to_ihls(input_image, ihls_image);

	std::vector< cv::Mat > log_image;
	colorconversion::rgb_to_log_rb(input_image, log_image);
	int nhs_mode = 0; // nhs_mode == 0 -> red segmentation / nhs_mode == 1 -> blue segmentation
	cv::Mat nhs_image_seg_red;
	segmentation::seg_norm_hue(ihls_image, nhs_image_seg_red, nhs_mode);

	nhs_mode = 1; // nhs_mode == 0 -> red segmentation / nhs_mode == 1 -> blue segmentation

	cv::Mat nhs_image_seg_blue = nhs_image_seg_red.clone();
	segmentation::seg_norm_hue(ihls_image, nhs_image_seg_blue, nhs_mode);

	cv::Mat log_image_seg;
	segmentation::seg_log_chromatic(log_image, log_image_seg);

	cv::Mat merge_image_seg_with_red = nhs_image_seg_red.clone();
	cv::Mat merge_image_seg = nhs_image_seg_blue.clone();
	cv::bitwise_or(nhs_image_seg_red, log_image_seg, merge_image_seg_with_red);
	cv::bitwise_or(nhs_image_seg_blue, merge_image_seg_with_red, merge_image_seg);
	imshow("result", merge_image_seg);
	Mat temp2 = merge_image_seg.clone();
	fillHole(temp2, temp2);
	imshow("sadsa", temp2);
	vector<Vec3f> circles;
	HoughCircles(temp2, circles, CV_HOUGH_GRADIENT, 1, temp2.rows / 8, 200, 100, 0, 0);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(input_image, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(input_image, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}
	imshow("new image", input_image);
	waitKey(0);
	return  0;
}


