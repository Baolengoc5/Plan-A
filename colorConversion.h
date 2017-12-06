#pragma once
#include "math_util.h"

// stl library
#include <vector>
#include <cmath>

// OpenCV library
#include <opencv2/opencv.hpp>
#define M_PI 3.1415
namespace colorconversion {

	// Conversion from RGB to logarithm RB
	void rgb_to_log_rb(const cv::Mat& rgb_image, std::vector< cv::Mat >& log_chromatic_image);

	// Conversion from RGB to IHLS
	void convert_rgb_to_ihls(const cv::Mat& rgb_image, cv::Mat& ihls_image);

	// Theta computation
	inline double retrieve_theta(const float& r, const float& g, const float& b) { return acos((r - (g * 0.5) - (b * 0.5)) / sqrtf((r * r) + (g * g) + (b * b) - (r * g) - (r * b) - (g * b))); }
	// Hue computation -- H = θ if B <= G -- H = 2 * pi − θ if B > G
	inline double retrieve_normalised_hue(const float& r, const float& g, const float& b) { return (b <= g) ? (retrieve_theta(r, g, b) * 255.f / (2.f * M_PI)) : (((2.f * M_PI) - retrieve_theta(r, g, b)) * 255.f / (2.f * M_PI)); }
	// Luminance computation -- L = 0.210R + 0.715G + 0.072B
	inline double retrieve_luminance(const float& r, const float& g, const float& b) { return (0.210f * r) + (0.715f * g) + (0.072f * b); }
	// Saturation computation -- S = max(R, G, B) − min(R, G, B)
	inline double retrieve_saturation(const float& r, const float& g, const float& b) { return (mathutils::get_maximum(r, g, b) - mathutils::get_minimum(r, g, b)); }

}