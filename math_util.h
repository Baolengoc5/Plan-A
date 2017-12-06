#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

#define EPSILON 1e-9
#define FLAT_TRIANGLE_AREA 0.0001



namespace mathutils {

	// Maximum between three values
	inline float get_maximum(const float& r, const float& g, const float& b)
	{
		return (r >= g) ? ((r >= b) ? r : b) : ((g >= b) ? g : b);
	}

	// Minimum between three values
	inline float get_minimum(const float& r, const float& g, const float& b)
	{
		return (r <= g) ? ((r <= b) ? r : b) : ((g <= b) ? g : b);
	}

}


namespace cv {

	//////////////////////////////// PointPolar_ ////////////////////////////////
	/*!
	template 2D polar point class.
	The class defines a point in 2D polar space. Data type of the point coordinates is specified
	as a template parameter. There are a few shorter aliases available for user convenience.
	*/
	template<typename _Tp> class PointPolar_ {
	public:

		//typedef _Tp value_type;

		// various constructors
		PointPolar_() { phi = 0; theta = 0; }
		PointPolar_(const PointPolar_<_Tp>& pt) { phi = pt.phi; theta = pt.theta; }
		PointPolar_(_Tp _phi, _Tp _theta) { phi = _phi; theta = _theta; }

		// function euclidean to polar coordinates
		void eucl_to_polar(const _Tp& _x, const _Tp& _y) {
			// Convert Euclidean coordinate to polar coordinate
			theta = std::atan2(_y, _x);
			if (theta < 0)
				theta += 2 * M_PI;
			phi = std::sqrt(std::pow(_y, 2.00) + std::pow(_x, 2.00));
		}
		void eucl_to_polar(const cv::Point_<_Tp>& _pt) {
			// Convert Euclidean coordinate to polar coordinate
			theta = std::atan2(_pt.y, _pt.x);
			if (theta < 0)
				theta += 2 * 'M_PI';
			phi = std::sqrt(std::pow(_pt.y, 2.00) + std::pow(_pt.x, 2.00));
		}

		// function polar to euclidean coordinates
		cv::Point_<_Tp> polar_to_eucl() {
			// Convert polar to euclidean coordinates
			return cv::Point(phi * std::cos(theta), phi * std::sin(theta));
		}
		void polar_to_eucl(_Tp& _x, _Tp& _y) {
			// Convert polar to euclidean coordinates
			_x = phi * std::cos(theta);
			_y = phi * std::sin(theta);
		}

		// Overload of some operator
		bool operator<(const PointPolar_<_Tp> &pt) const { return theta < pt.theta; }
		bool operator>(const PointPolar_<_Tp> &pt) const { return theta > pt.theta; }
		friend std::ostream& operator<<(std::ostream& os, const PointPolar_<_Tp>& pt)
		{
			os << "[" << pt.phi << ", " << pt.theta << "]";
			return os;
		}

		_Tp phi, theta; //< the point coordinates
	};

	typedef PointPolar_<int> PointPolar2i;
	typedef PointPolar2i PointPolar;
	typedef PointPolar_<float> PointPolar2f;
	typedef PointPolar_<double> PointPolar2d;

}
