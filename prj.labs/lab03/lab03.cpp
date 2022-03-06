#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

double converting_func(int x) {
	return std::sqrt(x) * std::abs(20 * std::sin(0.05 * x));
}

int main() {
	cv::Mat image = cv::imread("./data/cross_0256x0256.png", cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cout << "Image not found" << std::endl;
		std::cin.get();
		return -1;
	}

	cv::Mat lookup_table(1, 256, CV_8UC1);
	for (int i = 0; i < lookup_table.cols; ++i) {
		double res = converting_func(i);
		if (res > 255)
			lookup_table.at<uchar>(0, i) = 255;
		else if (res < 0) 
			lookup_table.at<uchar>(0, i) = 0;
		else
			lookup_table.at<uchar>(0, i) = (uchar)res;
	}

	cv::Mat grey, modified_grey, modified_image, lut_check;

	//converting image to greyscale
	cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

	//LUT vizualization (gradient)
	cv::resize(lookup_table, lut_check, cv::Size(lookup_table.cols * 3, lookup_table.rows * 100));

	cv::LUT(grey, lookup_table, modified_grey);
	cv::LUT(image, lookup_table, modified_image);

	//LUT vizualization (plot)
	cv::Mat viz_func(256, 256, CV_8UC1, 255);
	for (int i = 0; i < viz_func.cols; ++i) {
		uchar y = lookup_table.at<uchar>(0, i);
		viz_func.at<uchar>(255 - y, i) = 0;
	}
	cv::line(viz_func, cv::Point(0, 0), cv::Point(0, 256), 0, 2);
	cv::line(viz_func, cv::Point(0, 256), cv::Point(256, 256), 0, 4);
	cv::line(viz_func, cv::Point(0, 0), cv::Point(5, 0), 0, 2);
	cv::line(viz_func, cv::Point(256, 256), cv::Point(256, 249), 0, 4);
	cv::putText(viz_func, "255", cv::Point(2, 9), 1,
		cv::getFontScaleFromHeight(1, 6, 1), cv::Scalar(0, 220, 0), 1, 8, false);
	cv::putText(viz_func, "255", cv::Point(236, 253), 1,
		cv::getFontScaleFromHeight(1, 6, 1), cv::Scalar(0, 220, 0), 1, 8, false);
	cv::putText(viz_func, "0", cv::Point(4, 252), 1,
		cv::getFontScaleFromHeight(1, 6, 1), cv::Scalar(0, 220, 0), 1, 8, false);
	cv::resize(viz_func, viz_func, cv::Size(512, 512));

	cv::imwrite("lab03_rgb.png", image);
	cv::imwrite("lab03_grey.png", grey);
	cv::imwrite("lab03_grey_res.png", modified_grey);
	cv::imwrite("lab03_rgb_res.png", modified_image);
	cv::imwrite("lab03_grad_func.png", lut_check);
	cv::imwrite("lab03_viz_func.png", viz_func);
	return 0;
}

