#include <opencv2/opencv.hpp>

void Gradient(cv::Mat& img) {
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			img.col(j).setTo(j / 3);
}

void GammaCorrectionPow(cv::Mat& img, const double gammaValue) {
	img.convertTo(img, CV_32FC1, 1.0 / 255.0);	 
	cv::pow(img, gammaValue, img);
	img.convertTo(img, CV_8UC1, 255);   
}

void GammaCorrectionAt(cv::Mat& img, const double gammaValue) {
	img.convertTo(img, CV_32FC1, 1.0 / 255.0);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			float& color = img.at<float>(i, j);
			color = pow(color, gammaValue);
		}
	img.convertTo(img, CV_8UC1, 255);
}

int main() {
	cv::Mat I_1(60, 768, CV_8UC1, cv::Scalar(0));
	Gradient(I_1);
	cv::Mat G_1 = I_1, G_2 = I_1;
	//1st Gamma correction
	auto begin_pow = std::chrono::steady_clock::now();
	GammaCorrectionPow(G_1, 2.3);
	auto end_pow = std::chrono::steady_clock::now();
	auto res_pow = std::chrono::duration_cast<std::chrono::milliseconds>(end_pow - begin_pow);
	//2nd Gamma correction
	auto begin_at = std::chrono::steady_clock::now();
	GammaCorrectionAt(G_2, 2.3);
	auto end_at = std::chrono::steady_clock::now();
	auto res_at = std::chrono::duration_cast<std::chrono::milliseconds>(end_at - begin_at);
	//Image merging
	I_1.push_back(G_1);
	I_1.push_back(G_2);
	cv::imshow("Result", I_1);
	cv::imwrite("lab01.png", I_1);
	cv::waitKey(0);
	std::cout << "1 method's time: " << res_pow.count() << "ms" << std::endl;
	std::cout << "2 method's time: " << res_at.count() << "ms" << std::endl;
	return 0;
}