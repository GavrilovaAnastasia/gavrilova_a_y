#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>

//concat all images in one
cv::Mat concatenate(std::vector<cv::Mat>& images, const int cols_num, const int rows_num) {
    const int rows = rows_num * images[0].rows;
    const int cols = cols_num * images[0].cols;
    cv::Mat res(rows, cols, CV_32FC1);
    for (int i = 0; i < images.size(); i++)
        images[i].copyTo(res(cv::Rect(images[0].cols * (i % cols_num), images[0].rows * (i / cols_num),
            images[i].cols, images[i].rows)));
    return res;
}

cv::Mat draw_image(const uchar bg_color, const uchar circle_color) {
    cv::Mat image(150, 150, CV_32FC1, cv::Scalar(bg_color));
    cv::circle(image, cv::Point(image.cols / 2, image.rows / 2), 70, cv::Scalar(circle_color), -1, 8, 0);
    return image;
}


int main() {
    std::vector<cv::Mat> images;
    std::vector<uchar> colors = { 0, 127, 255 };
    images.reserve(6);

    for (int i = 0; i < images.capacity() / 2; i++)
        images.push_back(draw_image(colors[i % 3], colors[(i + 1) % 3]));
    for (int i = 0; i < images.capacity() / 2; i++)
        images.push_back(draw_image(colors[(i + 1) % 3], colors[i % 3]));

    cv::Mat res = concatenate(images, 3, 2);

    cv::Mat I1, I2;
    float kernel_data1[] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
    cv::Mat kernel1(3, 3, CV_32FC1, kernel_data1);
    filter2D(res, I1, -1, kernel1);

    float kernel_data2[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
    cv::Mat kernel2(3, 3, CV_32FC1, kernel_data2);
    filter2D(res, I2, -1, kernel2);

    cv::Mat I_heometric(I1.size(), CV_32FC1);
    for (int i = 0; i < I_heometric.rows; i++) {
        for (int j = 0; j < I_heometric.cols; j++) {
            float I1_val = I1.at<float>(i, j);
            float I2_val = I2.at<float>(i, j);
            float I_val = sqrt(I1_val * I1_val + I2_val * I2_val);
            if (I_val < 0)
                I_val = (I_val + 255.0f) / 2;
            I_heometric.at<float>(i, j) = I_val;
        }
    }

    cv::imwrite("concat_image.png", res);
    cv::imwrite("I1.png", I1);
    cv::imwrite("I2.png", I2);
    cv::imwrite("I_heometric.png", I_heometric);
    cv::waitKey(0);
}

