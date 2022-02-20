#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

cv::Mat merge_images(cv::Mat& b, cv::Mat& g, cv::Mat& r) {
    std::vector<cv::Mat> channels;
    cv::Mat res_img;
    channels.push_back(b);
    channels.push_back(g);
    channels.push_back(r);
    merge(channels, res_img);
    return res_img;
}

cv::Mat concatenate(cv::Mat& img11, cv::Mat& img12, cv::Mat& img21, cv::Mat& img22) {
    const int rows = 2 * img11.rows;
    const int cols = 2 * img11.cols;
    cv::Mat res(rows, cols, CV_8UC3);
    img11.copyTo(res(cv::Rect(0, 0, img11.cols, img11.rows)));
    img21.copyTo(res(cv::Rect(0, img11.rows, img21.cols, img21.rows)));
    img12.copyTo(res(cv::Rect(img11.cols, 0, img12.cols, img12.rows)));
    img22.copyTo(res(cv::Rect(img11.cols, img11.rows, img22.cols, img22.rows)));
    return res;
}

cv::Mat draw_histogram(cv::Mat channels[]) {
    const int hist_width = 512, hist_height = 400, hist_size = 256;
    int bin_width = cvRound((double)hist_width / hist_size);
    cv::Mat histogram(hist_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat blue_hist, green_hist, red_hist;
    //color values ​​range is [0, 255]
    float r[] = { 0, 256 };
    const float* hist_range[] = { r };
    const int hist_rows = channels[0].rows;
    const int hist_cols = channels[0].cols;
    //calculating hist
    cv::calcHist(&channels[0], 1, 0, cv::Mat(), blue_hist, 1, &hist_size, hist_range, true, false);
    cv::calcHist(&channels[1], 1, 0, cv::Mat(), green_hist, 1, &hist_size, hist_range, true, false);
    cv::calcHist(&channels[2], 1, 0, cv::Mat(), red_hist, 1, &hist_size, hist_range, true, false);
    
    cv::normalize(blue_hist, blue_hist, 0, histogram.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(green_hist, green_hist, 0, histogram.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(red_hist, red_hist, 0, histogram.rows, cv::NORM_MINMAX, -1, cv::Mat());

    //drawing hist
    for (int i = 1; i < hist_size; i++) {
        cv::line(
            histogram,
            cv::Point(bin_width * (i - 1), hist_height - cvRound(blue_hist.at<float>(i - 1))),
            cv::Point(bin_width * (i), hist_height - cvRound(blue_hist.at<float>(i))),
            cv::Scalar(255, 0, 0), 1, 8, 0);
        cv::line(
            histogram,
            cv::Point(bin_width * (i - 1), hist_height - cvRound(green_hist.at<float>(i - 1))),
            cv::Point(bin_width * (i), hist_height - cvRound(green_hist.at<float>(i))),
            cv::Scalar(0, 255, 0), 1, 8, 0);
        cv::line(
            histogram,
            cv::Point(bin_width * (i - 1), hist_height - cvRound(red_hist.at<float>(i - 1))),
            cv::Point(bin_width * (i), hist_height - cvRound(red_hist.at<float>(i))),
            cv::Scalar(0, 0, 255), 1, 8, 0);
    }
    return histogram;
}

int main() {
    cv::Mat image = cv::imread("./data/cross_0256x0256.png", cv::IMREAD_COLOR);
    cv::Mat black(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    if (image.empty())
    {
        std::cout << "Image not found" << std::endl;
        std::cin.get();
        return -1;
    }
    
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(25);
    cv::imwrite("cross_0256x0256_025.jpg", image, compression_params);

    cv::Mat jpeg_image = cv::imread("./cross_0256x0256_025.jpg", cv::IMREAD_COLOR);
    if (jpeg_image.empty())
    {
        std::cout << "JPEG Image not found" << std::endl;
        std::cin.get();
        return -1;
    }

    cv::Mat png_channels[3], jpeg_channels[3];
    cv::split(image, png_channels);
    cv::split(jpeg_image, jpeg_channels);

    cv::Mat png_blue = merge_images(png_channels[0], black, black);
    cv::Mat png_green = merge_images(black, png_channels[1], black);
    cv::Mat png_red = merge_images(black, black, png_channels[2]);

    cv::Mat jpeg_blue = merge_images(jpeg_channels[0], black, black);
    cv::Mat jpeg_green = merge_images(black, jpeg_channels[1], black);
    cv::Mat jpeg_red = merge_images(black, black, jpeg_channels[2]);

    cv::Mat res_png = concatenate(image, png_red, png_green, png_blue);
    cv::Mat res_jpeg = concatenate(jpeg_image, jpeg_red, jpeg_green, jpeg_blue);
    
    cv::Mat png_hist = draw_histogram(png_channels);
    cv::Mat jpeg_hist = draw_histogram(jpeg_channels);

    cv::Mat hists(png_hist.rows + 40, 2 * png_hist.cols + 30, CV_8UC3, cv::Scalar(50, 50, 50));
    png_hist.copyTo(hists(cv::Rect(10, 30, png_hist.cols, png_hist.rows)));
    jpeg_hist.copyTo(hists(cv::Rect(png_hist.cols + 20, 30, jpeg_hist.cols, jpeg_hist.rows)));
    cv::putText(hists, "PNG image histogram", cv::Point(10, 20), 2, 
        cv::getFontScaleFromHeight(2, 12, 1), cv::Scalar(0, 220, 0), 1, 8, false);
    cv::putText(hists, "JPEG image histogram", cv::Point(png_hist.cols + 20, 20), 2,
        cv::getFontScaleFromHeight(2, 12, 1), cv::Scalar(0, 220, 0), 1, 8, false);
    
    cv::imwrite("cross_0256x0256_png_channels.png", res_png);
    cv::imwrite("cross_0256x0256_jpg_channels.png", res_jpeg);
    cv::imwrite("cross_0256x0256_hists.png", hists);
    cv::imshow("JPEG 25%", jpeg_image);
    cv::imshow("PNG channels", res_png);
    cv::imshow("JPEG channels", res_jpeg);
    cv::imshow("PNG and JPEG histogram", hists);
    cv::waitKey(0);
    return 0;
}