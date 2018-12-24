#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui_c.h"

using namespace std;
using namespace cv;

#define FIVE_AND_SIX

#define DISPLAY_IMG(win_name, img) namedWindow(win_name, CV_WINDOW_AUTOSIZE); \
imshow(win_name, img)

/* 5 */
int maxCorners = 23;
int maxTrackbar = 100;
double qualityLevel = 0.01;
double minDistance = 10;
const string window_src_name = "Source Image";
RNG rng(12345);

void goodFeaturesToTrack_Demo(int, void*)
{
	Mat src, src_gray;
	src = imread("Cat.jpg");
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Parameters for Shi-Tomasi algorithm
	maxCorners = MAX(maxCorners, 1);
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3, gradientSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;

	/// Copy the source image
	Mat copy = src.clone();

	/// Apply corner detection
	goodFeaturesToTrack(src_gray,
		corners,
		maxCorners,
		qualityLevel,
		minDistance,
		Mat(),
		blockSize,
		gradientSize,
		useHarrisDetector,
		k);


	/// Draw corners detected
	cout << "** Number of corners detected: " << corners.size() << endl;
	int radius = 4;
	for (size_t i = 0; i < corners.size(); i++)
	{
		circle(copy, corners[i], radius, Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED);
	}

	/// Show what you got
	namedWindow(window_src_name);
	imshow(window_src_name, copy);
}

/* 6 */
int maskSize0 = DIST_MASK_5;
int voronoiType = -1;
int edgeThresh = 100;
int distType0 = DIST_L1;

static void onTrackbar(int, void*)
{
	Mat src, src_gray;
	src = imread("Cat.jpg");
	cvtColor(src, src_gray, CV_BGR2GRAY);

	static const Scalar colors[] =
	{
		Scalar(0,0,0),
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};

	int maskSize = voronoiType >= 0 ? DIST_MASK_5 : maskSize0;
	int distType = voronoiType >= 0 ? DIST_L2 : distType0;

	Mat edge = src_gray >= edgeThresh, dist, labels, dist8u;

	if (voronoiType < 0)
		distanceTransform(edge, dist, distType, maskSize);
	else
		distanceTransform(edge, dist, labels, distType, maskSize, voronoiType);

	if (voronoiType < 0)
	{
		// begin "painting" the distance transform result
		dist *= 5000;
		pow(dist, 0.5, dist);

		Mat dist32s, dist8u1, dist8u2;

		dist.convertTo(dist32s, CV_32S, 1, 0.5);
		dist32s &= Scalar::all(255);

		dist32s.convertTo(dist8u1, CV_8U, 1, 0);
		dist32s *= -1;

		dist32s += Scalar::all(255);
		dist32s.convertTo(dist8u2, CV_8U);

		Mat planes[] = { dist8u1, dist8u2, dist8u2 };
		merge(planes, 3, dist8u);
	}
	else
	{
		dist8u.create(labels.size(), CV_8UC3);
		for (int i = 0; i < labels.rows; i++)
		{
			const int* ll = (const int*)labels.ptr(i);
			const float* dd = (const float*)dist.ptr(i);
			uchar* d = (uchar*)dist8u.ptr(i);
			for (int j = 0; j < labels.cols; j++)
			{
				int idx = ll[j] == 0 || dd[j] == 0 ? 0 : (ll[j] - 1) % 8 + 1;
				float scale = 1.f / (1 + dd[j] * dd[j] * 0.0004f);
				int b = cvRound(colors[idx][0] * scale);
				int g = cvRound(colors[idx][1] * scale);
				int r = cvRound(colors[idx][2] * scale);
				d[j * 3] = (uchar)b;
				d[j * 3 + 1] = (uchar)g;
				d[j * 3 + 2] = (uchar)r;
			}
		}
	}

	imshow("Distance Map", dist8u);
}

/* 7 */
void averageFilter(Mat& Input, Mat& Output)
{
	int row = Input.rows;
	int col = Input.cols;
	//uchar *input_data=Input.data;
	//uchar *output_data=Output.data;
	int size = 1;
	uchar count = 0;
	uchar sum = 0;

	//int nChannels=1;

	for (int j = 1; j < Input.rows - 1; ++j)
	{
		const uchar* previous = Input.ptr<uchar>(j - 1);
		const uchar* current = Input.ptr<uchar>(j);
		const uchar* next = Input.ptr<uchar>(j + 1);

		uchar* output = Output.ptr<uchar>(j);

		for (int i = 1; i < (Output.cols - 1); ++i)
		{
			double dResult = (current[i] + current[i - 1] + current[i + 1] +
				previous[i] + next[i] + previous[i - 1] + previous[i + 1] +
				next[i - 1] + next[i + 1]) / 9.0;
			*output++ = (unsigned char)dResult;
		}
	}
}


int main(int argc, char** argv)
{
	Mat src, /* 1 */
		src_gray, /* 2 */
		res_equa, /* 3 */
		dst, detected_edges, /* 4 */
		average_res, /* 6 */
		integral_im /* 7 */
		;
	
	const string window_res_equa_name = "Equalized Ñontrast";
	const string window_src_gray_name = "Source Gray Image";
	const string window_dst_name = "Dst Image";
	const string window_average_res_name = "average_res Image";
	const string window_integral_im_name = "integral_im Image";

	/* ###################################################################################### */
	/* 1. Read image from a file - DONE */
	src = imread("Cat.jpg");

	if (!src.data)
	{
		return -1;
	}

	/* ###################################################################################### */
	/* 2. Convert image to grayscale - DONE */
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/* ###################################################################################### */
	/* 3. Improve the contrast - DONE */
	equalizeHist(src_gray, res_equa);

	/* ###################################################################################### */
	/* 4. Find edge points - DONE */
	int lowThreshold = 50;
	const int max_lowThreshold = 100;
	const int ratio = 3;
	const int kernel_size = 3;

	blur(src_gray, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size/*, true*/);

	//detected_edges = Scalar::all(255) - detected_edges;
	dst = Scalar::all(0);
	src.copyTo(dst, detected_edges);

	/*distanceTransform(detected_edges, average_res, CV_DIST_L2, 3);

	Mat result(src_gray.rows, src_gray.cols, src_gray.type());
	integral(src_gray, integral_im);
	for (int i = 1; i < integral_im.rows; i++)
		for (int j = 1; j < integral_im.cols; j++)
		{
			int size = 1 * (int)average_res.at<float>(i - 1, j - 1);
			if (size % 2 == 0) size--;

			int i_top = max(i - size / 2, 1);
			int i_bot = min(i + size / 2, integral_im.rows - 1);
			int j_left = max(j - size / 2, 1);
			int j_right = min(j + size / 2, integral_im.cols - 1);

			int count = (i_bot - i_top + 1)*(j_right - j_left + 1);
			double sum = integral_im.at<int>(i_bot, j_right)
				+ integral_im.at<int>(i_top - 1, j_left - 1)
				- integral_im.at<int>(i_top - 1, j_right)
				- integral_im.at<int>(i_bot, j_left - 1);

			result.at<uchar>(i - 1, j - 1) = sum / count;
		}*/

#ifdef FIVE_AND_SIX
	/* ###################################################################################### */
	/* 5. Find corner points - DONE */
	namedWindow(window_src_name);
	createTrackbar("Max corners:", window_src_name, &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo);
	imshow(window_src_name, src);
	goodFeaturesToTrack_Demo(0, 0);

	/* ###################################################################################### */
	/* 6. Build a distance map - DONE */
	//distanceTransform(detected_edges, average_res, CV_DIST_L2, 3);
	namedWindow("Distance Map", 1);
	createTrackbar("Brightness Threshold", "Distance Map", &edgeThresh, 255, onTrackbar, 0);

	// Call to update the view
	onTrackbar(0, 0);
#endif // FIVE_AND_SIX

	/* ###################################################################################### */
	/* 7. Get rid of noise by averaging - DONE */
	//averageFilter(src, average_res);
	Mat noise_src, noise_src_gray, noise_detected_edges, noise_average_res;
	noise_src = imread("Noise.jpg");
	DISPLAY_IMG("Source shum image", noise_src);
	cvtColor(noise_src, noise_src_gray, CV_BGR2GRAY);
	blur(noise_src_gray, noise_detected_edges, Size(3, 3));
	Canny(noise_detected_edges, noise_detected_edges, lowThreshold, lowThreshold * ratio, kernel_size, true);

	noise_detected_edges = Scalar::all(255) - noise_detected_edges;

	distanceTransform(noise_detected_edges, noise_average_res, CV_DIST_L2, 3);

	Mat result(noise_src_gray.rows, noise_src_gray.cols, noise_src_gray.type());
	integral(noise_src_gray, integral_im); // Integral image

	for (int i = 1; i < integral_im.rows; i++)
		for (int j = 1; j < integral_im.cols; j++)
		{
			int size = 1 * (int)noise_average_res.at<float>(i - 1, j - 1);
			if (size % 2 == 0) size--;

			int i_top = max(i - size / 2, 1);
			int i_bot = min(i + size / 2, integral_im.rows - 1);
			int j_left = max(j - size / 2, 1);
			int j_right = min(j + size / 2, integral_im.cols - 1);

			int count = (i_bot - i_top + 1)*(j_right - j_left + 1);
			double sum = integral_im.at<int>(i_bot, j_right)
				+ integral_im.at<int>(i_top - 1, j_left - 1)
				- integral_im.at<int>(i_top - 1, j_right)
				- integral_im.at<int>(i_bot, j_left - 1);

			result.at<uchar>(i - 1, j - 1) = sum / count;
		}

	/* Image output */
	DISPLAY_IMG("Source", src);
	DISPLAY_IMG(window_src_gray_name, src_gray);
	DISPLAY_IMG(window_res_equa_name, res_equa);	
	DISPLAY_IMG(window_dst_name, dst);
	DISPLAY_IMG(window_average_res_name, result);

	waitKey(0);

	/* Removing Windows */
	cvDestroyAllWindows();

	return 0;
}