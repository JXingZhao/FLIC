/*
 *
 *
 */

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <opencv2/opencv.hpp> 
#define MAX_PIXEL_VALUE 256
#define MAX_VALUE 35536
struct Spsize;

struct Pixel
{
	int index;
	int x;
	int y;
	union {
		int l;
		int r;
	};
	union {
		int a;
		int g;
	};
	int b;
};

struct Component
{
    double score;
	int num;
	int num_spa;
    int label;
	double lMean;
	double lMean_t;
	double aMean;
	double aMean_t;
	double bMean;
	double bMean_t;
	double xMean;
	double xMean_t;
	double yMean;
	double yMean_t;
    int lHist[MAX_PIXEL_VALUE];
    int aHist[MAX_PIXEL_VALUE];
    int bHist[MAX_PIXEL_VALUE];
	Pixel start;
	int max_x;
	int min_x;
	int max_y;
	int min_y;
	cv::Mat M_dist;
};



struct Spsize
{
    double h;
    double w;
};

struct Score
{
    int label;
    double score;
};


class Utils
{
public:
    template <class T>
    static T square(const T &a) { return a * a; }
};
#endif
