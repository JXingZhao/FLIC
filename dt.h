#ifndef DT_H
#define DT_H

#include <algorithm>
#include "definitions.h"
#define INF 1E20

/* dt of 1d function using squared distance */
template <class T>
static T square(const T &a) { return a * a; };
static float *dt(float *f, int n) {
	float *d = new float[n];
	int *v = new int[n];
	float *z = new float[n + 1];
	int k = 0;
	v[0] = 0;
	z[0] = -INF;
	z[1] = +INF;
	for (int q = 1; q <= n - 1; q++) {
		float s = ((f[q] + square(q)) - (f[v[k]] + square(v[k]))) / (2 * q - 2 * v[k]);
		while (s <= z[k]) {
			k--;
			s = ((f[q] + square(q)) - (f[v[k]] + square(v[k]))) / (2 * q - 2 * v[k]);
		}
		k++;
		v[k] = q;
		z[k] = s;
		z[k + 1] = +INF;
	}

	k = 0;
	for (int q = 0; q <= n - 1; q++) {
		while (z[k + 1] < q)
			k++;
		d[q] = square(q - v[k]) + f[v[k]];
	}

	delete[] v;
	delete[] z;
	return d;
}

/* dt of 2d function using squared distance */
static void dt(Mat &im) {
	int width = im.cols;
	int height = im.rows;
	float *f = new float[max(width, height)];

	// transform along columns
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			f[y] = im.at<float>(y, x);
		}
		float *d = dt(f, height);
		for (int y = 0; y < height; y++) {
			im.at<float>(y, x) = d[y];
		}
		delete[] d;
	}

	// transform along rows
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			f[x] = im.at<float>(y, x);
		}
		float *d = dt(f, width);
		for (int x = 0; x < width; x++) {
			im.at<float>(y, x) = d[x];
		}
		delete[] d;
	}

	delete f;
}




#endif
