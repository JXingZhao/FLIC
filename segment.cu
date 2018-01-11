#include<iostream>
#include<cuda_runtime.h>
#include<opencv2\opencv.hpp>
#include<device_launch_parameters.h>
#include"utils.h"
using namespace std;
using namespace cv;
static __device__ double compute_distance(const Component &center, const Pixel &point, const double &area, int sign)
{
	double colorDiff = (center.lMean - point.l) * (center.lMean - point.l) + (center.aMean - point.a) * (center.aMean - point.a) + (center.bMean - point.b) * (center.bMean - point.b);
	double spaceDiff = (center.xMean - point.x) * (center.xMean - point.x) + (center.yMean - point.y) * (center.yMean - point.y);
	double diff = 0.f;
	if (sign == 0)
		diff = colorDiff;
	else if (sign == 1)
	{

		diff = colorDiff + spaceDiff * 10;
	}
	return diff;
}

__device__ int _max(int x, int y)
{
	return x > y ? x : y;
}

__device__ int _min(int x, int y)
{
	return x < y ? x : y;
}

extern __global__ void clustering(const int* img,int* imgLabels, Component* _components, const Score* scores, const bool* is_test_, const int* _height_, const int* _width_, int* xbuff_, int* ybuff_)
{
	int i = threadIdx.x;
	
	Component &component = _components[scores[i].label];
	if (component.score < 1.0)
		return;
	int count = 1;
	const int numDirections = 4;
	const bool is_test = *is_test_;
	const int _width = *_width_;
	const int _height = *_height_;
	const int sz = _height * _width;
	const int dxs[numDirections] = { -1, 0, 1, 0 };
	const int dys[numDirections] = { 0, -1, 0, 1 };
	int* xbuff = xbuff_ + i * _width * _height ;
	int* ybuff = ybuff_ + i * _width * _height ;
	//
	xbuff[0] = component.min_x;
	ybuff[0] = component.min_y;



	int x = xbuff[0];
	int y = ybuff[0];
	while (true)
	{
		x = xbuff[0];
		while (true)
		{
			if ((x >= 0 && x < _width) && (y >= 0 && y < _height) && (component.label == imgLabels[y * _width + x]))
			{
				xbuff[count] = x;
				ybuff[count] = y;
				

				++count;
			}
			if (x == component.max_x)
				break;
			x++;
		}

		if (y == component.max_y)
			break;
		y++;

	}
	//}





	int label = component.label;
	Pixel point;
	for (int c = 0; c < count; ++c)
	{
		int labels[numDirections] = { 0 };  // Left, right, top, bottom
		double minDiff = 1.0e10;
		int finalLabel = 0;
		labels[0] = imgLabels[(ybuff[c] * _width + _max(0, xbuff[c] - 1))];
		labels[1] = imgLabels[(ybuff[c] * _width + _min(_width - 1, xbuff[c] + 1))];
		labels[2] = imgLabels[(_max(0, ybuff[c] - 1) * _width + xbuff[c])];
		labels[3] = imgLabels[(_min(_height - 1, ybuff[c] + 1) * _width + xbuff[c])];
		/*	labels[4] = imgLabels.at<int>(max(0,ybuff[c] -1), max(0, xbuff[c] - 1));
		labels[5] = imgLabels.at<int>(max(0, ybuff[c] - 1), min(_width - 1, xbuff[c] + 1));
		labels[6] = imgLabels.at<int>(min(_height - 1, ybuff[c] + 1), max(0, xbuff[c] - 1));
		labels[7] = imgLabels.at<int>(min(_height - 1, ybuff[c] + 1), min(_width - 1, xbuff[c] + 1)); */
		/*	if (labels[0] == label && labels[2] == label && labels[1] == label && labels[3] == label)
		continue;*/
		if (labels[0] == labels[1] && labels[2] == labels[3] && labels[0] == labels[2])
			continue;
		//for (int i = 0; i < 4; i++)
		//{
		//	if (labels[i] != label)
		//	{
		//		sn = 1;
		//	}
		//}
		//if (sn == 0)
		//	continue;
		point.l = img[ybuff[c] * _width * 3 + xbuff[c] * 3 + 0];
		point.a = img[ybuff[c] * _width * 3 + xbuff[c] * 3 + 1];
		point.b = img[ybuff[c] * _width * 3 + xbuff[c] * 3 + 2];
		point.x = xbuff[c];
		point.y = ybuff[c];
		//minDiff = compute_distance(_components[label], point, _components[label].num, 1);
		for (int j = 0; j < 4; ++j)
		{
			double diff = 0;
			if (is_test)
				diff = compute_distance(_components[labels[j]], point, _components[labels[j]].num, 0);
			else
				diff = compute_distance(_components[labels[j]], point, _components[labels[j]].num, 1);

			if (diff < minDiff)
			{
				minDiff = diff;
				finalLabel = labels[j];
			}
		}
		//cout << minDiff << endl;
		if (finalLabel == label)
			continue;
		imgLabels[ybuff[c] * _width + xbuff[c]] = finalLabel;

		_components[finalLabel].lMean = (_components[finalLabel].lMean * _components[finalLabel].num + point.l) / (_components[finalLabel].num + 1);
		_components[finalLabel].aMean = (_components[finalLabel].aMean * _components[finalLabel].num + point.a) / (_components[finalLabel].num + 1);
		_components[finalLabel].bMean = (_components[finalLabel].bMean * _components[finalLabel].num + point.b) / (_components[finalLabel].num + 1);
		_components[finalLabel].xMean = (_components[finalLabel].xMean * _components[finalLabel].num + point.x) / (_components[finalLabel].num + 1);
		_components[finalLabel].yMean = (_components[finalLabel].yMean * _components[finalLabel].num + point.y) / (_components[finalLabel].num + 1);
		_components[finalLabel].num += 1;

		if (_components[label].num > 1)
		{
			_components[label].lMean = (_components[label].lMean * _components[label].num - point.l) / (_components[label].num - 1);
			_components[label].aMean = (_components[label].aMean * _components[label].num - point.a) / (_components[label].num - 1);
			_components[label].bMean = (_components[label].bMean * _components[label].num - point.b) / (_components[label].num - 1);
			_components[label].xMean = (_components[label].xMean * _components[label].num - point.x) / (_components[label].num - 1);
			_components[label].yMean = (_components[label].yMean * _components[label].num - point.y) / (_components[label].num - 1);
		}
		_components[label].num -= 1;


		//// Update start point
		//update_start_point(_components[finalLabel], point);

	}



	count = 1;
	xbuff[0] = component.max_x;
	ybuff[0] = component.max_y;




	x = xbuff[0];
	y = ybuff[0];
	while (true)
	{
		x = xbuff[0];
		while (true)
		{
			if ((x >= 0 && x < _width) && (y >= 0 && y < _height) && (component.label == imgLabels[(y * _width + x)]))
			{
				xbuff[count] = x;
				ybuff[count] = y;
			
				
				++count;
			}
			if (x == component.min_x)
				break;
			x--;
		}
		if (y == component.min_y)
			break;
		y--;
		//}

	}



	label = component.label;
	for (int c = 0; c < count; ++c)
	{
		int labels[numDirections] = { 0 };  // Left, right, top, bottom
		double minDiff = 1.0e10;
		int finalLabel = 0;
		labels[0] = imgLabels[(ybuff[c] * _width + _max(0, xbuff[c] - 1))];
		labels[1] = imgLabels[(ybuff[c] * _width + _min(_width - 1, xbuff[c] + 1))];
		labels[2] = imgLabels[(_max(0, ybuff[c] - 1) * _width + xbuff[c])];
		labels[3] = imgLabels[(_min(_height - 1, ybuff[c] + 1) * _width + xbuff[c])];
		if (labels[0] == labels[1] && labels[2] == labels[3] && labels[0] == labels[2])
			continue;


		point.l = img[ybuff[c] * _width * 3 + xbuff[c] * 3 + 0];
		point.a = img[ybuff[c] * _width * 3 + xbuff[c] * 3 + 1];
		point.b = img[ybuff[c] * _width * 3 + xbuff[c] * 3 + 2];
		point.x = xbuff[c];
		point.y = ybuff[c];

		for (int j = 0; j < 4; ++j)
		{
			double diff = compute_distance(_components[labels[j]], point, _components[labels[j]].num, 1);

			if (diff < minDiff)
			{
				minDiff = diff;
				finalLabel = labels[j];
			}
		}
		//cout << minDiff << endl;
		if (finalLabel == label)
			continue;

		imgLabels[(ybuff[c] * _width + xbuff[c])] = finalLabel;

	
		if (is_test)
		{
			_components[finalLabel].xMean = (_components[finalLabel].xMean * _components[finalLabel].num + point.x) / (_components[finalLabel].num + 1);
			_components[finalLabel].yMean = (_components[finalLabel].yMean * _components[finalLabel].num + point.y) / (_components[finalLabel].num + 1);
			_components[finalLabel].num += 1;

			if (_components[label].num > 1)
			{
				_components[label].xMean = (_components[label].xMean * _components[label].num - point.x) / (_components[label].num - 1);
				_components[label].yMean = (_components[label].yMean * _components[label].num - point.y) / (_components[label].num - 1);
			}
			_components[label].num -= 1;


			// Update start point
			/*update_start_point(_components[finalLabel], point);*/
		}
		else
		{
			_components[finalLabel].lMean = (_components[finalLabel].lMean * _components[finalLabel].num + point.l) / (_components[finalLabel].num + 1);
			_components[finalLabel].aMean = (_components[finalLabel].aMean * _components[finalLabel].num + point.a) / (_components[finalLabel].num + 1);
			_components[finalLabel].bMean = (_components[finalLabel].bMean * _components[finalLabel].num + point.b) / (_components[finalLabel].num + 1);
			_components[finalLabel].xMean = (_components[finalLabel].xMean * _components[finalLabel].num + point.x) / (_components[finalLabel].num + 1);
			_components[finalLabel].yMean = (_components[finalLabel].yMean * _components[finalLabel].num + point.y) / (_components[finalLabel].num + 1);
			_components[finalLabel].num += 1;

			if (_components[label].num > 1)
			{
				_components[label].lMean = (_components[label].lMean * _components[label].num - point.l) / (_components[label].num - 1);
				_components[label].aMean = (_components[label].aMean * _components[label].num - point.a) / (_components[label].num - 1);
				_components[label].bMean = (_components[label].bMean * _components[label].num - point.b) / (_components[label].num - 1);
				_components[label].xMean = (_components[label].xMean * _components[label].num - point.x) / (_components[label].num - 1);
				_components[label].yMean = (_components[label].yMean * _components[label].num - point.y) / (_components[label].num - 1);
			}
			_components[label].num -= 1;


		/* Update start point
		update_start_point(_components[finalLabel], point);*/
		}
	}
	////delete[] xbuff;
	////delete[] ybuff;
}


extern cudaError_t clusterWithCuda(const Mat& img, Mat& imgLabels, Component* _components, const Score* scores, const bool is_test, const int _numsp)
{
	int _height = img.rows;
	int _width = img.cols;
	int* img_;
	int* imgLabels_;
	Component* _components_;
	Score* scores_;
	bool* is_test_;
	cudaError_t status;
	int* _height_;
	int* _width_;
	int* xbuff_;
	int* ybuff_;
	status = cudaSetDevice(0);

	if (status != cudaSuccess)
	{
		cout << "cudaSetDevice failed" << endl;
		
	}
	status = cudaMalloc((void**)&img_, _height * _width * 3 * sizeof(int));
	if (status != cudaSuccess)
	{
		cout << "cudaMalloc failed1" << endl;
		//goto Error;
	}
	status = cudaMalloc((void**)&imgLabels_, _height * _width * sizeof(int));
	if (status != cudaSuccess)
	{
		cout << "cudaMalloc failed2" << endl;
		//goto Error;
	}
	status = cudaMalloc((void**)&_components_, _numsp * sizeof(Component));
	if (status != cudaSuccess)
	{
		cout << "cudaMalloc failed3" << endl;
		//goto Error;
	}
	status = cudaMalloc((void**)&is_test_, sizeof(bool));
	if (status != cudaSuccess)
	{
		cout << "cudaMalloc failed4" << endl;
		//goto Error;
	}
	status = cudaMalloc((void**)&scores_, _numsp * sizeof(Score));
	if (status != cudaSuccess)
	{
		cout << "cudaMalloc failed5" << endl;
		//goto Error;
	}
	status = cudaMalloc((void**)&_height_, sizeof(int));
	if (status != cudaSuccess)
	{
		cout << "cudaMalloc failed6" << endl;
		//goto Error;
	}
	status = cudaMalloc((void**)&_width_, sizeof(int));
	if (status != cudaSuccess)
	{
		cout << "cudaMalloc failed7" << endl;
		//goto Error;
	}
	status = cudaMalloc((void**)&xbuff_, _numsp * _height * _width * sizeof(int));
	if (status != cudaSuccess)
	{
		cout << "cudaMalloc failed16" << endl;
		//goto Error;
	}
	status = cudaMalloc((void**)&ybuff_, _numsp * _height * _width * sizeof(int));
	if (status != cudaSuccess)
	{
		cout << "cudaMalloc failed17" << endl;
		//goto Error;
	}
	int* img_tmp = new int[_width * _height * 3];
	for (int h = 0; h < _height; ++h)
	{
		for (int w = 0; w < _width; ++w)
		{
			img_tmp[h * _width * 3 + w * 3 + 0] = static_cast<int>(img.at<Vec3b>(h, w)[0]);
			img_tmp[h * _width * 3 + w * 3 + 1] = static_cast<int>(img.at<Vec3b>(h, w)[1]);
			img_tmp[h * _width * 3 + w * 3 + 2] = static_cast<int>(img.at<Vec3b>(h, w)[2]);
		}
	}
	status = cudaMemcpy(img_, img_tmp, _width * _height * 3 * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cout << "cudaMemcpy failed8" << endl;
		//goto Error;
	}
	int* imgLabels_tmp = new int[_width * _height];
	for (int h = 0; h < _height; ++h)
	{
		for (int w = 0; w < _width; ++w)
		{
			imgLabels_tmp[h * _width + w] = imgLabels.at<int>(h, w);
		}
	}
	status = cudaMemcpy(imgLabels_, imgLabels_tmp, _width * _height * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cout << "cudaMemcpy failed9" << endl;
		//goto Error;
	}
	status = cudaMemcpy(_components_, _components, _numsp * sizeof(Component), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cout << "cudaMemcpy failed10" << endl;
		//goto Error;
	}
	status = cudaMemcpy(scores_, scores, _numsp * sizeof(Score), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cout << "cudaMemcpy failed11" << endl;
		//goto Error;
	}
	status = cudaMemcpy(is_test_, &is_test, sizeof(bool), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cout << "cudaMemcpy failed12" << endl;
		//goto Error;
	}
	status = cudaMemcpy(_width_, &_width, sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cout << "cudaMemcpy failed13" << endl;
		//goto Error;
	}
	status = cudaMemcpy(_height_, &_height, sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cout << "cudaMemcpy failed14" << endl;
		//goto Error;
	}
	clustering << <1, _numsp >> >(img_, imgLabels_, _components_, scores_, is_test_, _height_, _width_, xbuff_, ybuff_);

	status = cudaThreadSynchronize();
	if (status != cudaSuccess)
	{
		cout << "cudaThreadSynchronize failed" << endl;
		//goto Error;
	}






	status = cudaMemcpy(imgLabels_tmp, imgLabels_, _width * _height * sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		cout << "cudaMemcpy failed15" << endl;
		//goto Error;
	}

	for (int h = 0; h < _height; ++h)
	{
		for (int w = 0; w < _width; ++w)
		{
			imgLabels.at<int>(h, w) = imgLabels_tmp[h * _width + w];
		}
	}


	cudaFree(img_);
	cudaFree(imgLabels_);
	cudaFree(_components_);
	cudaFree(scores_);
	cudaFree(is_test_);
	cudaFree(_height_);
	cudaFree(_width_);
	cudaFree(xbuff_);
	cudaFree(ybuff_);
	delete[]img_tmp;
	delete[]imgLabels_tmp;
	return status;

}