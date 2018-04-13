
#include "hsp.h"
#include "utils.h"
#include "definitions.h"
#include "BSDSBench\Bench.h"
#include "dt.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>
#include <set>
#include <assert.h>
#include<device_launch_parameters.h>
#include<cuda_runtime.h>

using std::string;
using std::vector;
using std::set;
using std::cout;
using std::endl;
using namespace cv;
#define INF 1E20


bool compare_components(Score &a, Score &b)
{
    return a.score > b.score;
}


HSuperpixel::HSuperpixel() : _components(NULL), _width(0), _height(0), _channel(0), _xstrips(0), _ystrips(0), _numsp(0), c(0.0f)
{

}

HSuperpixel::~HSuperpixel()
{
	release_components();
}

void HSuperpixel::load_image(const string &imgName, Mat &out)
{
    Mat img = imread(imgName.c_str());
    cvtColor(img, img, COLOR_RGB2Lab);
//    GaussianBlur(img, img, Size(3, 3), 0.5, 0);
	for (int h = 0; h < img.rows; ++h) {
		for (int w = 0; w < img.cols; ++w) {
			img.at<Vec3b>(h, w)[0] /= 1.2;
//			img.at<Vec3b>(h, w)[2] /= 1.5;
		}
	}
    out = img;
}



void HSuperpixel::init(Mat &img, Mat &imgLabels, const Spsize &spSize)
{
    _width = img.cols;
    _height = img.rows;
	assert(_width > 0 && _height > 0);
    _channel = img.channels();
	_xstrips =  (0.5 + (double)_width) / spSize.w;
	_ystrips =  (0.5 + (double)_height) / spSize.h;
	int xerr = _width - spSize.w*_xstrips;
	if (xerr < 0)
	{
		_xstrips--;
		xerr = _width - spSize.w*_xstrips;
	}
	int yerr = _height - spSize.h*_ystrips;
	if (yerr < 0)
	{
		_ystrips--;
		yerr = _height - spSize.h*_ystrips;
	}
	double xerrperstrip = double(xerr) / double(_xstrips);
	double yerrperstrip = double(yerr) / double(_ystrips);
	

	int xrest = (_width - _xstrips * spSize.w) / 2;
	int yrest = (_height - _ystrips * spSize.h) / 2;
	_numsp = _xstrips * _ystrips;
	origin_point.clear();
	// Initialize indexes of each sp
	int label = 0;
	for (int y = 0; y < _ystrips; ++y)
    {
		for (int x = 0; x < _xstrips; ++x)
        {
			int wstart = x * spSize.w + x * xerrperstrip;
			int wend = (x + 1) * spSize.w + (x + 1) * xerrperstrip;
			int hstart = y * spSize.h + y * yerrperstrip;
			int hend = (y + 1) * spSize.h + (y + 1) * yerrperstrip;
			if (x == 0)
				wstart = 0;
			if (y == 0)
				hstart = 0;
			if (x == _xstrips - 1)
				wend = _width;
			if (y == _ystrips - 1)
				hend = _height;

            label = y * _xstrips + x;

            for (int h = hstart; h < hend; ++h)
            {
                int *pImgRowLabels = imgLabels.ptr<int>(h);
                for (int w = wstart; w < wend; ++w)
                    pImgRowLabels[w] = label;
            }
        }
    }

}



void HSuperpixel::draw_borders_manifold(unsigned int*& ubuff, int*& labels)
{
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	/*const int dx4[4] = { -1, 1, 0, 0 };
	const int dy4[4] = { 0, 0, -1, 1 };*/
	int width = _width;
	int height = _height;
	int sz = width*height;
	vector<bool> istaken(sz, false);
	vector<int> contourx(sz); vector<int> contoury(sz);
	int mainindex(0); int cind(0);
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			int np(0);
			for (int i = 0; i < 8; i++)
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height))
				{
					int index = y*width + x;
					//if( false == istaken[index] )//comment this to obtain internal contours
					{
						if (labels[mainindex] != labels[index]) np++;
					}
				}
			}
			//if( np > 1 )//change to 2 or 3 for thinner lines
			if (np > 2)
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				cind++;
			}
			mainindex++;
		}
	}
	
	/*for (int i = 0; i<sz; i++)
	{
		ubuff[i] = 0xffffff;
	}*/

	int numboundpix = cind;//int(contourx.size());
	for (int j = 0; j < numboundpix; j++)
	{
		int ii = contoury[j] * width + contourx[j];
		ubuff[ii] = 0xff0000;
	}
}
void HSuperpixel::enforce_label_connectivity(const int* labels, const int	width, const int height, int*& nlabels, int& numlabels)
{
	//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[8] = { -1, 0, 1, 0 , -1 , -1 ,1 ,1};
	const int dy4[8] = { 0, -1, 0, 1 , -1 ,1 ,-1 ,1};
	/*const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };*/
	const int sz = width*height;
	//nlabels.resize(sz, -1);
	for (int i = 0; i < sz; i++)
		nlabels[i] = -1;
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			if (0 > nlabels[oindex])
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				for (int n = 0; n < 8; n++)
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if ((x >= 0 && x < width) && (y >= 0 && y < height))
					{
						int nindex = y*width + x;
						if (nlabels[nindex] >= 0)
							adjlabel = nlabels[nindex];
					}
				}

				int count(1);
				for (int c = 0; c < count; c++)
				{
					for (int n = 0; n < 8; n++)
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if ((x >= 0 && x < width) && (y >= 0 && y < height))
						{
							int nindex = y*width + x;

							if (0 > nlabels[nindex] && labels[oindex] == labels[nindex])
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if (count <= (double)(_width * _height) / spcount / 5 )
				{
					for (int c = 0; c < count; c++)
					{
						int ind = yvec[c] * width + xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if (xvec) delete[] xvec;
	if (yvec) delete[] yvec;
}

void HSuperpixel::transform_from_mat(Mat &mat, int *buff)
{
	int width = mat.cols;
	int height = mat.rows;
	for (int h = 0; h < height; ++h)
	{
		int *pMat = mat.ptr<int>(h);
		for (int w = 0; w < width; ++w)
		{
			buff[h * width + w] = pMat[w];
		}
	}
}
void HSuperpixel :: transform_from_mat_img(Mat &mat, unsigned int *buff)
{
	int width = mat.cols;
	int height = mat.rows;
	for (int h = 0; h < height; ++h)
	{
		Vec3b *pMat = mat.ptr<Vec3b>(h);
		for (int w = 0; w < width; ++w)
		{
			unsigned int g = pMat[w][0];
			unsigned int b = pMat[w][1];
			unsigned int r = pMat[w][2];
			buff[h * width + w] = (r << 16) + (g << 8) + b;
		}
	}
}
void HSuperpixel::transform_to_mat(int *buff, Mat &mat)
{
	int width = mat.cols;
	int height = mat.rows;
	for (int h = 0; h < height; ++h)
	{
		int *pMat = mat.ptr<int>(h);
		for (int w = 0; w < width; ++w)
		{
			pMat[w] = buff[h * width + w];
		}
	}
}
void HSuperpixel::transform_to_mat_img(unsigned int *buff, Mat &mat)
{
	int width = mat.cols;
	int height = mat.rows;
	for (int h = 0; h < height; ++h)
	{
		Vec3b *pMat = mat.ptr<Vec3b>(h);
		for (int w = 0; w < width; ++w)
		{
			unsigned int value = buff[h * width + w];
			pMat[w][2] = static_cast<unsigned char>(value >> 16);
			pMat[w][0] = static_cast<unsigned char>((value >> 8) & 0xFF);
			pMat[w][1] = static_cast<unsigned char>(value & 0xFF);
		}
	}
}
void HSuperpixel::release_components()
{
	delete[] _components;
}

void HSuperpixel::segment(const string& imgName, const string& outputPath, const int sparea)
{
	c = 20;
	Mat img; 
	is_test = false;
	Mat img_rgb;
	string outName;
	Spsize spsize;
	int numLabels;
	img_rgb=imread(imgName);
	load_image(imgName, img);
	
	unsigned int* ubuff = new unsigned int[img.rows * img.cols];
	transform_from_mat_img(img_rgb, ubuff);
	transform_to_mat_img(ubuff, img_rgb);
	Mat imgLabels = Mat::zeros(img.rows, img.cols, CV_32S);
	//int spcount = img.rows * img.cols / sparea;
	spcount = sparea;
	const int superpixelsize = 0.5 + double(img.cols * img.rows) / double(spcount);
	const int step = sqrt(double(superpixelsize)) + 0.5;
	spsize.h = step;
	spsize.w = step;
	c = 1.0 / ((step / c)*(step / c));

	init(img, imgLabels, spsize);
	update_components(img, imgLabels);
	for (int i = 0; i < 2; i++)
	{
		update_score();
		forward_clustering(img, imgLabels);
		update_components(img, imgLabels);
	}




	cvtColor(img, img, COLOR_Lab2RGB);
	int *labels = new int[img.rows * img.cols];
	transform_from_mat(imgLabels, labels);
	int* finalLabels = new int[img.rows * img.cols];
	enforce_label_connectivity(labels, img.cols, img.rows, finalLabels, numLabels);
	_numsp = numLabels;
	transform_to_mat(finalLabels, imgLabels);

	Mat color_img = Mat::zeros(img.rows, img.cols, CV_8UC3);
	CmIllu::label2Rgb<int>(imgLabels, color_img);
	
	draw_borders_manifold(ubuff, finalLabels);



	


	
	transform_to_mat_img(ubuff, img_rgb);
	imwrite(outputPath, img_rgb);
	delete[] ubuff;
	delete[] labels;
	delete[] finalLabels;
}
void HSuperpixel::get_output_name(const string &src, string &out)
{
    string::size_type p = src.rfind(".");
    string tmp = src;
    out = tmp.insert(p, "_out");
}



void HSuperpixel::forward_clustering(const Mat &img, Mat &imgLabels)
{
	const int numDirections = 4;
	const int sz = _height * _width;
	const int dxs[numDirections] = { -1, 0, 1, 0 };
	const int dys[numDirections] = { 0, -1, 0, 1 };
	//const int dxs[numDirections] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	//const int dys[numDirections] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	Score *scores = new Score[_numsp];
	int sn = 0;
	// Score each component and then sort them

	for (int i = 0; i < _numsp; ++i)
	{
		scores[i].label = _components[i].label;
		scores[i].score = _components[i].score;
		
	}
	std::sort(scores, scores + _numsp, compare_components);
	
	
	int* xbuff = new int[sz];
	int* ybuff = new int[sz];
	for (int i = 0; i < _numsp; ++i)
	{
		
		Component &component = _components[scores[i].label];
		
		int count = 1;
		xbuff[0] = component.min_x;
		ybuff[0] = component.min_y;


		if (ybuff[0] < 0 || ybuff[0] >= _height)
		{
			cout << ybuff[0] << endl;
		}
		if (xbuff[0] < 0 || xbuff[0] >= _width)
		{
			cout << xbuff[0] << endl;
		}

	


		int x = xbuff[0];
		int y = ybuff[0];
		while (true)
		{
			x = xbuff[0];
			while (true)
			{
				if ((x >= 0 && x < _width) && (y >= 0 && y < _height) && (component.label == imgLabels.at<int>(y, x)) )
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
			labels[0] = imgLabels.at<int>(ybuff[c], max(0, xbuff[c] - 1));
			labels[1] = imgLabels.at<int>(ybuff[c], min(_width - 1, xbuff[c] + 1));
			labels[2] = imgLabels.at<int>(max(0, ybuff[c] - 1), xbuff[c]);
			labels[3] = imgLabels.at<int>(min(_height - 1, ybuff[c] + 1), xbuff[c]);
		//	labels[4] = imgLabels.at<int>(max(0,ybuff[c] -1), max(0, xbuff[c] - 1));
		//	labels[5] = imgLabels.at<int>(max(0, ybuff[c] - 1), min(_width - 1, xbuff[c] + 1));
		//	labels[6] = imgLabels.at<int>(min(_height - 1, ybuff[c] + 1), max(0, xbuff[c] - 1));
		//	labels[7] = imgLabels.at<int>(min(_height - 1, ybuff[c] + 1), min(_width - 1, xbuff[c] + 1)); 
			
			if (labels[0] == labels[1] && labels[2] == labels[3] && labels[0] == labels[2] && label == labels[0])
				continue;
			point.l = img.at<Vec3b>(ybuff[c], xbuff[c])[0];
			point.a = img.at<Vec3b>(ybuff[c], xbuff[c])[1];
			point.b = img.at<Vec3b>(ybuff[c], xbuff[c])[2];
			point.x = xbuff[c];
			point.y = ybuff[c];
			for (int j = 0; j < 4; ++j)
			{
				double diff = 0;
				if (true)
					diff = compute_distance(_components[labels[j]], point, _components[labels[j]].num, 0);
				else
					diff = compute_distance(_components[labels[j]], point, _components[labels[j]].num, 1);
			
				if (diff < minDiff)
				{
					minDiff = diff;
					finalLabel = labels[j];
				}
			}

			if (finalLabel == label)
				continue;
			imgLabels.at<int>(ybuff[c], xbuff[c]) = finalLabel;
			
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


			// Update start point
			update_start_point(_components[finalLabel], point);

		}



		count = 1;
		xbuff[0] = component.max_x;
		ybuff[0] = component.max_y;



		if (ybuff[0] < 0 || ybuff[0] >= _height)
		{
			cout << ybuff[0] << endl;
		}
		if (xbuff[0] < 0 || xbuff[0] >= _width)
		{
			cout << xbuff[0] << endl;
		}
	
		 x = xbuff[0];
		 y = ybuff[0];
		while (true)
		{
			x = xbuff[0];
			while (true)
			{
				if ((x >= 0 && x < _width) && (y >= 0 && y < _height) && (component.label == imgLabels.at<int>(y, x)))
				{
					xbuff[count] = x;
					ybuff[count] = y;
					assert(y >= 0 && y < _height);
					assert(x >= 0 && x < _width);
					
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
			labels[0] = imgLabels.at<int>(ybuff[c], max(0, xbuff[c] - 1));
			labels[1] = imgLabels.at<int>(ybuff[c], min(_width - 1, xbuff[c] + 1));
			labels[2] = imgLabels.at<int>(max(0, ybuff[c] - 1), xbuff[c]);
			labels[3] = imgLabels.at<int>(min(_height - 1, ybuff[c] + 1), xbuff[c]);
		//	labels[4] = imgLabels.at<int>(max(0, ybuff[c] - 1), max(0, xbuff[c] - 1));
		//	labels[5] = imgLabels.at<int>(max(0, ybuff[c] - 1), min(_width - 1, xbuff[c] + 1));
		//	labels[6] = imgLabels.at<int>(min(_height - 1, ybuff[c] + 1), max(0, xbuff[c] - 1));
		//	labels[7] = imgLabels.at<int>(min(_height - 1, ybuff[c] + 1), min(_width - 1, xbuff[c] + 1));
	
			if (labels[0] == labels[1] && labels[2] == labels[3] && labels[0] == labels[2] && label == labels[0])
				continue;
			point.l = img.at<Vec3b>(ybuff[c], xbuff[c])[0];
			point.a = img.at<Vec3b>(ybuff[c], xbuff[c])[1];
			point.b = img.at<Vec3b>(ybuff[c], xbuff[c])[2];
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
			move_number++;
			imgLabels.at<int>(ybuff[c], xbuff[c]) = finalLabel;

		
			if (true)
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
				update_start_point(_components[finalLabel], point);
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

				update_start_point(_components[finalLabel], point);
			}
		}

	}
	delete[] scores;
	delete[] xbuff;
	delete[] ybuff;
}

void HSuperpixel::update_sp_center(Component &component)
{
	
    int lSum = 0;
    int aSum = 0;
    int bSum = 0;
	if (component.num > 0)
	{
		for (int i = 0; i < MAX_PIXEL_VALUE; ++i)
		{
			lSum += component.lHist[i] * i;
			aSum += component.aHist[i] * i;
			bSum += component.bHist[i] * i;
		}
		component.lMean = lSum / component.num;
		component.aMean = aSum / component.num;
		component.bMean = bSum / component.num;
		component.xMean /= component.num;
		component.yMean /= component.num;

	}
}

void HSuperpixel::update_components(Mat &img, Mat &imgLabels)
{
	if (_components)
		delete[] _components;
	_components = new Component[_numsp];
	for (int i = 0; i < _numsp; ++i)
	{
		_components[i].score = 0.0f;
		_components[i].lMean = 0;
		_components[i].aMean = 0;
		_components[i].bMean = 0;
		_components[i].xMean = 0;
		_components[i].yMean = 0;
		_components[i].num = 0;
		_components[i].M_dist = Mat::zeros(3, 3, CV_64FC1);
		memset(_components[i].lHist, 0, 256 * sizeof(int));
		memset(_components[i].aHist, 0, 256 * sizeof(int));
		memset(_components[i].bHist, 0, 256 * sizeof(int));
	}

	for (int w = _width - 1; w >= 0; --w)
	{
		for (int h = _height - 1; h >= 0; --h)
		{
			int label = imgLabels.at<int>(h, w);
			_components[label].label = label;
			_components[label].start.x = w;
			_components[label].start.y = h;
			_components[label].xMean += w;
			_components[label].yMean += h;
			_components[label].num += 1;
		    _components[label].lHist[img.at<Vec3b>(h, w)[0]] += 1;
			_components[label].aHist[img.at<Vec3b>(h, w)[1]] += 1;
			_components[label].bHist[img.at<Vec3b>(h, w)[2]] += 1;
			_components[label].min_x = w;
			}
		}
	
	


	for (int w = 0; w < _width; ++w)
	{
		for (int h = 0; h < _height; ++h)
		{
			int label = imgLabels.at<int>(h, w);
			_components[label].label = label;
			_components[label].max_x = w;
		}
	}
	for (int h = _height - 1; h >= 0; --h)
	{
		for (int w = 0; w < _width; ++w)
		{
			int label = imgLabels.at<int>(h, w);
			_components[label].label = label;
			_components[label].min_y = h;
		}
	}
	for (int h = 0; h < _height; ++h)
	{
		for (int w = 0; w < _width; ++w)
		{
			int label = imgLabels.at<int>(h, w);
			_components[label].label = label;
			_components[label].max_y = h;
		}
	}




	vector<Point2i>v_tmp;
	v_tmp.resize(_numsp);
	Point2i p_tmp;
	for (int i = 0; i < _numsp; ++i)
	{
		update_sp_center(_components[i]);
	}
}



void HSuperpixel::score_each_sp(Component &component)
{
    double lScore = 0.0;
    double aScore = 0.0;
    double bScore = 0.0;
	if (component.num > 0)
	{
		for (int i = 0; i < MAX_PIXEL_VALUE; ++i)
		{
			if (component.lHist[i] != 0)
				lScore += (i - component.lMean)*(i - component.lMean)*component.lHist[i];
			if (component.aHist[i] != 0)
				aScore += (i - component.aMean)*(i - component.aMean)*component.aHist[i];
			if (component.bHist[i] != 0)
				bScore += (i - component.bMean)*(i - component.bMean)*component.bHist[i];
		
		}
		lScore = lScore / component.num;
		aScore = aScore / component.num;
		bScore = bScore / component.num;
		component.score = (lScore + aScore + bScore) / 3;
	}
}

void HSuperpixel::segment_for_evaluation(const std::string& imgName, const int sparea, int &numLabels, int itr_1 )
{
	itr = itr_1;
	Mat img;
	is_test = false;
	string outName;
	Spsize spsize;
	spcount = sparea;
	c = 4; 
	load_image(imgName, img);
	Mat imgLabels = Mat::zeros(img.rows, img.cols, CV_32S);
	
	const int superpixelsize = 0.5 + double(img.cols * img.rows) / double(spcount);
	const int step = sqrt(double(superpixelsize)) + 0.5;
	spsize.h = step;
	spsize.w = step;
	c = 1.0 / ((step / c)*(step / c));
	init(img, imgLabels, spsize);
	update_components(img, imgLabels);
	for (int i = 0; i < itr; i++)
	{	
		update_score();
		forward_clustering(img, imgLabels);
		update_components(img, imgLabels);
	}
	cvtColor(img, img, COLOR_Lab2RGB);
	int *labels = new int[img.rows * img.cols];
	transform_from_mat(imgLabels, labels);
	int *finalLabels = new int[img.rows * img.cols];
	enforce_label_connectivity(labels, img.cols, img.rows, finalLabels, numLabels);
	transform_to_mat(finalLabels, imgLabels);
	for (int h = 0; h < imgLabels.rows; ++h)
		for (int w = 0; w < imgLabels.cols; ++w)
			imgLabels.at<int>(h, w) += 1;
	_labels.push_back(imgLabels);
}
 
// Write matrix to binary file
bool HSuperpixel::matWrite(CStr &filename, vecM &_M)
{
	const int matCnt = _M.size();
	vecM M(matCnt);
	for (int i = 0; i < matCnt; i++) {
		M[i].create(_M[i].size(), _M[i].type());
		memcpy(M[i].data, _M[i].data, M[i].step*M[i].rows);
	}
	FILE* file = fopen(_S(filename), "wb");
	if (file == NULL || M[0].empty())
		return false;
	fwrite("CmMat", sizeof(char), 5, file);
	fwrite(&matCnt, sizeof(int), 1, file);
	for (int i = 0; i < matCnt; i++) {
		int headData[3] = { M[i].cols, M[i].rows, M[i].type() };
		fwrite(headData, sizeof(int), 3, file);
		fwrite(M[i].data, sizeof(char), M[i].step * M[i].rows, file);
	}
	std::fclose(file);
	return true;
}

// Read matrix from binary file
bool HSuperpixel::matRead(CStr &filename, vecM &_M)
{
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	fread(buf, sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		std::cout << "Invalidate CvMat data file " << _S(filename) << endl;
		return false;
	}
	int matCnt;
	fread(&matCnt, sizeof(int), 1, f);
	_M.resize(matCnt);
	for (int i = 0; i < matCnt; i++) {
		int headData[3]; // Width, Height, Type
		fread(headData, sizeof(int), 3, f);
		Mat M(headData[1], headData[0], headData[2]);
		fread(M.data, sizeof(char), M.step * M.rows, f);
		M.copyTo(_M[i]);
	}
	std::fclose(f);
	return true;
}

void HSuperpixel::bsds_bench(const string& resultDir, const string & gtDir, const int numTest, vector<string> &testSet)
{
	CStr _FResDir = resultDir;
	//cout << _FResDir << endl; 
	if (GetFileAttributesA(_S(_FResDir)) == INVALID_FILE_ATTRIBUTES)
		CmFile::MkDir(_FResDir);
	for (int i = 0; i < numTest; i++)
	{
		vecM tmp;
		tmp.push_back(_labels[i]);
		matWrite(_FResDir + testSet[i], tmp);
	}

	allBench(testSet, gtDir, _FResDir, _FResDir + "eval/", 1, 0.0039f, false);
	plot_eval(_FResDir + "eval/");
}



