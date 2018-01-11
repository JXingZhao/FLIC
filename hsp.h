/*
 * Author: Qibin Hou
 * Email: andrewhoux@gmail.com
 */

#ifndef HSP_H
#define HSP_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include<math.h>
#include "utils.h"
#include "definitions.h"
#include "CmIllu.h"
enum Directions { LEFT_TO_RIGHT, RIGHT_TO_LEFT, TOP_TO_BOTTOM, BOTTOM_TO_TOP };

class HSuperpixel
{
public:
    HSuperpixel();
    ~HSuperpixel();
	int spcount;
	int move_number;
	int itr;
	void segment(const std::string& imgName, const std::string& outputPath, const int sparea);
	void segment_for_evaluation(const std::string& imgName, const int sparea, int &numLabels,int itr_1);
    void set_parameters(const double c, const double s) 
    {
        this->c = c;
    }
	void bsds_bench(const std::string& resultDir, const string & gtDir, const int numTest, std::vector<std::string> &testSet);
	bool matRead(CStr &filename, vecM &_M);
	std::vector<cv::Mat> _labels;
private:

	inline double compute_distance(const Component &center, const Pixel &point,const double &area,int sign);
	inline void update_start_point(Component &component, Pixel &point);
	inline void update_score();
	inline double compute_color_distance(const Pixel& point, const Pixel & point2)
	{
		double colorDiff = Utils::square(point2.l - point.l) + Utils::square(point2.a - point.a) + Utils::square(point2.b - point.b);
		return sqrt(colorDiff);
	}
   
	void forward_clustering(const cv::Mat &img, cv::Mat &imgLabels);
    void update_sp_center(Component &component);

	void update_components(Mat &img, Mat &imgLabels);
	void initial_update_components(Mat &img, Mat &imgLabels);

    void score_each_sp(Component &component);

	bool matWrite(CStr &filename, vecM &_M);
	
	void release_components();
	void enforce_label_connectivity(const int* labels, const int	width, const int height, int*& nlabels,
		int& numlabels);
	void transform_from_mat(Mat &mat, int *buff);
	void transform_to_mat(int *buff, Mat &mat);
	void get_output_name(const std::string &src, std::string &out);
	void draw_borders(cv::Mat &img, cv::Mat &imgLabels);
	void draw_borders_manifold(unsigned int*& ubuff, int*& labels);
	void init(cv::Mat &img, cv::Mat &imgLabels, const Spsize &spsize);
	void load_image(const std::string& imgName, cv::Mat &out);
	void show_image(const cv::Mat &img, const std::string &name);
	void show_image(const cv::Mat &img, const char *name);
	void Merge(cv::Mat &imgLabels);
	void transform_to_mat_img(unsigned int *buff, Mat &mat);
	void transform_from_mat_img(Mat &mat, unsigned int *buff);
private:
	
    Component *_components;
    int _width;
    int _height;
    int _channel;
    int _numsp;
	int _xstrips;
	int _ystrips;
    double c;
	bool is_test;
	std::vector<vector<Point2i>>origin_point;
};

inline double HSuperpixel::compute_distance(const Component &center, const Pixel &point,const double &area,int sign)
{
	double colorDiff = Utils::square(center.lMean - point.l) + Utils::square(center.aMean - point.a) + Utils::square(center.bMean - point.b);
	double spaceDiff = Utils::square(center.xMean - point.x) + Utils::square(center.yMean - point.y);
	double diff = 0.f;
	if (sign == 0)
		diff = colorDiff;
	else if (sign == 1)
	{
		
		diff = colorDiff + spaceDiff * c ;
	}
	return diff;
}

inline void HSuperpixel::update_start_point(Component &component, Pixel &point)
{
	if (point.x > component.max_x)
		component.max_x = point.x;
	if (point.x < component.min_x)
		component.min_x = point.x;
	if (point.y > component.max_y)
		component.max_y = point.y;
	if (point.y < component.min_y)
		component.min_y = point.y;
}
inline void HSuperpixel:: update_score()
{
	for (int i = 0; i < _numsp; ++i)
	{
		score_each_sp(_components[i]);	
	}
}
#endif
