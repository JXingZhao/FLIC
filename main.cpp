/*
 *
 *
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>
#include "hsp.h"
#include "definitions.h"
#include <sstream>
#include <direct.h>
#include <io.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

#define TEST

void usage(int argc, char *argv[])
{

	cout << "Usage: "
		<< argv[0] << "[options]"
		<< endl;
	cout << "options:" << endl;
	cout << "\t--test: only for segmentation" << endl;
	cout << "\t--bench: evaluation using the BSDS benchmark standard" << endl;
	system("pause");
	exit(1);

}

int load_image_list(const string fileName, vector<string> &str)
{
	std::ifstream infile(fileName);
	string line;
	while (getline(infile, line) && line.size())
		str.push_back(line);
	return static_cast<int>(str.size());
}

void get_output_name(const string &src, string &out, string &dest, string &tag)
{
	string::size_type p = src.rfind("/");
	string tmp = src.substr(p + 1);
	p = tmp.rfind(".");
	tmp = tmp.insert(p, tag);
	dest = out + tmp;
}

int main(int argc, char *argv[])
{

	int start = clock();
	string workDir("E:/Dataset/BSR/BSDS500/");
	string outputDir = "E:/Dataset/BSR/result/";
	const char* c_output = outputDir.data();
	if (access(c_output, 00))
	{
		rmdir(c_output);
		mkdir(c_output);
	}
	else
	{
		mkdir(c_output);
	}
	string imageDir = "E:/Dataset/BSR/BSDS500/JPEGImages/";
	vector<string> testList;
	int numTest = load_image_list(workDir + "ImageSets/Main/test.txt", testList);
	HSuperpixel hsp;
	string inputName;
	string outputName;
	string tag("_flic");
	
	
	
	int average_num = 0;
	int numLabels;
	
	for (int i = 0; i < numTest; ++i)
	{
		inputName = imageDir + testList[i] + ".jpg";		
		cout << "Processing " << inputName << " " << (i + 1) << "/" << numTest << endl;
#ifdef TEST
		get_output_name(inputName, outputDir, outputName, tag);
		outputName = outputName.substr(0, outputName.rfind('.')).append(".png");
		hsp.segment(inputName, outputName, 100);
#else
		hsp.segment_for_evaluation(inputName,200, numLabels, 2);
		average_num += numLabels;
#endif
	}
	cout << average_num / 200 << endl;
	int end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "Running time: " << time << "ms" << endl;
#ifndef TEST
	hsp.bsds_bench(outputDir, workDir + "Annotations/", numTest, testList);
#endif
	hsp._labels.clear();
	
	
	system("pause");
    
	
}
