
#include <string>
#include <vector>
#include <iostream>
#include <Windows.h>
#include <assert.h>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "Bench.h"
#include "../definitions.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace cv;

#define CV_Assert_(expr, args) \
{\
    if (!(expr)) { \
	   string msg(args); \
	   printf("%s in %s:%d\n", msg.c_str(), __FILE__, __LINE__); \
	   cv::error(cv::Exception(CV_StsAssert, msg, __FUNCTION__, __FILE__, __LINE__)); \
		}\
}

void plot_eval(const string evalDir, const string col)
{
	FILE *file = fopen((evalDir + "result.m").c_str(), "w");
	fprintf(file, "clear all;\nclc;\n\n\n");
	fprintf(file, "h = figure;\ntitle('Boundary Benchmark on BSDS');\nhold on;\n\n");
	fprintf(file, "plot(0.700762,0.897659,'go','MarkerFaceColor','g','MarkerEdgeColor','g','MarkerSize',10);\n\n");
	fprintf(file, "[p,r] = meshgrid(0.01:0.01:1,0.01:0.01:1);\n");
	fprintf(file, "F = 2*p.*r./(p+r);\n");
	fprintf(file, "[C,h] = contour(p,r,F);\n\n");
	fprintf(file, "map = zeros(256,3); map(:,1) = 0; map(:,2) = 1; map(:,3) = 0; colormap(map);\n\n");
	fprintf(file, "box on;\ngrid on;\n");
	fprintf(file, "set(gca,'XTick',0:0.1:1);\nset(gca,'YTick',0:0.1:1);\n");
	fprintf(file, "set(gca,'XGrid','on');\nset(gca,'YGrid','on');\n");
	fprintf(file, "xlabel('Recall');\nylabel('Precision');\n");
	fprintf(file, "title('');\naxis square;\naxis([0 1 0 1]);\n\n\n");

	FILE *fid = fopen((evalDir + "eval_bdry_thr.txt").c_str(), "r");
	if (fid == NULL)
	{
		cout << "Can't open file eval_bdry_thr.txt for reading!" << endl;
		exit(1);
	}
	int nthresh;
	fscanf(fid, "%10d\n", &nthresh);
	vector<Vec4f> prvals(nthresh);
	for (int i = 0; i < nthresh; i++)
		fscanf(fid, "%10f %10f %10f %10f\n", &prvals[i][0], &prvals[i][1], &prvals[i][2], &prvals[i][3]);
	for (int i = 0; i < prvals.size(); i++) {
		if (prvals[i][1] < 0.01)
			prvals.erase(prvals.begin() + i--);
	}
	std::fclose(fid);

	fid = fopen((evalDir + "eval_bdry.txt").c_str(), "r");
	if (fid == NULL)
	{
		cout << "Can't open file eval_bdry.txt for reading!" << endl;
		exit(1);
	}
	vector<float> evalRes(8);
	fscanf(fid, "%10f %10f %10f %10f %10f %10f %10f %10f\n", &evalRes[0], &evalRes[1], &evalRes[2],
		&evalRes[3], &evalRes[4], &evalRes[5], &evalRes[6], &evalRes[7]);
	std::fclose(fid);

	fprintf(file, "hold on\nprvals = [");
	const int size_pr = prvals.size();
	for (int i = 0; i < size_pr - 1; i++)
		fprintf(file, "%f %f %f %f;", prvals[i][0], prvals[i][1], prvals[i][2], prvals[i][3]);
	if (size_pr >= 1)
		fprintf(file, "%f %f %f %f", prvals[size_pr - 1][0], prvals[size_pr - 1][1], prvals[size_pr - 1][2], prvals[size_pr - 1][3]);
	fprintf(file, "];\n");
	fprintf(file, "evalRes = [%f %f %f %f %f %f %f %f];\n", evalRes[0], evalRes[1], evalRes[2],
		evalRes[3], evalRes[4], evalRes[5], evalRes[6], evalRes[7]);

	if (size_pr > 1)
		fprintf(file, ("plot(prvals(1:end,2),prvals(1:end,3),'" + col + "','LineWidth',3);\n").c_str());
	else
		fprintf(file, ("plot(evalRes(2),evalRes(3),'o','MarkerFaceColor','" + col + "','MarkerEdgeColor','" + col + "','MarkerSize',8);\n").c_str());
	fprintf(file, "hold off\n");

	printf("\nBoundary\n");
	printf("ODS: F( %1.4f, %1.4f ) = %1.4f   [th = %1.2f]\n", evalRes[1], evalRes[2], evalRes[3], evalRes[0]);
	printf("OIS: F( %1.4f, %1.4f ) = %1.4f\n", evalRes[4], evalRes[5], evalRes[6]);
	printf("Area_PR = %1.4f\n", evalRes[7]);

	fid = fopen((evalDir + "eval_cover.txt").c_str(), "r");
	if (fid == NULL)
	{
		cout << "Can't open file eval_cover.txt for reading!" << endl;
		exit(1);
	}
	vector<float> evalRegRes(4);
	fscanf(fid, "%10f %10f %10f %10f\n", &evalRegRes[0], &evalRegRes[1], &evalRegRes[2], &evalRegRes[3]);
	std::fclose(fid);
	printf("\nRegion\n");
	printf("GT covering: ODS = %1.2f [th = %1.2f]. OIS = %1.2f. Best = %1.2f\n", evalRegRes[1], evalRegRes[0], evalRegRes[2], evalRegRes[3]);

	fid = fopen((evalDir + "eval_RI_VOI.txt").c_str(), "r");
	if (fid == NULL)
	{
		cout << "Can't open file eval_RI_VOI.txt for reading!" << endl;
		exit(1);
	}
	vector<float> evalRegRes1(6);
	fscanf(fid, "%10f %10f %10f %10f %10f %10f\n", &evalRegRes1[0], &evalRegRes1[1], &evalRegRes1[2],
		&evalRegRes1[3], &evalRegRes1[4], &evalRegRes1[5]);
	std::fclose(fid);
	printf("Rand Index: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n", evalRegRes1[1], evalRegRes1[0], evalRegRes1[2]);
	printf("Var. Info.: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n", evalRegRes1[4], evalRegRes1[3], evalRegRes1[5]);

	std::fclose(file);
}

void allBench(vector<string> iids, const string gtDir, const string inDir, const string outDir, int nthresh, float maxDist, bool thinpb)
{
	CmFile::MkDir(outDir);
#pragma omp parallel for
	for (int i = 0; i < iids.size(); i++) {
		const string evFile1 = outDir + iids[i] + "_ev1.bin";
		//const string evFile2 = outDir + iids[i] + "_ev2.bin";
		//const string evFile3 = outDir + iids[i] + "_ev3.bin";
		//const string evFile4 = outDir + iids[i] + "_ev4.bin";
		//if (GetFileAttributesA(evFile4.c_str()) != INVALID_FILE_ATTRIBUTES)
		//	continue;

		const string inFile = inDir + iids[i];
		const string b_gtFile = gtDir + "Boundaries/" + iids[i];
		//const string s_gtFile = gtDir + "Segmentation/" + iids[i];

		vector<int> b_thresh, g_thresh;
		vector<float> b_cntR, b_sumR, b_cntP, b_sumP, g_cntR, g_sumR, g_cntP, g_sumP;
		float cntR_best;
		evaluation_bdry_image(inFile, b_gtFile, evFile1, b_thresh, b_cntR,
			b_sumR, b_cntP, b_sumP, nthresh, maxDist, thinpb);
		//evaluation_reg_image(inFile, s_gtFile, evFile2, evFile3, evFile4, g_thresh,
		//	g_cntR, g_sumR, g_cntP, g_sumP, &cntR_best, nthresh);
	}
	collect_eval_bdry(outDir);
	//collect_eval_reg(outDir);

	vector<string> ev1, ev2, ev3, ev4;
	CmFile::GetNames(outDir + "*_ev1.bin", ev1);
	//CmFile::GetNames(outDir + "*_ev2.bin", ev2);
	//CmFile::GetNames(outDir + "*_ev3.bin", ev3);
	//CmFile::GetNames(outDir + "*_ev4.bin", ev4);
	//assert(ev1.size() == ev2.size() && ev2.size() == ev3.size() && ev3.size() == ev4.size());
	for (int i = 0; i < ev1.size(); i++)
	{
		remove((outDir + ev1[i]).c_str());
		//remove((outDir + ev2[i]).c_str());
		//remove((outDir + ev3[i]).c_str());
		//remove((outDir + ev4[i]).c_str());
	}
}