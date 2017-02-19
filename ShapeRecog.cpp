
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include<cv.h>
#include <iostream>  
#include "math.h"  
#include "time.h"
#include<fstream>
#include <Eigen/Dense>  
#include<Eigen/Eigen>
#include <Eigen/Eigenvalues>
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using namespace cv;
using namespace std;
#define MAXVEX 100
CvMat*cam_intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
int sliderPos = 70;
Mat image;
Matrix3d K = Matrix3d::Zero(3, 3);
vector<RotatedRect> dox;					   //新轮廓拟合的 RotatedRect集

//========================================= 图分类 ==============================================================

typedef int Status;	// Status是函数的类型,其值是函数结果状态代码，如OK等 
typedef char VertexType; // 顶点类型应由用户定义  
typedef int EdgeType; // 边上的权值类型应由用户定义 
typedef struct
{
	//VertexType vexs[MAXVEX];  顶点表 
	EdgeType arc[MAXVEX][MAXVEX];// 邻接矩阵，可看作边表 
	int numNodes, numEdges; // 图中当前的顶点数和边数  
}MGraph;
bool visited[MAXVEX];
vector<vector<int>> temp1111(1000); // 分类轮廓 类标签 即第一组（1，3，7，6，5） 第二组（2，4，8，9，0）
int k = -1;
MatrixXd G = MatrixXd::Zero(3,3);

void computePosition(Matrix3d K,double R, MatrixXd g)
{
	//声明变量
	double u1, u2, u3, la1, la2, la3;
	Vector3d e1, e2, e3, f1, f2, f3;
	//=============q=K'gK==============
	/*Matrix3d K;
	K << 193.148468018, 0, 99.1016159058,
		0, 192.275054932, 121.850112915,
		0, 0, 1;*/
	Matrix3d q;
	q = K.transpose()*g*K;


	//==============对矩阵进行特征分解============================
	EigenSolver<Matrix3d> es(q);
	Matrix3d u = es.pseudoEigenvalueMatrix();
	Matrix3d e = es.pseudoEigenvectors();

	//cout << u << endl;
	//cout << e << endl;

	//=============提取特征值和特征向量===========================
	u1 = u(0, 0);
	u2 = u(1, 1);
	u3 = u(2, 2);
	e1 = e.col(0);
	e2 = e.col(1);
	e3 = e.col(2);

	//============找出同号的特征值，并对lambda赋值================
	if (u1*u2 > 0)
	{
		if (abs(u1) > abs(u2))
		{
			la1 = u1;
			la2 = u2;
			la3 = u3;
			f2 = e2;
			f3 = e3;
		}
		else
		{
			la1 = u2;
			la2 = u1;
			la3 = u3;
			f2 = e1;
			f3 = e3;
		}
	}
	else if (u1*u3>0)
	{
		if (abs(u1) > abs(u3))
		{
			la1 = u1;
			la2 = u3;
			la3 = u2;
			f2 = e3;
			f3 = e2;
		}
		else
		{
			la1 = u3;
			la2 = u1;
			la3 = u2;
			f2 = e1;
			f3 = e2;
		}
	}
	else
	{
		if (abs(u2) > abs(u3))
		{
			la1 = u2;
			la2 = u3;
			la3 = u1;
			f2 = e3;
			f3 = e1;
		}
		else
		{
			la1 = u3;
			la2 = u2;
			la3 = u1;
			f2 = e2;
			f3 = e1;
		}
	}

	//=============计算对应的特征向量==================
	if (f3(2) > 0)
	{
		e3 = f3;
	}
	else
	{
		e3 = -f3;
	}
	e2 = f2;
	e1 = e2.cross(e3);

	// cout << "la1=\n"<<la1 << endl << "la2=\n"<<la2 <<endl<<"la3=\n"<< la3 << endl;
	//cout << "e1=\n"<<e1 << endl <<"e2=\n"<< e2 << endl <<"e3=\n"<< e3 << endl; 


	//================求对应后的旋转矩阵================
	Matrix3d pTrans;
	pTrans << e1(0), e2(0), e3(0),
		e1(1), e2(1), e3(1),
		e1(2), e2(2), e3(2);
	//cout << "pTrans=\n" << pTrans << endl;

	//===============计算圆心坐标及法向量===============
	Vector3d O1, N1, O2, N2;

	O1 << R*sqrt(abs(la3)*(abs(la1) - abs(la2)) / (abs(la1)*(abs(la1) + abs(la3)))),
		0,
		R*sqrt(abs(la1)*(abs(la2) + abs(la3)) / (abs(la3)*(abs(la1) + abs(la3))));
	N1 << sqrt((abs(la1) - abs(la2)) / (abs(la1) + abs(la3))),
		0,
		-sqrt((abs(la2) + abs(la3)) / (abs(la1) + abs(la3)));
	O2 << -R*sqrt(abs(la3)*(abs(la1) - abs(la2)) / (abs(la1)*(abs(la1) + abs(la3)))),
		0,
		R*sqrt(abs(la1)*(abs(la2) + abs(la3)) / (abs(la3)*(abs(la1) + abs(la3))));
	N2 << -sqrt((abs(la1) - abs(la2)) / (abs(la1) + abs(la3))),
		0,
		-sqrt((abs(la2) + abs(la3)) / (abs(la1) + abs(la3)));

	//cout << "O1=\n" << O1 << endl << "N1=\n" << N1 << endl << "O2=\n" << O2 << endl << "N2=\n" << N2 << endl;


	//===============投影到相机坐标系下圆心坐标为Of1，Of2，法向量为Nf1，Nf2===========

	Vector3d Of1, Of2, Nf1, Nf2;
	Of1 = pTrans*O1;
	Nf1 = pTrans*N1;
	Of2 = pTrans*O2;
	Nf2 = pTrans*N2;
	cout << "Of1=\n" << Of1 << endl << "Nf1=\n" << Nf1 << endl << "Of2=\n" << Of2 << endl << "Nf2=\n" << Nf2 << endl;


}

void calculatefunc(vector<RotatedRect> dox,MatrixXd &G){
	int size_dox = dox.size();
	
	double A, B, C, D, E, F;
	double a, b, thetaOfRat, xOfCenter, yOfCenter;
	a = max(dox[0].size.height, dox[0].size.width); 
	b = min(dox[0].size.height, dox[0].size.width);
	xOfCenter = dox[0].center.x;
	yOfCenter = dox[0].center.y;
	thetaOfRat = dox[0].angle*3.14 / 180 - 3.14 / 2;
	A = a*a*sin(thetaOfRat)*sin(thetaOfRat) + b*b*cos(thetaOfRat)*cos(thetaOfRat);
	B = 2 * (a*a - b*b)*sin(thetaOfRat)*cos(thetaOfRat);
	C = b*b*sin(thetaOfRat)*sin(thetaOfRat) + a*a*cos(thetaOfRat)*cos(thetaOfRat);
	D = -2 * A*xOfCenter - B*yOfCenter;
	E = -2 * C*yOfCenter - B*xOfCenter;
	F = -2 * a*a*b*b + A*xOfCenter*xOfCenter + B*xOfCenter*yOfCenter + C*yOfCenter*yOfCenter;
	G << A, B / 2, D / 2,
		B / 2, C, E / 2,
		D, E / 2, F;
	
}

void save_result(CvMat*cam_rotation_all, CvMat*cam_translation_vector_all,
	CvMat*cam_intrinsic_matrix, CvMat*cam_distortion_coeffs, char*pathc, int sucesses)
{
	fstream Yeah_result;
	Yeah_result.open(pathc, ofstream::out);
	Yeah_result << setprecision(12) << "fc[0] =" << CV_MAT_ELEM(*cam_intrinsic_matrix, float, 0, 0
		) << "; fc[1] =" << CV_MAT_ELEM(*cam_intrinsic_matrix, float, 1, 1) << "; //CAM的焦距\n";
	Yeah_result << setprecision(12) << "cc[0] = " << CV_MAT_ELEM(*cam_intrinsic_matrix, float, 0, 2)
		<< "; cc[1] = " << CV_MAT_ELEM(*cam_intrinsic_matrix, float, 1, 2) << ";//CAM中心点\n";
	Yeah_result << setprecision(12) << "kc[0] =" << CV_MAT_ELEM(*cam_distortion_coeffs, float, 0, 0) <<
		"; kc[1] =" << CV_MAT_ELEM(*cam_distortion_coeffs, float, 1, 0) <<
		";  kc[2] =" << CV_MAT_ELEM(*cam_distortion_coeffs, float, 2, 0) <<
		";  kc[3] =" << CV_MAT_ELEM(*cam_distortion_coeffs, float, 3, 0)
		<< ";  kc[4] =0;//畸变参数，请参照MATLAB里的定义\n\n外参数:\n";
	for (int i = 0; i<sucesses; ++i)
	{
		Yeah_result << "r:(" << setprecision(12) << CV_MAT_ELEM(*cam_rotation_all, float, i, 0) << "\t," << CV_MAT_ELEM(*cam_rotation_all, float, i, 1) << "\t," << CV_MAT_ELEM(*cam_rotation_all, float, i, 2) << ")\n";
		Yeah_result << "t:(" << setprecision(12) << CV_MAT_ELEM(*cam_translation_vector_all, float, i, 0) << "\t," << CV_MAT_ELEM(*cam_translation_vector_all, float, i, 1) << "\t," << CV_MAT_ELEM(*cam_translation_vector_all, float, i, 2) << ")\n\n\n";
	}


}

void MatrixConvert(CvMat &a, Matrix3d &K){

	K(2, 2) = 1;
	K(1, 1) = CV_MAT_ELEM(*cam_intrinsic_matrix, float, 1, 1);
	K(0, 0) = CV_MAT_ELEM(*cam_intrinsic_matrix, float, 0, 0);

}
Matrix3d calculateInitMatrixK(){
	CvMat*cam_object_points2;
	CvMat*cam_image_points2;
	int cam_board_n;
	int successes = 0;
	int img_num, cam_board_w, cam_board_h, cam_Dx, cam_Dy;
	cout << "输入的图像的组数\n";
	cin >> img_num;
	cout << "输入**真实**棋盘格的##横轴##方向的角点个数\n";
	cin >> cam_board_w;
	cout << "输入**真实**棋盘格的##纵轴##方向的角点个数\n";
	cin >> cam_board_h;
	cout << "输入**真实**棋盘格的##横轴##方向的长度\n";
	cin >> cam_Dx;
	cout << "输入**真实**棋盘格的##纵轴##方向的长度\n";
	cin >> cam_Dy;
	cam_board_n = cam_board_w*cam_board_h;
	/*
	//init
	//
	//
	*/

	//camera init
	CvSize cam_board_sz = cvSize(cam_board_w, cam_board_h);
	CvMat*cam_image_points = cvCreateMat(cam_board_n*(img_num), 2, CV_32FC1);
	CvMat*cam_object_points = cvCreateMat(cam_board_n*(img_num), 3, CV_32FC1);
	CvMat*cam_point_counts = cvCreateMat((img_num), 1, CV_32SC1);
	CvPoint2D32f*cam_corners = new CvPoint2D32f[cam_board_n];
	int cam_corner_count;
	int cam_step;
	//CvMat*cam_intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
	CvMat*cam_distortion_coeffs = cvCreateMat(4, 1, CV_32FC1);
	CvSize cam_image_sz;
	//window intit
	cvNamedWindow("window", 0);

	//get image size
	//IplImage *cam_image_temp = cvLoadImage("..\\cam\\cam1.bmp", 0);
	IplImage *cam_image_temp = cvLoadImage("..\\cam\\cam1.jpg", 0);
	cam_image_sz = cvGetSize(cam_image_temp);
	char failurebuf[20] = { 0 };
	/*
	//extract cornner
	// camera image
	//
	// pattern
	*/




	/*
	//extrat the cam cornner
	//
	//
	//
	*/
	fstream cam_data;
	cam_data.open("..\\output\\TXT\\cam_corners.txt", ofstream::out);
	fstream cam_object_data;
	cam_object_data.open("..\\output\\TXT\\cam_object_data.txt", ofstream::out);
	//process the prj image so that we can easy find cornner
	for (int ii = 1; ii < img_num + 1; ii++)
	{
		char cambuf[20] = { 0 };
		//sprintf(cambuf, "..\\cam\\cam%d.bmp", ii);
		sprintf(cambuf, "..\\cam\\cam%d.jpg", ii);
		IplImage *cam_image = cvLoadImage(cambuf, 0);

		//extract cam cornner
		int cam_found = cvFindChessboardCorners(cam_image, cam_board_sz, cam_corners, &cam_corner_count,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		cvFindCornerSubPix(cam_image, cam_corners, cam_corner_count,
			cvSize(11, 11), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cvDrawChessboardCorners(cam_image, cam_board_sz, cam_corners, cam_corner_count, cam_found);

		if (cam_corner_count != cam_board_n)
			cout << "find cam" << ii << "  corner failed!\n";

		//when cam and prj are success store the result
		if (cam_corner_count == cam_board_n) {
			//store cam result
			cam_step = successes*cam_board_n;
			for (int i = cam_step, j = 0; j < cam_board_n; ++i, ++j) {
				CV_MAT_ELEM(*cam_image_points, float, i, 0) = cam_corners[j].x;
				CV_MAT_ELEM(*cam_image_points, float, i, 1) = cam_corners[j].y;
				CV_MAT_ELEM(*cam_object_points, float, i, 0) = (j / cam_board_w)*cam_Dx;
				CV_MAT_ELEM(*cam_object_points, float, i, 1) = (j % cam_board_w)*cam_Dy;
				CV_MAT_ELEM(*cam_object_points, float, i, 2) = 0.0f;
				cam_data << cam_corners[j].x << "\t" << cam_corners[j].y << "\n";
				cam_object_data << (j / cam_board_w)*cam_Dx << "\t" << (j %cam_board_w)*cam_Dy << "\t0\n";
			}
			CV_MAT_ELEM(*cam_point_counts, int, successes, 0) = cam_board_n;
			successes++;
			cout << "success number" << successes << endl;
			cvShowImage("window", cam_image);
			cvWaitKey(500);
		}



	}
	if (successes < 2)
		exit(0);
	/*
	//restore the success point
	*/
	//cam
	cam_image_points2 = cvCreateMat(cam_board_n*(successes), 2, CV_32FC1);
	cam_object_points2 = cvCreateMat(cam_board_n*(successes), 3, CV_32FC1);
	CvMat*cam_point_counts2 = cvCreateMat((successes), 1, CV_32SC1);
	for (int i = 0; i < successes*cam_board_n; ++i){
		CV_MAT_ELEM(*cam_image_points2, float, i, 0) = CV_MAT_ELEM(*cam_image_points, float, i, 0);
		CV_MAT_ELEM(*cam_image_points2, float, i, 1) = CV_MAT_ELEM(*cam_image_points, float, i, 1);
		CV_MAT_ELEM(*cam_object_points2, float, i, 0) = CV_MAT_ELEM(*cam_object_points, float, i, 0);
		CV_MAT_ELEM(*cam_object_points2, float, i, 1) = CV_MAT_ELEM(*cam_object_points, float, i, 1);
		CV_MAT_ELEM(*cam_object_points2, float, i, 2) = CV_MAT_ELEM(*cam_object_points, float, i, 2);

	}
	for (int i = 0; i < successes; ++i){
		CV_MAT_ELEM(*cam_point_counts2, int, i, 0) = CV_MAT_ELEM(*cam_point_counts, int, i, 0);
	}
	cvSave("..\\output\\XML\\cam_corners.xml", cam_image_points2);

	cvReleaseMat(&cam_object_points);
	cvReleaseMat(&cam_image_points);
	cvReleaseMat(&cam_point_counts);


	/*
	//calibration for camera
	//
	*/
	//calib for cam
	CV_MAT_ELEM(*cam_intrinsic_matrix, float, 0, 0) = 1.0f;
	CV_MAT_ELEM(*cam_intrinsic_matrix, float, 1, 1) = 1.0f;
	CvMat* cam_rotation_all = cvCreateMat(successes, 3, CV_32FC1);
	CvMat* cam_translation_vector_all = cvCreateMat(successes, 3, CV_32FC1);
	cvCalibrateCamera2(
		cam_object_points2,
		cam_image_points2,
		cam_point_counts2,
		cam_image_sz,
		cam_intrinsic_matrix,
		cam_distortion_coeffs,
		cam_rotation_all,
		cam_translation_vector_all,
		0//CV_CALIB_FIX_ASPECT_RATIO  
		);
	cvSave("..\\output\\XML\\cam_intrinsic_matrix.xml", cam_intrinsic_matrix);
	cvSave("..\\output\\XML\\cam_distortion_coeffs.xml", cam_distortion_coeffs);
	//calib 
	cvSave("..\\output\\XML\\cam_rotation_all.xml", cam_rotation_all);
	cvSave("..\\output\\XML\\cam_translation_vector_all.xml", cam_translation_vector_all);
	char path1[100] = "..\\output\\result_data_no_optim.txt";
	save_result(cam_rotation_all, cam_translation_vector_all,
		cam_intrinsic_matrix, cam_distortion_coeffs, path1, successes);
}
//================================================ 判断两个  rect 矩形 相交的面积 ====================================
float bbOverlap(const Rect& box1, const Rect& box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}
void DFS(MGraph G, int i, int k)
{
	int j;
	visited[i] = true;
	printf("%d ", i);

	temp1111[k].push_back(i);
	for (j = 0; j < G.numNodes; j++)
	if (G.arc[i][j] == 1 && !visited[j])
		DFS(G, j, k);
}

void DFSTraverse(MGraph G)
{

	int i;
	for (i = 0; i < G.numNodes; i++)
		visited[i] = false;
	for (i = 0; i < G.numNodes; i++){
		if (!visited[i])

		{

			k += 1;
			cout << endl;
			cout << "第" << k << "类" << endl;
			DFS(G, i, k);
			cout << endl;
		}

	}
}

void CreateMGraph(MGraph *G, vector<Point2d> &baocun, int n)// 建立无向网图的邻接矩阵表示
{
	int i, j, k;
	G->numEdges = baocun.size();
	G->numNodes = n;

	for (i = 0; i < G->numNodes; i++)
	for (j = 0; j < G->numNodes; j++)
		G->arc[i][j] = 0;
	for (k = 0; k <G->numEdges; k++)
	{


		int i = baocun[k].x; int j = baocun[k].y;
		G->arc[i][j] = 1;
		G->arc[j][i] = G->arc[i][j];
	}
}
void processImage(int h, void*)
{
	vector<vector<Point> > contours;                        //点轮廓 vector
	vector<vector<Point> > baocun1(1000);                   // 新的轮廓
	vector<Rect> cox;                                      //旧轮廓 -- >> RotatedRect -->>Rect 用来判断相交面积
	vector<RotatedRect> box;                               //旧轮廓拟合的 RotatedRect
	RotatedRect aox;                                       //新轮廓拟合的 RotatedRect
	vector<Point2d>  baocun;                               //图的边 即 （1，3）（2，4）etc.

	MGraph G;
	Mat bimage = image >= sliderPos;

	findContours(bimage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cout << contours.size() << endl;



	//在cimage上面绘图  
	Mat cimage = Mat::zeros(bimage.size(), CV_8UC3);
	//drawContours(cimage, contours, 15, Scalar::all(255), 1, 8);
	/*drawContours(cimage, contours, 15, Scalar::all(255), 1, 8);
	imshow("result", cimage);
	system("pause");*/


	for (size_t i = 0; i < contours.size(); i++)
	{

		size_t count = contours[i].size();

		if (count < 6)
			continue;



		box.push_back(fitEllipse(contours[i]));


		cox.push_back(box[i].boundingRect());
	}

	for (size_t i = 0; i < contours.size() - 1; i++){
		for (int j = i + 1; j < contours.size() - 1; j++){
			float temp = bbOverlap(cox[i], cox[j]);
			if (temp > 0){                               //判断两个轮廓是否有关系  阈值为0.1
				//cout << temp << endl;
				Point2d cc(i, j);
				baocun.push_back(cc);
			}


		}
	}
	//===============================================  图处理 归类椭圆弧  ============================================
	CreateMGraph(&G, baocun, contours.size());//baocun 是图的边， contours.size()是轮廓的个数，也就是图中Nodes的数目
	DFSTraverse(G);
	for (int i = 0; i < k + 1; i++){
		for (int j = 0; j < temp1111[i].size(); j++){
			if (j == 0){
				int k = temp1111[i][j];
				baocun1[i] = contours[k];
			}
			int kk = temp1111[i][j];
			baocun1[i].insert(baocun1[i].end(), contours[kk].begin(), contours[kk].end());

		}
	}


	for (size_t i = 0; i < k; i++)
	{

		size_t count = baocun1[i].size();

		if (count < 200){
			cout << "去除第" << i << "类" << endl;
			continue;
		}
		aox = fitEllipse(baocun1[i]);
		if (MAX(aox.size.width, aox.size.height) > MIN(aox.size.width, aox.size.height) * 8)
		{
			cout << "去除第" << i << "类" << endl;
			continue;
		}
		//绘制轮廓  
		drawContours(cimage, baocun1, (int)i, Scalar::all(255), 1, 8);
		//cout << i << endl;

		//绘制椭圆  
		ellipse(cimage, aox, Scalar(0, 0, 255), 1, CV_AA);
		dox.push_back(aox);
		//绘制椭圆  
		// ellipse(cimage, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0,255,255), 1, CV_AA);  
		std::cout << "第" << i << "类" << "size = " << aox.size << " , "
			<< "center =" << aox.center << " , "
			<< "angle = " << aox.angle << std::endl;

		//绘制矩形框  
		Point2f vtx[4];
		//成员函数points 返回 4个矩形的顶点(x,y)  
		aox.points(vtx);
		for (int j = 0; j < 4; j++)
			line(cimage, vtx[j], vtx[(j + 1) % 4], Scalar(0, 255, 0), 1, CV_AA);
		imshow("result", cimage);
		char key = waitKey();
		if (key == 27)
			continue;

	}

}
int main(int argc, char** argv)
{
	const char* filename = "..//img//test_1.png";
	image = imread(filename, 0);
	if (image.empty())
	{
		cout << "Couldn't open image " << endl;
		return 0;
	}
	GaussianBlur(image, image, Size(7, 7), 0, 0);       //高斯模糊（Gaussian Blur）
	Canny(image, image, 100, 200, 3);

	imshow("source", image);
	namedWindow("result", 1);

	// Create toolbars. HighGUI use.  
	// 创建一个滑动块  
	createTrackbar("threshold", "result", &sliderPos, 255, processImage);
	processImage(0, 0);
	calculateInitMatrixK();
	MatrixConvert(*cam_intrinsic_matrix, K);
	calculatefunc(dox,G);
	double radius;
	cout << "输入圆的半径：\n";
	cin >> radius;

	computePosition(K, radius, G);

	// Wait for a key stroke; the same function arranges events processing  
	waitKey();
	return 0;

}

