#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include "Grid.cuh"
#include "SplitStepSolver.cuh"
#include "NLSUtility.h"
#include "ExportData.h"

#include <vector>
#include <algorithm>

//2*M_PI
#define M_PI2 6.2831853071795864769252867665590057683943387987502

#define N_FFT 4096//Frequency Sampling*Duration


int main()
{
	int N;
	double Xmin;
	double Xmax;
	double dt;

	int For1;
	int For2;

	N = N_FFT;
	Xmin = -2;
	Xmax = 2;
	dt = .001;

	//For1 = 100000;
	For1 = 100;
	For2 = 50;

	bool b_demo(true);
	//b_demo = false;


	

	std::cout << "Begin to solve NLS : \n" << "Please enter parameters !\n";
	std::cout << "Xmin :";
	if (!b_demo)
	{
		std::cin >> Xmin;
	}
	std::cout << "\nXmax :";
	if (!b_demo)
	{
		std::cin >> Xmax;
	}
	std::cout << "\nN :";
	if (!b_demo)
	{
		std::cin >> N;
	}
	std::cout << "\ndt :";
	if (!b_demo)
	{
		std::cin >> dt;
	}
	std::cout << "\n\nIter Loop 1:";
	if (!b_demo)
	{
		std::cin >> For1;
	}
	std::cout << "\nIter Loop 2:";
	if (!b_demo)
	{
		std::cin >> For2;
	}

	std::cout << "\n\n\nAxis X :";
	std::cout << "\n\t[Xmin , Xmax] , N :[" << Xmin << " , " << Xmax << "] , " << N;
	std::cout << "\n\tSpatial resolution :" << (Xmax - Xmin) / static_cast<double>(N);
	std::cout << "\n\tFrequency resolution :"<< 2.*M_PI / static_cast<double>(N);
	std::cout << "\nTime resolution :" << static_cast<double>(For2)*dt;
	std::cout << "\nTotal time :" << static_cast<double>(For2*For1)*dt;
	std::cout << "\n\n\nBegin computing!\n";
	
	//FFT plan
	cufftHandle plan;
	cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
	
	Grid *S = new Grid(Xmin, Xmax, N);
	NLSUtility::GaussPulseLinear(S, 50, .5, -6, -60);
	
	//Shift pulse +25

	//Shift pulse with add -50
	//S->syncDeviceToHost();

	//std::vector<cmplx> A;
	//std::vector<cmplx> B;

	//for (int i = 0; i < N; ++i)
	//{
		//A.push_back(S->getHostData()[(i + N / 4) % N]);
		//B.push_back(S->getHostData()[(i + 3 * N / 4) % N]);
		//std::cout << (i + N / 2) % N << "\t" << (i + 3 * N / 2) % N << "\n";
	//}
	//for (int i = 0; i < N; ++i)
	//{
		//S->getHostData()[i] = cuCexp(iMul(-M_PI))*A.at(i) + B.at(i);
	//}
	

	//A.clear();
	//B.clear();
	
	//S->syncHostToDevice();

	double t(0);
	
	for (int i = 0; i <= For1; ++i)
	{
		//exportData(S, "Plot/data" + std::to_string(i) + ".ds");
		exportCsvXY(S, "B:/data" + std::to_string(i) + ".csv");
		exportCsvXTY(S, "Plot/dataT.csv", i);
		//exportCsv2DMatlab(S, "B:/Matlab.csv", i);
		double M = NLSUtility::computeTotalMass(S);
		exportMassOverTime(M, "B:/dataE.csv", i);

		
		std::cout << "Time :" << t << "\tM :" << M << std::endl;
		if (i == For1)
			break;
		for (int j = 0; j < For2; ++j)
		{
			SplitStep(S->getDeviceData(), dt, N_FFT, Xmax - Xmin, &plan,1);
			t += dt;
		}
	}
	//getchar();
	return 0;
}
