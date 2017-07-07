#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include "Grid.cuh"
#include "SplitStepSolver.cuh"
#include "NLSUtility.h"
#include "ExportData.h"

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
	Xmin = -100;
	Xmax = 100;
	dt = .00001;

	For1 = 1000;
	For2 = 2000;

	bool b_demo(true);


	

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
	NLSUtility::GaussPulseLinear(S, 1, .5, -6, -60);
	
	double t(0);

	for (int i = 0; i <= For1; ++i)
	{
		//exportData(S, "Plot/data" + std::to_string(i) + ".ds");
		exportCsvXY(S, "Plot/data" + std::to_string(i) + ".csv");
		std::cout << t << std::endl;
		if (i == For1)
			break;
		for (int j = 0; j < For2; ++j)
		{
			SplitStep(S->getDeviceData(), dt, N_FFT, Xmax - Xmin, &plan);
			t += dt;
		}
	}
	
	
	
	//getchar();
	return 0;
}
