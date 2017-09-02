#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include "Grid.cuh"
#include "Grid2D.cuh"

#include "SplitStepSolver.cuh"
#include "SplitStepSolver2D.cuh"

#include "NLSUtility.h"
#include "ExportData.h"

#include <vector>
#include <algorithm>

//2*M_PI
#define M_PI2 6.2831853071795864769252867665590057683943387987502

#define N_FFT 2048//Frequency Sampling*Duration


int main()
{
	int Nx;
	double Xmin;
	double Xmax;
	
	int Ny;
	double Ymin;
	double Ymax;
	
	double dt;

	int For1;
	int For2;

	Nx = N_FFT;
	Xmin = -20;
	Xmax = 20;

	Ny = N_FFT;
	Ymin = -20;
	Ymax = 20;

	dt = .0001;

	//For1 = 100000;
	For1 = 50;
	For2 = 500;

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
	std::cout << "\nNx :";
	if (!b_demo)
	{
		std::cin >> Nx;
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
	std::cout << "\n\t[Xmin , Xmax] , Nx :[" << Xmin << " , " << Xmax << "] , " << Nx;
	std::cout << "\n\tSpatial resolution :" << (Xmax - Xmin) / static_cast<double>(Nx);

	std::cout << "\n\n\nAxis Y :";
	std::cout << "\n\t[Ymin , Ymax] , Ny :[" << Ymin << " , " << Ymax << "] , " << Ny;
	std::cout << "\n\tSpatial resolution :" << (Ymax - Ymin) / static_cast<double>(Ny);

	std::cout << "\n\tX-Frequency resolution :"<< 2.*M_PI / static_cast<double>(Nx);
	std::cout << "\n\tY-Frequency resolution :" << 2.*M_PI / static_cast<double>(Ny);
	std::cout << "\nTime resolution :" << static_cast<double>(For2)*dt;
	std::cout << "\nTotal time :" << static_cast<double>(For2*For1)*dt;
	std::cout << "\n\n\nBegin computing!\n";
	
	//FFT plan
	cufftHandle plan;
	
	
	//cufftPlan1d(&plan, Nx, CUFFT_Z2Z, 1);
	//Grid *S = new Grid(Xmin, Xmax, Nx);
	//NLSUtility::GaussPulseLinear(S, 50, .5, -6, -60);
	
	cufftPlan2d(&plan, Nx, Ny, CUFFT_Z2Z);
	Grid2D *S = new Grid2D(Xmin, Xmax, Nx, Ymin, Ymax, Ny);
	NLSUtility::GaussMonoPulse2D(S);

	double t(0);
	
	for (int i = 0; i <= For1; ++i)
	{
		//exportData(S, "Plot/data" + std::to_string(i) + ".ds");
		//exportCsvXY(S, "B:/data" + std::to_string(i) + ".csv");
		//exportCsvXTY(S, "Plot/dataT.csv", i);
		exportCsv2DMatlab(S, "B:/Matlab.csv", i);
		double M = 0;
		//M = NLSUtility::computeTotalMass(S);
		//exportMassOverTime(M, "B:/dataE.csv", i);

		
		std::cout << "Time :" << t << "\tM :" << M << std::endl;
		if (i == For1)
			break;
		for (int j = 0; j < For2; ++j)
		{
			//SplitStep(S->getDeviceData(), dt, N_FFT, Xmax - Xmin, &plan,1);
			SplitStep2D(S->getDeviceData(), dt, Nx*Ny, Nx, Ny, Xmax - Xmin, Ymax - Ymin, &plan, 1);
			
			t += dt;
		}
	}
	//getchar();
	return 0;
}
