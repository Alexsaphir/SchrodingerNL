#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <string>

#include "Axis.h"
#include "Signal.cuh"
#include "SignalFFT.cuh"


//2*M_PI
#define M_PI2 6.2831853071795864769252867665590057683943387987502

#define N_FFT 512 //Frequency Sampling*Duration

void exportData(const Axis *X, const Signal *S, const std::string &name)
{
	std::ofstream file;
	file.open(name);
	file << 2 << std::endl;
	for (int i = 0; i < S->getSignalPoints(); ++i)
		file << X->getLinearValueAt(i) << " " << S->getHostData()[i].x << " " << S->getHostData()[i].y << "\n";//(S->getHostData()[i].x<0?-1.:1)*
	file.close();
}

void exportData(const Axis *X, SignalFFT *S, const std::string &name)
{//Export data of the host
	std::ofstream file;
	file.open(name);
	file << 1 << std::endl;
	for (int i = 0; i < S->getSignalPoints(); ++i)
		file << X->getFrequency(i) << " " << (S->getHostData()[i].x<0 ? -1. : 1.)*cuCabs(S->getHostData()[i]) << "\n";
	file.close();
}

void computeError(const Signal *S1, const Signal *S2, Signal *E)
{
	//S1, S2, E: same size
	for (int i = 0; i < S1->getSignalPoints(); ++i)
	{
		E->getHostData()[i] = S1->getHostData()[i] - S2->getHostData()[i];
	}
}

int main()
{
	Axis X(-1, 1, N_FFT);
	
	Signal S(-1, 1, N_FFT);//Input signal
	Signal Sout(-1, 1, N_FFT);//Output Signal

	SignalFFT Sfft(N_FFT);//FFT Signal

	GaussPulseLinear(&S, 5.);
	exportData(&X, &S, "Plot/data.ds");//Save the initial signal

	Sfft.computeFFT(&S);
	Sfft.syncDeviceToHost();//Send data to RAM
	Sfft.reorderData();//Shift Frequency for the data on the host
	exportData(&X, &Sfft, "Plot/dataFFT.ds");//Save FFT of the Signal computed

	//Sfft.smoothFilterCesaro();//Apply Filtering
	//Sfft.smoothFilterLanczos();
	Sfft.smoothFilterRaisedCosinus();

	Sfft.ComputeSignal(&Sout);//Get back the signal in physical space
	Sout.syncDeviceToHost();//Send data to RAM

	exportData(&X, &Sout, "Plot/dataN.ds");
	
	//getchar();
	return 0;
}
