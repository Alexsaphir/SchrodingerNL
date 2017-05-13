#include <iostream>
#include <fstream>
#include <string>


#include "Axis.h"
#include "Signal.cuh"
#include "SignalFFT.cuh"
#include "ExportData.h"
#include "WaveEquation1D.h"

#include "RungeKutta.cuh"

//2*M_PI
#define M_PI2 6.2831853071795864769252867665590057683943387987502

#define N_FFT 1024//Frequency Sampling*Duration


void computeError(const Signal *S1, const Signal *S2, Signal *E)
{
	//S1, S2, E: same size
	for (int i = 0; i < S1->getSignalPoints(); ++i)
	{
		E->getHostData()[i] = S1->getHostData()[i] - S2->getHostData()[i];
	}
}

void extrapolateFromFFTCoeff(SignalFFT* const Sfft, Signal* const S)
{//FFT data are on the device
	double N = Sfft->getSignalPoints();
	Axis X(S->getXmin(), S->getXmax(), S->getSignalPoints());
	
	Sfft->syncDeviceToHost();
	
	for (int t = 0; t < S->getSignalPoints();++t)
	{
		double x = X.getFFTValueAt(t);
		//Compute fourrier approximation
		cmplx z = make_cuDoubleComplex(0, 0);
		
		for (int i = 0; i < N; ++i)
		{
			double Freq;
			//Positive Frequency are between 0 and <N/2
			//Negative Frequency are between N/2and <N
			if (i < N / 2)
				Freq = static_cast<double>(i);
			else if (i > N / 2)
				Freq = static_cast<double>(i) - N;
			else
				Freq = 0.;//k=N/2
			z = z + Sfft->getHostData()[i] * cuCexp(iMul(Freq*x));
		}
		z = z*1./std::sqrt(N);
		S->getHostData()[t] = z;
	}
}

int main()
{
	
	
	Axis XInit(-M_PI, M_PI, N_FFT);

	Signal *S;
	Signal *SInit;
	SignalFFT *Sfc, *Sfn;
	Signal *TmpA, *TmpB;//Need less memory than SignalFFT

	SInit = new Signal(-M_PI, M_PI, N_FFT);
	Sfc = new SignalFFT(N_FFT);
	Sfn = new SignalFFT(N_FFT);
	TmpB = new Signal(-M_PI, M_PI, N_FFT);
	TmpA = new Signal(-M_PI, M_PI, N_FFT);
	

	GaussPulseLinear(SInit, 10);
	//Init Sinit and put it in Sfc
	//for (int i = 0; i < N_FFT; ++i)
	//{	
	//	double a = -1. / M_PI / M_PI;
	//	double b = 2. / M_PI;

	//	double x = XInit.getLinearValueAt(i);
	//	SInit->getHostData()[i] = make_cuDoubleComplex((a*x + b)*x, 0);
	//	//SInit->getHostData()[i] = make_cuDoubleComplex(0, 0);
	//	if (i == 0 || i == N_FFT - 1)
	//	{
	//		SInit->getHostData()[i] = make_cuDoubleComplex(0, 0);
	//	}
	//}

	SInit->syncHostToDevice();
	Sfc->computeFFT(SInit);
	delete SInit;

	S = new Signal(-M_PI, M_PI, N_FFT * 4);
	Axis X(-M_PI, M_PI, N_FFT*4);
	
	
	double dt = .001;
	double t = 0;

	for (int i = 0; i < 10; ++i)
	{
		extrapolateFromFFTCoeff(Sfc, S);
		exportData(&X, S, "Plot/data" + std::to_string(i) + ".ds");
		std::cout << t << std::endl;
		if (i == 9)
			break;

		for (int j = 0; j < 20000; ++j)
		{
			RungeKutta4(Sfc->getDeviceData(), Sfn->getDeviceData(), TmpA->getDeviceData(), TmpB->getDeviceData(), dt, N_FFT);
			t += dt;
			std::swap(Sfc, Sfn);
		}
	}
	



	delete S;
	delete Sfc;
	delete Sfn;
	delete TmpA;
	delete TmpB;



	return 0;
}
