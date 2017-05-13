#include "WaveEquation1D.h"


WaveEquation1D::WaveEquation1D(int nbPts, double dt)
{
	X = new Axis(0, 2.*M_PI, nbPts);
	S = new Signal(0, 2.*M_PI, nbPts);
	Sfreq = new Signal(0, 2.*M_PI, nbPts);
	Sder = new SignalFFT(nbPts);
	m_dt = dt;
}

void WaveEquation1D::init()
{
	for (int i = 0; i < S->getSignalPoints(); ++i)
	{
		S->getHostData()[i] = make_cuDoubleComplex(std::sin(M_PI*std::cos(X->getLinearValueAt(i))),0);
	}
	S->syncHostToDevice();
}

void WaveEquation1D::computeStep()
{
	Sder->computeFFT(S);
	Sder->syncDeviceToHost();

	int nbPts = Sder->getSignalPoints();

	for (int i = 0; i < nbPts; ++i)
	{
		double k = static_cast<double>(i);
		cmplx Coeff;


		//Positive Frequency are between 0 and <N/2
		//Negative Frequency are between N/2and <N
		if (i < nbPts / 2)
		{
			k = k *2.*M_PI*k / static_cast<double>(nbPts);
			Coeff = 1. + iMul(m_dt*k) - .5*m_dt*m_dt*k*k;
			
		}
		else if (i > nbPts / 2)
		{
			k = (k - static_cast<double>(nbPts))*2.*M_PI*k / static_cast<double>(nbPts);
			Coeff = 1. + iMul(m_dt*k) - .5*m_dt*m_dt*k*k;
			
		}
		else
		{
			k = 0;
			
			Coeff = make_cuDoubleComplex(1.,0);//k=N/2
		}
		
		Sder->getHostData()[i] = cuCmul(Sder->getHostData()[i], Coeff);
	}
	Sder->smoothFilterRaisedCosinus();
	Sder->syncHostToDevice();
	Sder->ComputeSignal(S);
}



WaveEquation1D::~WaveEquation1D()
{
	delete X;
	delete S;
	delete Sfreq;
	delete Sder;
}
