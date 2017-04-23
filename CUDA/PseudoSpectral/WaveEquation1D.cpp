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
	Sfreq->fillDevice(Sder->getDeviceData());
	Sder->firstDerivative();
	Sder->syncDeviceToHost();
	Sfreq->syncDeviceToHost();
	for (int i = 0; i < S->getSignalPoints(); ++i)
	{
		Sder->getHostData()[i] = Sfreq->getHostData()[i]+m_dt*Sder->getHostData()[i];
	}
	Sder->smoothFilterCesaro();
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
