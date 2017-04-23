#pragma once

#include "Signal.cuh"
#include "SignalFFT.cuh"

class WaveEquation1D
{
public:
	WaveEquation1D(int nbPts, double dt);
	
	void init();

	void computeStep();

	~WaveEquation1D();

public:
	Axis *X;
	Signal *S, *Sfreq;
	SignalFFT *Sder;
	double m_dt;

};

