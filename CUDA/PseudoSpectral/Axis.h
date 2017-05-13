#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

class Axis
{
public:
	Axis(double Xmin, double Xmax, int nbPts);

	double getLinearValueAt(int i) const;
	double getFFTValueAt(int i) const;
	double getChebyshevValueAt(int i) const;
	double getFrequency(int i) const;

	~Axis();
private:
	double m_Xmin;
	double m_Xmax;
	int m_nbPts;
};

