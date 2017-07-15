#include "NLSUtility.h"

#include <iostream>

void NLSUtility::GaussPulseLinear(Grid *S, double fc, double bw, double bwr, double tpr)
{
	Axis X = S->getAxis();

	double ref = std::pow(10., bwr / 20.);
	double a = -(M_PI*fc*bw)*(M_PI*fc*bw) / 4. / std::log(ref);

	for (int i = 0; i < X.getN(); ++i)
	{
		double t = X.getValueAt(i);
		double yenv = std::exp(-a*t*t);
		S->getHostData()[i] = yenv*make_cuDoubleComplex(std::cos(2.*M_PI*fc*t), std::sin(2.*M_PI*fc*t));
	}
	S->syncHostToDevice();
}

double NLSUtility::computeTotalMass(Grid * S)
{
	Axis X = S->getAxis();
	double dx = (X.getXmax() - X.getXmin()) / static_cast<double>(X.getN());
	
	double R(0);
	
	for (int i = 0; i < X.getN(); ++i)
	{
		R += cuCabs(S->getHostData()[i]) * cuCabs(S->getHostData()[i]) * dx;
	}
	return R;
}
