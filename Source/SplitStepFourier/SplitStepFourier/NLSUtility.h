#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include "Grid.cuh"



namespace NLSUtility
{
	void GaussPulseLinear(Grid *S, double fc, double bw, double bwr, double tpr);
	double computeTotalMass(Grid *S);
}