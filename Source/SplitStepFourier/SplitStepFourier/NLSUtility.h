#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include "Grid.cuh"
#include "Grid2D.cuh"



namespace NLSUtility
{
	void GaussPulseLinear(Grid *S, double fc, double bw, double bwr, double tpr);
	void GaussMonoPulse2D(Grid2D *S);

	double computeTotalMass(Grid *S);
}