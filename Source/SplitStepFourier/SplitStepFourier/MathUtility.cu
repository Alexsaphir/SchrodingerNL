#include "MathUtility.cuh"

namespace
{
	double __device__ __host__ lj(double j, double x, const std::vector<double> &Vx)
	{
		double l(1);
		for (int m = 0; m < Vx.size(); ++m)
		{
			if (m != j)
			{
				l *= (x - Vx.at(m)) / (Vx.at(j) - Vx.at(m));
			}
		}
		return l;
	}
}

void MathUtility::interpolationLagrange(double * x, cmplx * Res, int nbPtsX, double * xRL, cmplx * yL, cmplx * yR, int nbPtsLR)
{

	/*
	x		: array of x to interpolate
	Res		: array to save the value of the interpolation
	nbPtsX	: number of points to interpolate

	(xLR , yR yL)	: array of Points size = 2*nbPtsLR,2
	int nbPtsLR		: number of points for each yL and yR

	//the xL must be shift to the right
	xLR is sort so	: x(0)			<> yR(0)
					: x(nbPtsLR-1)	<> yR(nbPtsLR)
					: x(nbPtsLR)	<> yL(0)
					: x(2*nbPtsLR-1)<> yL(nbPtsLR-1)
	*/


	//Parse given value into vector
	std::vector<double> V_xRL;
	std::vector<cmplx> V_yRL;
	//fill vector
	for (int i = 0; i < 2 * nbPtsLR; ++i)
	{
		V_xRL.push_back(xRL[i]);
	}
	for (int i = 0; i < nbPtsLR; ++i)
	{
		V_yRL.push_back(yR[i]);
	}
	for (int i = 0; i < nbPtsLR; ++i)
	{
		V_yRL.push_back(yL[i]);
	}
	
	for (int i = 0; i < nbPtsX; ++i)
	{
		cmplx v = make_cuDoubleComplex(0, 0);
		for (int j = 0; j < nbPtsLR * 2; ++j)
		{
			v = v + V_yRL.at(j)*lj(j, x[i], V_xRL);
		}
		Res[i] = v;
	}
}


void __device__ MathUtility::BFSM1(cmplx * V, int N, double xmin, double xmax)
{
	

	int i = blockIdx.x *blockDim.x + threadIdx.x;


	if (i < N)
	{
		//shiffting up
		//h=max(0,-min(Y(1),Y(N))+1);
		double hr = std::max(0., -std::min(V[0].x, V[N - 1].x) + 1.);
		double hi = std::max(0., -std::min(V[0].y, V[N - 1].y) + 1.);
		cmplx h = make_cuDoubleComplex(hr, hi);
		cmplx fa = V[0];
		cmplx fb = V[N - 1];


		//Compute parameters for g: x-> alpha*x+beta,a linear function
		//Use complex to save the parameters for the real function and the imaginary function
		cmplx alpha = (fb - fa) / (xmax - xmin);
		cmplx beta = xmin*alpha + fa + h;
		double x = xmin + static_cast<double>(i)*(xmax - xmin) / (static_cast<double>(N);
		double x = xmin + static_cast<double>(i)*(xmax - xmin) / (static_cast<double>(N));
		V[i] = (V[i] + h) / (alpha*x + beta);
	}
}

void __device__ MathUtility::BFSM2(cmplx * V, int N, double xmin, double xmax)
{
	
}


