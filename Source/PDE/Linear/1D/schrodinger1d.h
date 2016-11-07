#ifndef SCHRODINGER1D_H
#define SCHRODINGER1D_H

#include "pdelinear1dvirtual.h"

class Schrodinger1D: public PDELinear1DVirtual
{
public:
	Schrodinger1D();
	Schrodinger1D(const Axis *F, int Past, int Future, Type timeStep);

	virtual void computeNextStep();
	virtual void InitialState();
	
	virtual void initializeLinearSolver();

	virtual cmplx at(const Point &P) const;

	~Schrodinger1D();
	
private:
	Type dt;
	Type dx;
	cmplx alpha;
};

#endif // SCHRODINGER1D_H
