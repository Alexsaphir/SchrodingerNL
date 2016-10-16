#ifndef HEAT1D_H
#define HEAT1D_H

#include "pdelinear1dvirtual.h"

#include "../../../Grid/grid1d.h"
#include "../../../Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h"

class Heat1D: PDELinear1DVirtual
{
public:
	Heat1D();
	Heat1D(const Axis &X, Type t, cmplx Binf, cmplx Bsup);


	virtual void computeNextStep() =0;
	virtual void InitialState() = 0;

	virtual ~Heat1D();
private:
	Grid1D *Grid1;
	Grid1D *Grid2;

	bool Grid1IsCurrent;

	cmplx BoundInf;
	cmplx BoundSup;

	Type dt,dx;
	LinearSolver *LS;
	ColumnDataProxy *C1,*C2;
};

#endif // HEAT1D_H
