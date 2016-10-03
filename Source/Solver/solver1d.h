#ifndef SOLVER1D_H
#define SOLVER1D_H

#include <QDebug>

#include "../Grid/gridmanager.h"
#include "../SparseMatrix/sparsematrix.h"


class Solver1D
{
public:
	Solver1D(const Axis &X, cmplx Binf, cmplx Bsup, Type timeStep);
	~Solver1D();



	Type getDt() const;
	Type getDx() const;
	cmplx getNextValue(int i) const;
	cmplx getOldValue(int i) const;
	Type getPos(int i) const;
	double getTime() const;
	cmplx getValue(int i) const;
	double getValueNorm(int i) const;
	Type getXmax() const;
	Type getXmin() const;
	int getN() const;
	int getT() const;
	void doStep();
	void doStepWithMatrix();
	void setValue(int i, cmplx y);


	void initPulse();
	void doFourrier();
	void undoFourrier();

private:
	Domain1D* getCurrentDomain() const;
	Domain1D* getNextDomain() const;
	Domain1D* getOldDomain() const;

	void switchDomain();

	void SolveImaginary();
	void SolveReal();
	Type V(int i) const;
	Type err_integ;

private:
	GridManager *GridM;
	SparseMatrix *Matrix;

	Type dt;//Temporal step
	int T;//Current time step
};

#endif // SOLVER1D_H
