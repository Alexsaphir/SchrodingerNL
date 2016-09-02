#ifndef SOLVER1D_H
#define SOLVER1D_H

#include <QDebug>

#include "gridmanager.h"

class Solver1D
{
public:
	Solver1D(double Xmin, double Xmax, double Xstep, cmplx Binf, cmplx Bsup, double timeStep);
	~Solver1D();



	double getDt() const;
	double getDx() const;
	cmplx getNextValue(int i) const;
	cmplx getOldValue(int i) const;
	double getPos(int i) const;
	double getTime() const;
	cmplx getValue(int i) const;
	double getValueNorm(int i) const;
	double getXmax() const;
	double getXmin() const;
	int getN() const;
	int getT() const;
	void doStep();
	void setValue(int i, cmplx y);


	void initPulse();

private:
	Domain1D* getCurrentDomain() const;
	Domain1D* getNextDomain() const;
	Domain1D* getOldDomain() const;

	void switchDomain();

	void SolveImaginary();
	void SolveReal();
	double V(int i) const;

private:
	bool Dom1isCurrent;
	bool Dom2isCurrent;
	bool Dom3isCurrent;

	Domain1D *Dom1;
	Domain1D *Dom2;
	Domain1D *Dom3;

	double dt;//Temporal step
	int T;//Current time step
};

#endif // SOLVER1D_H
