#ifndef GRID1D_H
#define GRID1D_H

#include <QVector>

#include<complex>

#include <QDebug>

typedef long double Type;
typedef std::complex<Type> cmplx;

class Grid1D
{
public:

	Grid1D(double Xmn, double Xmx, double Xsp);

	double getDx() const;
	double getPos(int i) const;
	cmplx getValue(int i) const;
	double getXmax() const;
	double getXmin() const;
	int getN() const;
	void setValue(int i, cmplx y);
	void setValueReal(int i, Type y);
	void setValueImag(int i, Type y);

private:
	QVector<cmplx> V;
	double Xmin;
	double Xmax;
	double Xstep;
	int nbPts;
};

#endif // GRID1D_H
