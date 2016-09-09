#ifndef GRID1D_H
#define GRID1D_H

#include <QVector>

#include<complex>

#include <QDebug>

typedef double Type;
typedef std::complex<Type> cmplx;

class Grid1D
{
public:

	Grid1D(Type Xmn, Type Xmx, Type Xsp);

	Type getDx() const;
	Type getPos(int i) const;
	cmplx getValue(int i) const;
	Type getXmax() const;
	Type getXmin() const;
	int getN() const;
	void setValue(int i, cmplx y);
	void setValueReal(int i, Type y);
	void setValueImag(int i, Type y);



private:
	QVector<cmplx> V;
	Type Xmin;
	Type Xmax;
	Type Xstep;
	int nbPts;
};

#endif // GRID1D_H
