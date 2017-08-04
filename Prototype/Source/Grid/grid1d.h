#ifndef GRID1D_H
#define GRID1D_H

#include <QVector>


#include "../type.h"
#include "../Axis/axis.h"


class Grid1D
{
public:

	Grid1D(const Axis *X);

	Type getDx() const;
	Type getPos(int i) const;
	cmplx getValue(int i) const;
	Type getXmax() const;
	Type getXmin() const;
	int getN() const;
	void setValue(int i, cmplx value);

	~Grid1D();


private:
	QVector<cmplx> V;
	Axis *AxisX;
};

#endif // GRID1D_H
