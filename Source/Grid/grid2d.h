#ifndef GRID2D_H
#define GRID2D_H

#include <QVector>

#include "../type.h"
#include "../axis.h"

class Grid2D
{
public:
	Grid2D(const Axis &X, const Axis &Y);

	Type getDx() const;
	Type getDy() const;
	Type getPosX(int i, int j) const;
	Type getPosY(int i, int j) const;
	cmplx getValue(int i,int j) const;
	Type getXmax() const;
	Type getXmin() const;
	Type getYmax() const;
	Type getYmin() const;
	int getNx() const;
	int getNy() const;
	void setValue(int i, int j, cmplx value);

	bool isInGrid(int i, int j) const;

private:
	QVector<cmplx> V;

	Axis AxisX;
	Axis AxisY;
};























#endif // GRID2D_H
