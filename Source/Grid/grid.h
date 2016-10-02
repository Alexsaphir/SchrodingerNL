#ifndef GRID_H
#define GRID_H

#include <QVector>

#include "../axis.h"
#include "../point.h"
#include "../type.h"


class Grid
{
public:
	Grid();

	void AddAxis(const Axis &X);
	void initGrid();

	Type getStep(int AxisN) const;

	int getN() const;

	cmplx getValue(const Point &Pos) const;
	void setValue(const Point &Pos, cmplx value);

private:
	int getIndexFromPos(const Point &P) const;

private:
	QVector<Axis*> Repere;
	QVector<cmplx> V;
	bool isInit;
};

#endif // GRID_H
