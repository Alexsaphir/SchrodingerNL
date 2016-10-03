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
	int getAxisN() const;
	int getIndexFromPos(const Point &Pos) const;
	int getN(int AxisN) const;
	int getN() const;
	Type getStep(int AxisN) const;
	cmplx getValue(const Point &Pos) const;
	cmplx getValue(int i) const;
	void initGrid();
	bool isInGrid(const Point &Pos) const;
	bool isInGrid(int i) const;
	void setValue(const Point &Pos, cmplx value);
	void setValue(int i, cmplx value);





private:
	QVector<Axis*> Repere;
	QVector<cmplx> V;
	bool isInit;
	int N;//Precomputed size of V
};

#endif // GRID_H
