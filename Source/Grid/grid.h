#ifndef GRID_H
#define GRID_H

#include <QVector>

#include "../Axis/axis.h"
#include "../frame.h"
#include "../point.h"
#include "../type.h"


class Grid
{
public:
	Grid();
	Grid(const Frame &F);
	Grid(const Grid &G);

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

	~Grid();
private:
	Axis *getAxis(int i) const;



private:
	Frame *Repere;
	QVector<cmplx> V;
	int N;//Precomputed size of V
};

#endif // GRID_H
