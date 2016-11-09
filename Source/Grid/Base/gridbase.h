#ifndef GRIDBASE_H
#define GRIDBASE_H

#include <QVector>

#include "../../Axis/axis.h"
#include "../../frame.h"
#include "../../point.h"
#include "../../type.h"

class GridBase
{
public:
	GridBase();
	GridBase(const Frame *F);
	GridBase(const GridBase &GP);
	~GridBase();

	//Frame getters
	int getNumberOfAxis() const;
	int getSizeOfAxis(int AxisN) const;
	Type getStepOfAxis(int AxisN) const;

	int getIndexFromPos(const Point &Pos) const;



	//Grid
	cmplx getValue(const Point &Pos) const;
	cmplx getValue(int i) const;
	int getSizeOfGrid() const;

	void setValue(const Point &Pos, cmplx value);
	void setValue(int i, cmplx value);

	bool isInGrid(const Point &Pos) const;
	bool isInGrid(int i) const;

protected:
	const Axis* getAxis(int i) const;

protected:
	int m_N;//Precomputed size of m_V
	QVector<cmplx> m_V;
	const Frame *m_Frame;
	const Point *m_Dimension;

};

#endif // GRIDBASE_H

//GridBase doesn't manage the allocation of Repere
//All index pass to a method of GridBase is supposed to be already check
