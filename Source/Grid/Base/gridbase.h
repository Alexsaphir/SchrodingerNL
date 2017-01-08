#ifndef GRIDBASE_H
#define GRIDBASE_H

#include <QVector>

#include "../../Axis/axis.h"
#include "../../frame.h"
#include "../../point.h"

#include "../../Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h"
#include "../../Matrix/Matrix/RowMatrix/DataProxy/rowdataproxy.h"

class RowDataProxy;
class ColumnDataProxy;

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
	Point getDimension() const;

	int getIndexFromPos(const Point &Pos) const;



	//Grid
	cmplx getValue(const Point &Pos) const;
	cmplx getValue(int i) const;
	int getSizeOfGrid() const;

	void setValue(const Point &Pos, cmplx value);
	void setValue(int i, cmplx value);

	bool isInGrid(const Point &Pos) const;
	bool isInGrid(int i) const;

	void reset();//Set all value of the grid to cmplx(0,0)

    ColumnDataProxy* getColumn() const;
    RowDataProxy* getRow() const;


	const Axis* getAxis(int i) const;

	void operator+=(const GridBase &G);
	void operator-=(const GridBase &G);

protected:
	int m_N;//Precomputed size of m_V
	QVector<cmplx> m_V;
	const Frame *m_Frame;
	const Point *m_Dimension;
    ColumnDataProxy *m_ProxyColumn;
    RowDataProxy *m_ProxyRow;
};

#endif // GRIDBASE_H

//GridBase doesn't manage the allocation of m_Frame
//All index pass to a method of GridBase is supposed to be already check
