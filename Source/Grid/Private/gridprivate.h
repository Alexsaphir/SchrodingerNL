#ifndef GRIDPRIVATE_H
#define GRIDPRIVATE_H

#include <QVector>

#include "../../Axis/axis.h"
#include "../../frame.h"
#include "../../point.h"
#include "../../type.h"

class GridPrivate
{
public:
	GridPrivate();
	GridPrivate(const Frame *F);
	GridPrivate(const GridPrivate &GP);
	~GridPrivate();

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
	const Frame* getFrame() const;

protected:
	int m_N;//Precomputed size of m_V
	QVector<cmplx> m_V;
	const Frame *m_Repere;
	const Point *m_Dimension;

};

#endif // GRIDPRIVATE_H

//GridPrivate doesn't manage the allocation of Repere
//All index pass to a method of GridPrivate is supposed to be already check
