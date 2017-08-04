#ifndef GRIDMANAGERBASE_H
#define GRIDMANAGERBASE_H

#include <QList>

#include "gridbase.h"

class GridManagerBase
{
public:
	GridManagerBase(int PastDomain, int FutureDomain, const Frame *F);
	~GridManagerBase();

	int getSizeStack() const;

	GridBase* getGrid(int i) const;
	GridBase* getCurrentGrid() const;
	GridBase* getNextGrid() const;
	GridBase* getOldGrid() const;

	cmplx getValue(const Point &P, int t) const;
	cmplx getValue(int i, int t) const;

	void switchGrid();

	ColumnDataProxy* getColumnAtTime(int t) const;
	RowDataProxy* getRowAtTime(int t) const;

protected:
	const Frame *m_Frame;
	int m_Size;//Number of DomainBase
	int m_offset;//Indice of the Current Domain

	QList<GridBase*> m_Stack;
};

#endif // GRIDMANAGERBASE_H
