#ifndef GRIDMANAGER_H
#define GRIDMANAGER_H

#include <QList>

#include "Base/gridmanagerbase.h"

class GridManager: public GridManagerBase
{
public:
	GridManager(int PastDomain, int FutureDomain, int TmpDomain, const Frame &F);
	~GridManager();

	void swapStackTmp(const GridBase *S, const GridBase *T);//Swap an element from the stack with a element in the temporary stack

	GridBase* getTemporaryDomain(int i) const;

private:
	const Frame *m_Frame;
	int m_tmpStackSize;
	QVector<GridBase*> m_tmpStack;
};

#endif // GRIDMANAGER_H
