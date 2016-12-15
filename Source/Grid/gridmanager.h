#ifndef GRIDMANAGER_H
#define GRIDMANAGER_H

#include <QList>

#include "Base/gridmanagerbase.h"

class GridManager: public GridManagerBase
{
public:
	GridManager(int PastDomain, int FutureDomain, const Frame &F);
	~GridManager();

private:
	const Frame *m_Frame;
};

#endif // GRIDMANAGER_H
