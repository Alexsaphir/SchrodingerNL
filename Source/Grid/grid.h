#ifndef GRID_H
#define GRID_H

#include "Private/gridprivate.h"


class Grid: public GridPrivate
{
public:
	Grid();
	Grid(const Frame &F);
	Grid(const Grid &G);

	~Grid();
private:
	const Frame *m_Frame;
};

#endif // GRID_H
