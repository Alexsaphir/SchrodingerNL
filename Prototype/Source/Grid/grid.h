#ifndef GRID_H
#define GRID_H

#include "Base/gridbase.h"


class Grid: public GridBase
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
