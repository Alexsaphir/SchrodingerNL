#include "grid.h"

Grid::Grid()
{
	isInit = false;
}

void Grid::AddAxis(const Axis &X)
{
	if(isInit)
		return;
	Axis *T;
	T = new Axis(X);
	Repere.append(T);
}

Type Grid::getStep(int AxisN) const
{
	if(AxisN >= Repere.size())
		return Repere.at(AxisN)->getAxisStep();
	else
		return 0.;
}

int Grid::getIndexFromPos(const Point &Pos) const
{
	if(Repere.empty())
		return -1;
	if(Pos.Dim()!=Repere.size())
		return -1;
	int index(Pos.getValue(0));
	for(int i=1; i<Repere.size(); ++i)
	{
		index+=Pos.getValue(i)*Repere.at(i-1)->getAxisN();
	}
	return index;
}

