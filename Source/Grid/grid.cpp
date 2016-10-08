#include "grid.h"

Grid::Grid()
{
	isInit = false;
	N = 0;
}

Grid::Grid(const Grid &G)
{
	//The user is not allowed to modify an Axis from Repere so We can use the same pointers
	int i(0);
	Axis *tmp;
	tmp=G.getAxis(i);
	while(tmp!=NULL)
	{
		Repere.append(tmp);
		++i;
		tmp=G.getAxis(i);
	}
	if(!G.ready())
	{
		isInit = false;

	}
	else
	{
		initGrid();
		for(int i=0; i<G.getN(); ++i)
			setValue(i,G.getValue(i));
	}
}

void Grid::AddAxis(const Axis &X)
{
	if(isInit)
		return;
	Axis *T;
	T = new Axis(X);
	Repere.append(T);
}

Axis* Grid::getAxis(int i) const
{
	if(i<0 || i>=Repere.size())
		return NULL;
	else
		return Repere.at(i);
}

int Grid::getAxisN() const
{
	if (!isInit)
		return 0;
	return Repere.size();
}

int Grid::getIndexFromPos(const Point &Pos) const
{
	if(!isInit)
		return -1;
	if(Repere.empty())
		return -1;
	if(Pos.Dim()!=Repere.size())
		return -1;
	if(Repere.size()==1)
		return Pos.getValue(0);
	//We now that Repere.size()>=2

	int index(Pos.getValue(0));
	index*=Repere.at(1)->getAxisN();
	index+=Pos.getValue(1);

	int tmp_size(1);
	tmp_size*=Repere.at(0)->getAxisN();
	tmp_size*=Repere.at(1)->getAxisN();

	for(int i=2; i<Repere.size(); ++i)
	{
		if (Pos.getValue(i)<0 || Pos.getValue(i)>=Repere.at(i)->getAxisN())
			return -1;
		index+=Pos.getValue(i)*tmp_size;
		tmp_size*=Repere.at(i)->getAxisN();
	}
	return index;
}

int Grid::getN() const
{
	if(!isInit)
		return 0;
	return N;
}

int Grid::getN(int AxisN) const
{
	if(!isInit)
		return 0;
	if(AxisN<0 || AxisN >= Repere.size())
		return 0;
	return Repere.at(AxisN)->getAxisN();
}

Type Grid::getStep(int AxisN) const
{
	if(!isInit)
		return 0;
	if(AxisN <0 || AxisN >= Repere.size())
		return 0.;
	return Repere.at(AxisN)->getAxisStep();
}

cmplx Grid::getValue(const Point &Pos) const
{
	if(!isInit)
		return cmplx(0,0);
	//It's the Domain class who catch the index error
	return V.at(getIndexFromPos(Pos));
}

cmplx Grid::getValue(int i) const
{
	if(!isInit)
		return cmplx(0,0);
	//It's the Domain class who catch the index error
	return V.at(i);
}


void Grid::initGrid()
{
	if(isInit)
		return;//Cancel multiple initialisation
	isInit = true;
	N = 0;
	//Compute the size of the Grid
	for(int i=0; i<Repere.size(); ++i)
	{
		if (N == 0)
		{
			N+= Repere.at(i)->getAxisN();
		}
		else
		{
			N*= Repere.at(i)->getAxisN();
		}
	}

	V.fill(cmplx(0,0), N);
}

bool Grid::isInGrid(const Point &Pos) const
{
	if(!isInit)
		return false;
	//We can check if each indice of the position are good for each Axis
	//Or we can compute the indice of Pos in the grid and do a simply test

	return isInGrid(getIndexFromPos(Pos));

}

bool Grid::isInGrid(int i) const
{
	if(!isInit)
		return false;
	if(i<0 || i>=N)
		return false;
	return true;
}

bool Grid::ready() const
{
	return isInit;
}

void Grid::setValue(const Point &Pos, cmplx value)
{
	if(!isInit)
		return;
	//It's the Domain class who catch the index error
	setValue(getIndexFromPos(Pos), value);

}

void Grid::setValue(int i, cmplx value)
{
	if(!isInit)
		return;
	//It's the Domain class who catch the index error
	V.replace(i, value);
}
