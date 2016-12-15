#include "gridbase.h"

GridBase::GridBase()
{//Default Constructor
	m_Frame = NULL;
	m_V.squeeze();//Delete all pre allocation of memory to have an empty QVector
	m_Dimension = NULL;
    m_ProxyColumn = NULL;
    m_ProxyRow = NULL;
}

GridBase::GridBase(const Frame *F): m_Frame(F)
{
	QVector<int> tmpSize;
	m_N = 0;
	//Compute the size of the Grid
	for(int i=0; i<m_Frame->size(); ++i)
	{
		tmpSize.push_back(m_Frame->at(i)->getAxisN());
		if (m_N == 0)
		{

			m_N+= tmpSize.last();
		}
		else
		{
			m_N*= tmpSize.last();
		}
	}
	//Fill the Grid with the good number of item
	m_V.fill(cmplx(0,0), m_N);
	m_Dimension = new Point(tmpSize);

    m_ProxyColumn = new ColumnDataProxy(this);
    m_ProxyRow = new RowDataProxy(this);
}

GridBase::GridBase(const GridBase &GP): m_N(GP.m_N), m_V(GP.m_V), m_Frame(GP.m_Frame), m_Dimension(new Point(*GP.m_Dimension))
{//Copy Constructor
}

int GridBase::getNumberOfAxis() const
{
	return m_Frame->size();
}

int GridBase::getSizeOfAxis(int AxisN) const
{
	if(AxisN < 0)
		return 0;
	if(AxisN >= m_Frame->size())
		return 0;
	return m_Frame->at(AxisN)->getAxisN();
}

Type GridBase::getStepOfAxis(int AxisN) const
{
	if(AxisN < 0)
		return 0;
	if(AxisN >= m_Frame->size())
		return 0;
	return m_Frame->at(AxisN)->getAxisStep();
}

const Axis* GridBase::getAxis(int i) const
{
	return m_Frame->at(i);
}

bool GridBase::isInGrid(const Point &Pos) const
{
	if(!m_Dimension)
		return false;//m_Dimension == NULL
	if(m_Frame->size() != Pos.Dim())
		return false;//m_Repere->size() is a precompute value
	for(int i=0; i<m_Frame->size(); ++i)
	{
        if(m_Dimension->at(i) < Pos.at(i) || Pos.at(i) < 0)
			return false;
	}
	return true;
}

Point GridBase::getDimension() const
{
	return Point(*m_Dimension);
}

bool GridBase::isInGrid(int i) const
{
	if((m_N >= i) || (i<0))
		return false;
	return true;
}

int GridBase::getIndexFromPos(const Point &Pos) const
{
	if(m_Frame->empty())
		return -1;
	if(Pos.Dim()!=m_Frame->size())
		return -1;

	int index(Pos.getValue(0));
	if(m_Frame->size()==1)
		return index;
	//We know that Repere.size()>=2
	index*=m_Frame->at(1)->getAxisN();
	index+=Pos.getValue(1);

	int tmp_size(1);
	tmp_size*=m_Frame->at(0)->getAxisN();
	tmp_size*=m_Frame->at(1)->getAxisN();

	for(int i=2; i<m_Frame->size(); ++i)
	{
		if (Pos.getValue(i)<0 || Pos.getValue(i)>=m_Frame->at(i)->getAxisN())
			return -1;
		index+=Pos.getValue(i)*tmp_size;
		tmp_size*=m_Frame->at(i)->getAxisN();
	}
	return index;
}

cmplx GridBase::getValue(const Point &Pos) const
{
	return m_V.at(this->getIndexFromPos(Pos));
}

cmplx GridBase::getValue(int i) const
{
	return m_V.at(i);
}

int GridBase::getSizeOfGrid() const
{
	return m_N;
}

void GridBase::setValue(const Point &Pos, cmplx value)
{
	m_V.replace(this->getIndexFromPos(Pos), value);
}

void GridBase::setValue(int i, cmplx value)
{
	m_V.replace(i, value);
}

ColumnDataProxy* GridBase::getColumn() const
{
    return m_ProxyColumn;
}

RowDataProxy* GridBase::getRow() const
{
    return m_ProxyRow;
}

GridBase::~GridBase()
{
	//m_Frame is not delete because it's not the job of this class to manage this object
	delete m_Dimension;
    delete m_ProxyColumn;
    delete m_ProxyRow;
}
