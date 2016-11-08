#include "gridprivate.h"

GridPrivate::GridPrivate()
{//Default Constructor
	m_Repere = NULL;
	m_V.squeeze();//Delete all pre allocation of memory to have an empty QVector
	m_Dimension = NULL;
}

GridPrivate::GridPrivate(const Frame *F): m_Repere(F)
{
	QVector<int> tmpSize;
	m_N = 0;
	//Compute the size of the Grid
	for(int i=0; i<m_Repere->size(); ++i)
	{
		tmpSize.push_back(m_Repere->at(i)->getAxisN());
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

}

GridPrivate::GridPrivate(const GridPrivate &GP): m_N(GP.m_N), m_V(GP.m_V), m_Repere(GP.m_Repere), m_Dimension(GP.m_Dimension)
{//Copy Constructor
}

int GridPrivate::getNumberOfAxis() const
{
	return m_Repere->size();
}

int GridPrivate::getSizeOfAxis(int AxisN) const
{
	if(AxisN < 0)
		return 0;
	if(AxisN >= m_Repere->size())
		return 0;
	return m_Repere->at(AxisN)->getAxisN();
}

Type GridPrivate::getStepOfAxis(int AxisN) const
{
	if(AxisN < 0)
		return 0;
	if(AxisN >= m_Repere->size())
		return 0;
	return m_Repere->at(AxisN)->getAxisStep();
}

const Axis* GridPrivate::getAxis(int i) const
{
	return m_Repere->at(i);
}

const Frame* GridPrivate::getFrame() const
{
	return m_Repere;
}

bool GridPrivate::isInGrid(const Point &Pos) const
{
	if(!m_Dimension)
		return false;//m_Dimension == NULL
	if(m_Repere->size() != Pos.Dim())
		return false;//m_Repere->size() is a precompute value
	for(int i=0; i<m_Repere->size(); ++i)
	{
		if(m_Dimension->at(i) != Pos.at(i))
			return false;
	}
	return true;
}

bool GridPrivate::isInGrid(int i) const
{
	if((m_N >= i) || (i<0))
		return false;
	return true;
}

int GridPrivate::getIndexFromPos(const Point &Pos) const
{
	if(m_Repere->empty())
		return -1;
	if(Pos.Dim()!=m_Repere->size())
		return -1;

	int index(Pos.getValue(0));
	if(m_Repere->size()==1)
		return index;
	//We now that Repere.size()>=2
	index*=m_Repere->at(1)->getAxisN();
	index+=Pos.getValue(1);

	int tmp_size(1);
	tmp_size*=m_Repere->at(0)->getAxisN();
	tmp_size*=m_Repere->at(1)->getAxisN();

	for(int i=2; i<m_Repere->size(); ++i)
	{
		if (Pos.getValue(i)<0 || Pos.getValue(i)>=m_Repere->at(i)->getAxisN())
			return -1;
		index+=Pos.getValue(i)*tmp_size;
		tmp_size*=m_Repere->at(i)->getAxisN();
	}
	return index;
}

cmplx GridPrivate::getValue(const Point &Pos) const
{
	return m_V.at(this->getIndexFromPos(Pos));
}

cmplx GridPrivate::getValue(int i) const
{
	return m_V.at(i);
}

int GridPrivate::getSizeOfGrid() const
{
	return m_N;
}

void GridPrivate::setValue(const Point &Pos, cmplx value)
{
	m_V.replace(this->getIndexFromPos(Pos), value);
}

void GridPrivate::setValue(int i, cmplx value)
{
	m_V.replace(i, value);
}

GridPrivate::~GridPrivate()
{
	//m_Repere is not delete because it's not the job of this class to manage this object
	delete m_Dimension;
}
