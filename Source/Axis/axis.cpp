#include "axis.h"

Axis::Axis(): m_Xmin(0), m_Xmax(0), m_nbPts(1)
{
}

Axis::Axis(Type Xmin, Type Xmax, int nbPts): m_Xmin(Xmin), m_Xmax(Xmax), m_nbPts(nbPts)
{
}

Axis::Axis(const Axis &A): m_Xmin(A.m_Xmin), m_Xmax(A.m_Xmax), m_nbPts(A.m_nbPts)
{
}

Type Axis::getAxisMax() const
{
	return m_Xmax;
}

Type Axis::getAxisMin() const
{
	return m_Xmin;
}

int Axis::getAxisN() const
{
	return m_nbPts;
}

Type Axis::getAxisStep() const
{
	return 0.;
}

Axis::~Axis()
{

}
