#include "point.h"

Point::Point()
{
	m_V.append(0);
}

Point::Point(QVector<int> V): m_V(V)
{
}

Point::Point(const Point &P): m_V(P.m_V)
{//Copy constructor
}

Point::Point(int x)
{
	m_V << x;
}

Point::Point(int x, int y)
{
	m_V << x << y;
}

Point::Point(int x, int y, int z)
{
	m_V << x << y << z;
}

int Point::Dim() const
{
	return m_V.size();
}

int Point::at(int i) const
{
	return  m_V.at(i);
}

int Point::getValue(int i) const
{
	if (i<0 || i>=m_V.size())
		return 0.;
	return m_V.at(i);
}

void Point::setValue(int i, int Coord)
{
	if(i<0)
		return;
	while(m_V.size()<i+1)//Test if there are enought item in the QVector
		m_V.append(0);
	m_V.replace(i, Coord);
}

int Point::x() const
{
	return getValue(0);
}

int Point::y() const
{
	return getValue(1);
}

int Point::z() const
{
	return getValue(2);
}
