#include "include/point.h"

Point::Point()
{
	V.append(0);
}

int Point::Dim() const
{
	return V.size();
}

Type Point::getValue(int i) const
{
	if (i<0 || i>=V.size())
		return 0.;
	return V.at(i);
}

void Point::setValue(int i, Type Coord)
{
	if(i<0)
		return;
	while(V.size()<i+1)//Test if there are enought item in the QVector
		V.append(0);
	V.replace(i, Coord);
}

Type Point::x() const
{
	return getValue(0);
}

Type Point::y() const
{
	return getValue(1);
}

Type Point::z() const
{
	return getValue(2);
}
