#ifndef POINT_H
#define POINT_H

#include <QVector>

#include "include/type.h"

class Point
{
public:
	Point();
	Type x() const;//V.at(0)
	Type y() const;//V.at(1)
	Type z() const;

	Type getValue(int i) const;
	int Dim() const;
	void setValue(int i, Type Coord);

private:
	QVector<Type> V;
};

#endif // POINT_H
