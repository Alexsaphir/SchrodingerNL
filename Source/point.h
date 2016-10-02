#ifndef POINT_H
#define POINT_H

#include <QVector>

#include "type.h"

class Point
{
public:
	Point();
	int x() const;//V.at(0)
	int y() const;//V.at(1)
	int z() const;

	int getValue(int i) const;
	int Dim() const;
	void setValue(int i, int Coord);

private:
	QVector<int> V;
};

#endif // POINT_H
