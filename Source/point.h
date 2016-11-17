#ifndef POINT_H
#define POINT_H

#include <QVector>

#include "type.h"

class Point
{
public:
	Point();
	Point(QVector<int> V);
	Point(const Point &P);
	Point(int x);
	Point(int x, int y);
	Point(int x, int y, int z);

	int x() const;//V.at(0)
	int y() const;//V.at(1)
	int z() const;

	int at(int i) const;
	int getValue(int i) const;
	int Dim() const;
	void setValue(int i, int Coord);
	void remove(int Coord);

private:
	QVector<int> m_V;
};

#endif // POINT_H

//The class Point describe a position on the grid.
//The grid can be see as a multidimensional array.
//But negative position permit to access to boundary conditions
