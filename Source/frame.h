#ifndef FRAME_H
#define FRAME_H

#include <QVector>

#include "Axis/axis.h"
#include "type.h"

class Frame
{
public:
	Frame();
	Frame(QVector<Axis *> axes);
	Frame(const Axis *X);
	Frame(const Axis *X, const Axis *Y);
	Frame(const Frame &F);
	Axis* getAxis(int i) const;
	Axis* at(int i) const;
	int size() const;
	bool empty() const;

	~Frame();

private:
	QVector<Axis*> Basis;
	int N;
};

#endif // FRAME_H
