#ifndef FRAME_H
#define FRAME_H

#include <QVector>

#include "axis.h"
#include "type.h"

class Frame
{
public:
	Frame();

private:
	QVector<Axis> Basis;

};

#endif // FRAME_H
