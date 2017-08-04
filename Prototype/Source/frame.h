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
	~Frame();

	const Axis *getAxis(int i) const;
	const Axis *at(int i) const;
	int size() const;
	bool empty() const;

private:
	QVector<const Axis*> m_Basis;
	int m_N;//Size of m_Basis
};

#endif // FRAME_H


//the Basis is see as a colletioc
// at(i) don't check if index is valid
