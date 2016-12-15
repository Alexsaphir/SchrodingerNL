#ifndef PDEVIRTUAL_H
#define PDEVIRTUAL_H


#include "../frame.h"
#include "../point.h"
#include "../type.h"
#include "../Grid/Base/gridmanagerbase.h"

class PDEVirtual
{
public:
	PDEVirtual();
	PDEVirtual(const Frame &F);
	PDEVirtual(const Frame &F, int Past, int Future);

	virtual void computeNextStep() =0;
	virtual void InitialState() = 0;

	virtual cmplx at(const Point &P) const;

	virtual ~PDEVirtual();

public:
	Frame *m_Frame;
	GridManagerBase *m_Space;

};

#endif // PDEVIRTUAL_H
