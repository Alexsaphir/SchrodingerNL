#ifndef GRIDMANAGER_H
#define GRIDMANAGER_H

#include <QList>

#include "../Domain/domain1d.h"

class GridManager
{
public:
	GridManager(Axis const &X, cmplx Binf, cmplx Bsup, int i, int d);

	int getSizeStack() const;
	Domain1D* getDomain(int i) const;
	Domain1D* getCurrentDomain() const;
	Domain1D* getNextDomain() const;
	Domain1D* getOldDomain() const;

	void switchDomain();

	~GridManager();

private:
	QList<Domain1D*> Stack;
	int Size;
	int offset;
};

#endif // GRIDMANAGER_H
