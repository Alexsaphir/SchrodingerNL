#include <QDebug>

#include "Axis/linearaxis.h"
#include "frame.h"
#include "Grid/Base/gridbase.h"
#include "Grid/grid.h"
#include "Domain/Base/domainbase.h"
#include "Domain/domain.h"
#include "Domain/Base/domainmanagerbase.h"
#include "Domain/domainmanager.h"

int main(int argc, char **argv)
{
	LinearAxis *X(NULL);
	Axis *Y(NULL);
	X = new LinearAxis(-1., 1., 1);
	Y = new LinearAxis(*X);

	Frame *F;
	F = new Frame(X, Y);

	GridBase *GB1;
	GridBase *GB2;
	GB1 = new GridBase(F);
	GB2 = new GridBase(*GB1);

	Grid *G1;
	Grid *G2;
	G1 = new Grid(*F);
	G2 = new Grid(*G1);

	DomainBase *DB1;
	DomainBase *DB2;
	DB1 = new DomainBase(F, cmplx(0,0));
	DB2 = new DomainBase(*DB1);

	Domain *D1;
	Domain *D2;
	Domain *D3;
	D1 = new Domain(*F, cmplx(0,0));
	D2 = new Domain(*D1);
	D3 = new Domain(X, Y, cmplx(0,0));

	DomainManagerBase *DMB;
	DMB = new DomainManagerBase(1, 1, F, cmplx(0,0));

	DomainManager *DM;
	DM = new DomainManager(1, 1, *F, cmplx(0,0));


	delete X;
	delete Y;
	delete F;
	delete GB1;
	delete GB2;
	delete G1;
	delete G2;
	delete DB1;
	delete DB2;
	delete D1;
	delete D2;
	delete D3;
	delete DMB;
	delete DM;

	return 0;
}
