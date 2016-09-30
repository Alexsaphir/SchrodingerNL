#include <QApplication>
#include <QDebug>

#include "include/solver1dgui.h"

int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	Solver1DGui W(Axis(-10,10,.1),0,0,.001);
	W.initPulse();
	W.show();
//	Axis X(-10,10,1);
//	Grid1D G(X);

//	qDebug() << X.getAxisMin() << G.getN();

	return app.exec();
	return 0;
}
