#include <QApplication>

#include "grid1d.h"
#include "solver1dgui.h"

int main(int argc, char **argv)
{
	//QApplication app(argc, argv);
	Solver1D W(-1.,1.,1.,0.,0.,.1);
	W.initPulse();
	qDebug() << "Value of Init Pulse : " << W.getValueNorm(0) << W.getValueNorm(1) << W.getValueNorm(2);

	W.doStep();
	qDebug() << "Value at Step 1 : " << W.getValueNorm(0) << W.getValueNorm(1) << W.getValueNorm(2);

	W.doStep();
	qDebug() << "Value at Step 2 : " << W.getValueNorm(0) << W.getValueNorm(1) << W.getValueNorm(2);

	W.doStep();//Step 3
	W.doStep();//Step 4
	W.doStep();//Step 5
	W.doStep();//Step 6
	W.doStep();//Step 7
	W.doStep();//Step 8
	W.doStep();//Step 9
	W.doStep();//Step 10
	qDebug() << "Value at Step 2 : " << W.getValueNorm(0) << W.getValueNorm(1) << W.getValueNorm(2);

	return 0;
	//return app.exec();
}



