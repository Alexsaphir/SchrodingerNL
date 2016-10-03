#include <QApplication>
#include <QDebug>

#include "SolverGui/solver1dgui.h"

int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	Solver1DGui W(Axis(-10,10,.1),0,0,.001);
	W.initPulse();
	W.refreshView();
	W.show();

	return app.exec();

}
