#include <QApplication>

#include "include/grid1d.h"
#include "include/solver1dgui.h"

int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	Solver1DGui W(-100.,100.,.1,0.,0.,.001);
	W.initPulse();
	W.refreshView();
	W.show();
	return app.exec();
}



