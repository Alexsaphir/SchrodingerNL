#include <QApplication>

#include "grid1d.h"
#include "solver1dgui.h"

int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	Solver1DGui W(-100.,100.,.01,-10.,-10.,.00000001);
	W.initPulse();
	W.refreshView();
	W.show();


	//return 0;
	return app.exec();
}



