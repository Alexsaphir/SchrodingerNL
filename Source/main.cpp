#include <QApplication>
#include <QDebug>

#include "Axis/linearaxis.h"
#include "PDE/Linear/1D/schrodinger1d.h"
#include "PDE/GUI/1D/pdegui1d.h"

int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	Schrodinger1D *PDE_Schrodinger = NULL;
	PDEGui1D *Gui_Schrodinger = NULL;
	Axis *Z = NULL;

	Z = new LinearAxis(-10,10,.1);

	PDE_Schrodinger = new Schrodinger1D(Z, .1);
	PDE_Schrodinger->initializeLinearSolver();
	PDE_Schrodinger->InitialState();

	Gui_Schrodinger = new PDEGui1D(PDE_Schrodinger);


	Gui_Schrodinger->show();


	//delete PDE_Schrodinger;
	//delete Gui_Schrodinger;
	//delete Axe;

	return app.exec();
}
