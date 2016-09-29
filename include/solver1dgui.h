#ifndef SOLVER1DGUI_H
#define SOLVER1DGUI_H

#include <QEvent>
#include <QGraphicsScene>
#include <QGraphicsSceneWheelEvent>
#include <QGraphicsView>
#include <QGridLayout>
#include <QKeyEvent>
#include <QMainWindow>

#include "solver1d.h"

class Solver1D;

class Solver1DGui : public Solver1D, public QMainWindow
{
public:
	Solver1DGui(const Axis &X, cmplx Binf, cmplx Bsup, Type timeStep);
	~Solver1DGui();

public slots:
	void refreshView();
protected:
	virtual bool eventFilter(QObject *obj, QEvent *event);
private:
	void Zoom(QGraphicsSceneWheelEvent *event);

private:


	QGraphicsScene *scene;
	QGraphicsView *view;
	QGridLayout *GridLayout;
	QScrollArea *Scroll;
	QWidget *SDI_Area;

};




#endif // SOLVER1DGUI_H
