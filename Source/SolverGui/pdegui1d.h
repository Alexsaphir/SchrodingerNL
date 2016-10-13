#ifndef PDEGUI1D_H
#define PDEGUI1D_H

#include <QEvent>
#include <QGraphicsScene>
#include <QGraphicsSceneWheelEvent>
#include <QGraphicsView>
#include <QGridLayout>
#include <QKeyEvent>
#include <QMainWindow>
#include "../PDE/Linear/1D/heat1d.h"

class PDEGui1D: public QMainWindow
{
public:
	PDEGui1D();
	~PDEGui1D();
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
	//QScrollArea *Scroll;
	QWidget *SDI_Area;
	Heat1D *H;
	Axis axe;
};

#endif // PDEGUI1D_H
