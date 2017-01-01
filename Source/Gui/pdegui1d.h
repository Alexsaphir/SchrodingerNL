#ifndef PDEGUI1D_H
#define PDEGUI1D_H

#include <QKeyEvent>
#include <QGraphicsSceneWheelEvent>
#include <QMainWindow>
#include <QGridLayout>
#include <QGraphicsScene>
#include <QGraphicsView>

#include "../Grid/Base/gridbase.h"

class PDEGui1D: public QMainWindow
{
	Q_OBJECT
public:
	PDEGui1D();
	PDEGui1D(GridBase *P);
	~PDEGui1D();

public slots:
	virtual void refreshView();

protected:
	virtual bool eventFilter(QObject *obj, QEvent *event);

protected:
	virtual void Zoom(QGraphicsSceneWheelEvent *event);

protected:
	GridBase *Problem;

	QGridLayout *GridLayout;
	QWidget *SDI_Area;
	QGraphicsScene *scene;
	QGraphicsView *view;

};

#endif // PDEGUI1D_H
