#ifndef PDEGUIVIRTUAL_H
#define PDEGUIVIRTUAL_H

#include <QEvent>
#include <QGraphicsSceneWheelEvent>
#include <QGridLayout>
#include <QKeyEvent>
#include <QMainWindow>

#include "../pdevirtual.h"


class PDEGuiVirtual: public QMainWindow
{
public:
	PDEGuiVirtual();
	~PDEGuiVirtual();

public slots:
	virtual void refreshView() = 0;
protected:
	virtual bool eventFilter(QObject *obj, QEvent *event);
protected:
	virtual void Zoom(QGraphicsSceneWheelEvent *event) = 0;

protected:
	PDEVirtual *Problem;


	QGridLayout *GridLayout;
	//QScrollArea *Scroll;
	QWidget *SDI_Area;

};

#endif // PDEGUIVIRTUAL_H
