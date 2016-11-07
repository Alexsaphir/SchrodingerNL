#ifndef PDEGUI1D_H
#define PDEGUI1D_H

#include <QGraphicsScene>
#include <QGraphicsView>

#include "../pdeguivirtual.h"

class PDEGui1D: public PDEGuiVirtual
{
public:
	PDEGui1D();
	PDEGui1D(PDEVirtual *P);
	~PDEGui1D();

public slots:
	virtual void refreshView();

protected:
	virtual void Zoom(QGraphicsSceneWheelEvent *event);

protected:
	QGraphicsScene *scene;
	QGraphicsView *view;
};

#endif // PDEGUI1D_H
