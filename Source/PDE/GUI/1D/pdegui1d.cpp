#include "pdegui1d.h"

PDEGui1D::PDEGui1D(): PDEGuiVirtual()
{
	scene = new QGraphicsScene;
	view = new QGraphicsView(scene);

	GridLayout->addWidget(view);


	view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	view->setDragMode(QGraphicsView::ScrollHandDrag);
	view->scene()->installEventFilter(this);
	view->fitInView( view->scene()->sceneRect(), Qt::KeepAspectRatio );
}

void PDEGui1D::refreshView()
{
	if(!Problem)
		return;

	view->scene()->clear();//Clear the QGraphicsScene

	QPen Pen;//Create a pen
	Pen.setWidth(1);
	Pen.setCosmetic(true);

	QPainterPath path;
	path.moveTo(0.,-std::norm(Problem->at(0)));

	Axis *Axe = Problem->Repere->at(0);

	for(uint i=1; i<Axe->getAxisN();++i)
	{
		path.lineTo((Type)(i)*Axe->getAxisStep(i)*.1,-std::norm(Problem->at(i)));//*.01 reduce the spread
	}
	view->scene()->addPath(path,Pen);
}

void PDEGui1D::Zoom(QGraphicsSceneWheelEvent *event)
{
	qreal scaleFactor=pow((double)2, event->delta() / 240.0);
	view->scale(scaleFactor, scaleFactor);
}

PDEGui1D::~PDEGui1D()
{
	delete scene;
	delete view;
}
