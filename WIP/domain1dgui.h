#ifndef DOMAIN1DTGUI_H
#define DOMAIN1DTGUI_H

#include <QEvent>
#include <QGraphicsScene>
#include <QGraphicsSceneWheelEvent>
#include <QGraphicsView>
#include <QGridLayout>
#include <QKeyEvent>
#include <QMainWindow>


#include "domain1d.h"
#include "solver1d.h"

class Domain1DGui : public QMainWindow
{
public:
	Domain1DGui(Domain1D *d);
	~Domain1DGui();

	void setDomain(Domain1D *d);
	void setValue(int i, Type y);
public slots:
	void refreshView();
protected:
	virtual bool eventFilter(QObject *obj, QEvent *event);
private:
	void Zoom(QGraphicsSceneWheelEvent *event);

private:

	Domain1D *domain;
	Solver1D *Solv;

	QGraphicsScene *scene;
	QGraphicsView *view;
	QGridLayout *GridLayout;
	QScrollArea *Scroll;
	QWidget *SDI_Area;

};



Domain1DGui::Domain1DGui(Domain1D *d): QMainWindow()
{

	domain = d;


	SDI_Area = new QWidget;
	GridLayout = new QGridLayout;

	scene = new QGraphicsScene;
	view = new QGraphicsView(scene);

	GridLayout->addWidget(view);


	view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	view->setDragMode(QGraphicsView::ScrollHandDrag);

	view->scene()->installEventFilter(this);

	SDI_Area->setLayout(GridLayout);
	setCentralWidget(SDI_Area);

	setWindowTitle("Solver1DGui");
	this->resize(800,800);
	refreshView();
	view->fitInView( view->scene()->sceneRect(), Qt::KeepAspectRatio );
}

bool Domain1DGui::eventFilter(QObject *obj, QEvent *event)
{
	if (event->type() == QEvent::GraphicsSceneWheel)
	{
		Zoom(static_cast<QGraphicsSceneWheelEvent*> (event));
		event->accept();//On ne propage pas l'event
		return true;
	}
	if (event->type()==QEvent::KeyPress)
	{
		QKeyEvent* key = static_cast<QKeyEvent*>(event);
		if ( (key->key()==Qt::Key_Enter) )
		{
			//Enter or return was pressed
		}

		return true;
	}

	return false;
}

void Domain1DGui::refreshView()
{
	this->setWindowTitle(QString::number(Solv->getTime()));
	scene->clear();
	QPen Pen;
	Pen.setWidth(1);
	Pen.setCosmetic(true);
	QPainterPath path;
	path.moveTo(0.,-std::abs(domain->getValue(0)));
	for(int i=1; i<domain->getN();++i)
	{
			path.lineTo((double)(i)*domain->getDx(),-std::abs(domain->getValue(i)));
	}
	view->scene()->addPath(path,Pen);
}

void Domain1DGui::setDomain(Domain1D *d)
{
	domain =d;
	refreshView();
}

void Domain1DGui::Zoom(QGraphicsSceneWheelEvent *event)
{
	qreal scaleFactor=pow((double)2, event->delta() / 240.0);//Calcul le Facteur de zoom
	view->scale(scaleFactor, scaleFactor);//Applique le zoom
}

Domain1DGui::~Domain1DGui()
{
	delete view;
	delete scene;
	delete GridLayout;

	delete domain;
	delete Solv;
}


#endif // DOMAIN1DTGUI_H
