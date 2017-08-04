#include "pdegui1d.h"

PDEGui1D::PDEGui1D(): QMainWindow()
{
	Problem = NULL;

	SDI_Area = new QWidget;
	GridLayout = new QGridLayout;

	SDI_Area->setLayout(GridLayout);
	setCentralWidget(SDI_Area);

	setWindowTitle("PDE numerical simulation ");
	this->resize(800,800);

	scene = new QGraphicsScene;
	view = new QGraphicsView(scene);

	GridLayout->addWidget(view);

	view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	view->setDragMode(QGraphicsView::ScrollHandDrag);
	view->scene()->installEventFilter(this);
	refreshView();
	view->fitInView( view->scene()->sceneRect(), Qt::KeepAspectRatio );
}

PDEGui1D::PDEGui1D(GridBase *P):QMainWindow()
{

	Problem = P;

	SDI_Area = new QWidget;
	GridLayout = new QGridLayout;

	SDI_Area->setLayout(GridLayout);
	setCentralWidget(SDI_Area);

	setWindowTitle("PDE numerical simulation ");
	this->resize(800,800);
	scene = new QGraphicsScene;
	view = new QGraphicsView(scene);

	GridLayout->addWidget(view);

	view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	view->setDragMode(QGraphicsView::ScrollHandDrag);
	view->scene()->installEventFilter(this);
	refreshView();
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
	path.moveTo(0.,-std::norm(Problem->getValue(0)));
//	path.moveTo(0.,-Problem->getValue(0).real());

	Type dx = Problem->getStepOfAxis(0);
	int N = Problem->getSizeOfAxis(0);

	for(int i=1; i<N;++i)
	{
		path.lineTo((Type)(i)*Problem->getStepOfAxis(0),-std::norm(Problem->getValue(i)));
//		path.lineTo((Type)(i)*Problem->getStepOfAxis(0),-Problem->getValue(i).real());
	}
	view->scene()->addPath(path,Pen);
}

void PDEGui1D::Zoom(QGraphicsSceneWheelEvent *event)
{
	qreal scaleFactor=pow((double)2, event->delta() / 240.0);
	view->scale(scaleFactor, scaleFactor);
}

bool PDEGui1D::eventFilter(QObject *obj, QEvent *event)
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
			qDebug() << "call";
			//Enter or return was pressed-
			//if (Problem)
				//Problem->computeNextStep();
		}
		refreshView();
		return true;
	}
	return false;
}

PDEGui1D::~PDEGui1D()
{
	delete scene;
	delete view;
}
