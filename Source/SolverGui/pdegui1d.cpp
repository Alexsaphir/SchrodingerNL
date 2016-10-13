#include "pdegui1d.h"

PDEGui1D::PDEGui1D(): QMainWindow()
{
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
	view->fitInView( view->scene()->sceneRect(), Qt::KeepAspectRatio );

	H = new Heat1D(Axis(-5,5,.01),1,0,0);
	H->initMatrix();
	H->pulse();
	axe=Axis(-5,5,.1);
}




void PDEGui1D::refreshView()
{
	//this->setWindowTitle(QString::number((double)(this->getT()*this->getDt())));
	view->scene()->clear();
	QPen Pen;
	Pen.setWidth(1);
	Pen.setCosmetic(true);
	QPainterPath path;
	path.moveTo(0.,-std::abs(H->get(0)));
	for(int i=1; i<axe.getAxisN();++i)
	{
		path.lineTo((double)(i)*.001,-100.*std::abs(H->get(i)));//*.01 reduce the spread
	}
	view->scene()->addPath(path,Pen);
	//view->fitInView( view->scene()->sceneRect(), Qt::KeepAspectRatio );
}

void PDEGui1D::Zoom(QGraphicsSceneWheelEvent *event)
{
	qreal scaleFactor=pow((double)2, event->delta() / 240.0);//Calcul le Facteur de zoom
	view->scale(scaleFactor, scaleFactor);//Applique le zoom
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
			//Enter or return was pressed-
			for(int i=0;i<1;++i)
				H->compute();
		}
		refreshView();
		return true;
	}

	return false;
}



PDEGui1D::~PDEGui1D()
{
	delete view;
	delete scene;
	delete GridLayout;
	//delete Scroll;
	delete SDI_Area;
}
