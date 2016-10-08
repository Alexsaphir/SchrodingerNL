#include "solver1dgui.h"


Solver1DGui::Solver1DGui(const Axis &X, cmplx Binf, cmplx Bsup, Type timeStep) : Solver1D(X, Binf, Bsup, timeStep), QMainWindow()
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
}

void Solver1DGui::refreshView()
{
	this->setWindowTitle(QString::number((double)(this->getT()*this->getDt())));
	view->scene()->clear();
	QPen Pen;
	Pen.setWidth(1);
	Pen.setCosmetic(true);
	QPainterPath path;
	path.moveTo(0.,-this->getValueNorm(0));
	for(int i=1; i<this->getN();++i)
	{
		path.lineTo((double)(i)*this->getDx()*.1,-this->getValueNorm(i));//*.01 reduce the spread
	}
	view->scene()->addPath(path,Pen);
	//view->fitInView( view->scene()->sceneRect(), Qt::KeepAspectRatio );
}

void Solver1DGui::Zoom(QGraphicsSceneWheelEvent *event)
{
	qreal scaleFactor=pow((double)2, event->delta() / 240.0);//Calcul le Facteur de zoom
	view->scale(scaleFactor, scaleFactor);//Applique le zoom
}

bool Solver1DGui::eventFilter(QObject *obj, QEvent *event)
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
			for(int i=0;i<100;++i)
				this->doStep();
		}
		if (key->key()==Qt::Key_Space)
			this->doFourrier();
		if(key->key()==Qt::Key_B)
			this->undoFourrier();
		refreshView();
		return true;
	}

	return false;
}

Solver1DGui::~Solver1DGui()
{
	delete view;
	delete scene;
	delete GridLayout;
	//delete Scroll;
	delete SDI_Area;
}
