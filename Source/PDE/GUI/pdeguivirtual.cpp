#include "pdeguivirtual.h"

PDEGuiVirtual::PDEGuiVirtual(): QMainWindow()
{
	Problem = NULL;

	SDI_Area = new QWidget;
	GridLayout = new QGridLayout;

	SDI_Area->setLayout(GridLayout);
	setCentralWidget(SDI_Area);

	setWindowTitle("PDE numerical simulation ");
	this->resize(800,800);
}

bool PDEGuiVirtual::eventFilter(QObject *obj, QEvent *event)
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
			if (!Problem)
				Problem->computeNextStep();
		}
		refreshView();
		return true;
	}
	return false;
}


PDEGuiVirtual::~PDEGuiVirtual()
{
	delete GridLayout;
	delete SDI_Area;
}
