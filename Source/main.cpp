#include <QDebug>

#include "Axis/linearaxis.h"
#include "frame.h"

int main(int argc, char **argv)
{
	LinearAxis *X(NULL);
	Axis *Y(NULL);

	X = new LinearAxis(-10., 10., .1);
	Y = new LinearAxis(*X);

	Frame *F;

	F = new Frame(X, Y);

	delete X;
	delete Y;
	delete F;

	return 0;
}
