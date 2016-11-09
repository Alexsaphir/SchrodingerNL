#ifndef DEBUGCLASS_H
#define DEBUGCLASS_H

#include <QDebug>
#include <QString>

#include <complex>

#include "Axis/axis.h"

QDebug operator<< (QDebug dbg, const cmplx &z);
QDebug operator<< (QDebug dbg, const Axis &z);



#endif // DEBUGCLASS_H
