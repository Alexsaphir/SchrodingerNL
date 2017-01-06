#ifndef DEBUGCLASS_H
#define DEBUGCLASS_H

#include <QDebug>
#include <QString>

#include <complex>

#include "Axis/axis.h"
#include "Matrix/corematrix.h"

QDebug operator<< (QDebug dbg, const cmplx &z);
QDebug operator<< (QDebug dbg, const Axis &z);
QDebug operator<< (QDebug dbg, const CoreMatrix &M);


#endif // DEBUGCLASS_H
