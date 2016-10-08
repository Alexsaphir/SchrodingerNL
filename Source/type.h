#ifndef TYPE_H
#define TYPE_H

#include <QDebug>

#include <QPair>
#include <QString>

#include <complex>

typedef double Type;
typedef std::complex<Type> cmplx;
typedef QPair<int, int> Position;

QDebug operator<< (QDebug dbg, const cmplx &z);

#endif // TYPE_H
