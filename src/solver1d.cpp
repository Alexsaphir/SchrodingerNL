#include "include/solver1d.h"

Solver1D::Solver1D(const Axis &X, cmplx Binf, cmplx Bsup, Type timeStep)
{
	GridM = new GridManager(X, Binf, Bsup, 3,1);

	dt = timeStep;
	T = 0;
	err_integ = 0;
}

Type Solver1D::getDt() const
{
	return dt;
}

Type Solver1D::getDx() const
{
	return getCurrentDomain()->getDx();
}

cmplx Solver1D::getNextValue(int i) const
{
	return getNextDomain()->getValue(i);
}

cmplx Solver1D::getOldValue(int i) const
{
	return getOldDomain()->getValue(i);
}

Type Solver1D::getPos(int i) const
{
	return getCurrentDomain()->getPos(i);
}

double Solver1D::getTime() const
{
	return (double)dt*(double)T;
}

cmplx Solver1D::getValue(int i) const
{
	return getCurrentDomain()->getValue(i);
}

double Solver1D::getValueNorm(int i) const
{
	return std::abs(getValue(i));


}

Type Solver1D::getXmax() const
{
	return getCurrentDomain()->getXmax();
}

Type Solver1D::getXmin() const
{
	return getCurrentDomain()->getXmin();
}

int Solver1D::getN() const
{
	return getCurrentDomain()->getN();
}

int Solver1D::getT() const
{
	return T;
}

void Solver1D::doStep()
{
	/*Les conditions au bords de type sont gérés par Domain
	 * Il faut juste specifier le calcul pour T=0
	*/
	Domain1D *Next		= getNextDomain();
	Domain1D *Current	= getCurrentDomain();
	Domain1D *Old		= getOldDomain();
	cmplx j(0,1.);
	Type d=getDt()/getDx()/getDx();

	if(T==0)
	{
		#pragma omp parallel for
		for(int i=0;i<getN();++i)
		{

			cmplx tmp(0.,0.);

			tmp += Current->getValue(i+1);
			tmp += Current->getValue(i-1);
			tmp -= (Type)2.*Current->getValue(i);

			tmp /= 2.;//Spécifique à T==0

			tmp *= d;
			tmp *= -j;//Mult by %i

			tmp += Current->getValue(i);//Current car T==0
			tmp+=-j*getDt()*V(i)*getValue(i);

			Next->setValue(i,tmp);
		}
	}
	else
	{
		#pragma omp parallel for
		for(int i=0;i<getN();++i)
		{
			cmplx tmp(0.,0.);
			tmp += Current->getValue(i+1);
			tmp += Current->getValue(i-1);
			tmp -= (Type)2.*Current->getValue(i);
			//qDebug() << i<< tmp.real() << tmp.imag();
			tmp *= d;
			tmp *=-j;

			tmp += Old->getValue(i);
			tmp+=-j*2.*getDt()*V(i)*getValue(i);
			Next->setValue(i,tmp);
		}
	}
	++T;
	switchDomain();
}

void Solver1D::setValue(int i, cmplx y)
{
	getCurrentDomain()->setValue(i, y);
}

void Solver1D::switchDomain()
{
	GridM->switchDomain();

}

Domain1D* Solver1D::getCurrentDomain() const
{
	return GridM->getCurrentDomain();
}

Domain1D* Solver1D::getNextDomain() const
{
	return GridM->getNextDomain();
}

Domain1D* Solver1D::getOldDomain() const
{
	return GridM->getDomain(-1);
}



Type Solver1D::V(int i) const
{
	return 0;
	if(i==getN()/3)
		return 1000.;
	else
		return 0;
}

void Solver1D::initPulse()
{
	for(int i=0; i<=(getN()-1)/2;++i)
	{
		Type x(getPos(i));
		cmplx w(0,100.*x);
		cmplx tmp=std::exp(-(x*x)/(Type)4.)*std::exp(w);

		setValue(getN()-1-i,tmp);
		setValue(i,std::conj(tmp));
	}
//	for(int i=0; i<getN();++i)
//	{
//		Type x(getPos(i));
//		cmplx w(0,100.*x);
//		cmplx tmp=std::exp(-(x*x)/(Type)4.)*std::exp(w);
//		setValue(i,tmp);
//	}
}

void Solver1D::doFourrier()
{
	getCurrentDomain()->doFourrier();
}

void Solver1D::undoFourrier()
{
	getCurrentDomain()->undoFourrier();
}


Solver1D::~Solver1D()
{
	delete GridM;
}
