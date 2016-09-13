#include "solver1d.h"

Solver1D::Solver1D(Type Xmin, Type Xmax, Type Xstep, cmplx Binf, cmplx Bsup, Type timeStep)
{
	GridM = new GridManager(Xmin, Xmax, Xstep, Binf, Bsup, 3,1);

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
	return dt*(double)T;
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
	static int count(0);
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
			tmp -= 2.*Current->getValue(i);

			tmp /= 2.;//Spécifique à T==0
			tmp *= d;
			tmp *= -j;//Mult by %i

			tmp += Current->getValue(i);//Current car T==0

			Next->setValue(i,tmp);
		}
		err_integ =std::abs(Integration::integrate(getOldDomain())-Integration::integrate(getCurrentDomain()));
		qDebug() << err_integ;
	}
	else
	{
		#pragma omp parallel for
		for(int i=0;i<getN();++i)
		{
			cmplx tmp(0.,0.);
			tmp += Current->getValue(i+1);
			tmp += Current->getValue(i-1);
			tmp -= 2.*Current->getValue(i);
			tmp *= d;
			tmp *=-j;

			tmp += Old->getValue(i);
			tmp += -j*dt*std::abs(Current->getValue(i))*std::abs(Current->getValue(i))*Current->getValue(i);

			Next->setValue(i,tmp);
		}
	}
	++T;
	switchDomain();
	//qDebug() << Integration::integrate(getCurrentDomain());



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
	return 0.;
	return getValueNorm(i);
}

void Solver1D::initPulse()
{
	for(int i=0; i<this->getN();++i)
	{
		Type x(this->getPos(i));
		cmplx j(0,100.*x);
		this->setValue(i,std::exp(-x*x/4.)*std::exp(j));
		this->setValue(i,.1);
		if(i<(this->getN()*490./1000.))
			this->setValue(i,0.);
		if(i>(this->getN()*510./1000.))
			this->setValue(i,0.);
	}

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
