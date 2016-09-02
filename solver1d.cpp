#include "solver1d.h"

Solver1D::Solver1D(double Xmin, double Xmax, double Xstep, cmplx Binf, cmplx Bsup, double timeStep)
{
	Dom1 = new Domain1D(Xmin, Xmax, Xstep, Binf, Bsup);
	Dom2 = new Domain1D(Xmin, Xmax, Xstep, Binf, Bsup);
	Dom3 = new Domain1D(Xmin, Xmax, Xstep, Binf, Bsup);

	Dom1isCurrent = true;
	Dom2isCurrent = false;
	Dom3isCurrent = false;

	dt = timeStep;
	T = 0;
}

double Solver1D::getDt() const
{
	return dt;
}

double Solver1D::getDx() const
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

double Solver1D::getPos(int i) const
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
	if(T==0)
		return getValue(i).real()*getValue(i).real() + getValue(i).imag()*getValue(i).imag();

	if((T)%2==0)
	{
		double temp(0.);

		temp+=getValue(i).real();
		temp*=temp;
		temp+=getNextValue(i).imag()*getOldValue(i).imag();

		return temp;
	}
	else
	{
		double temp(0.);

		temp+=getValue(i).imag();
		temp*=temp;
		temp+=getNextValue(i).real()*getOldValue(i).real();

		return temp;
	}
}

double Solver1D::getXmax() const
{
	return getCurrentDomain()->getXmax();
}

double Solver1D::getXmin() const
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
	//cmplx is complex<.....>


	cmplx cste(0., dt/(24.*getDx())/getDx());

	if (T>=0)
	{
		Domain1D* Old=getOldDomain();
		Domain1D* Current=getCurrentDomain();
		Domain1D* Next=getNextDomain();

		for(int i=0;i<getN();++i)
		{
			cmplx unit(0.,1.);
			cmplx tmp;
			tmp=getValue(i).imag()*getValue(i).imag() + getValue(i).real()*getValue(i).real();
			tmp*=getValue(i);
			tmp*=unit;
			tmp+=getValue(i);
			Next->setValue(i, cste*(-Current->getValue(i-2) + 16.L*Current->getValue(i-1) - 30.L*Current->getValue(i) + 16.L*Current->getValue(i+1) - Current->getValue(i+2) )-tmp);
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
	if(Dom1isCurrent)
	{
		Dom1isCurrent = false;
		Dom2isCurrent = true;
		return;
	}
	if(Dom2isCurrent)
	{
		Dom3isCurrent = true;
		Dom2isCurrent = false;
		return;
	}
	if(Dom3isCurrent)
	{
		Dom1isCurrent = true;
		Dom3isCurrent = false;
		return;
	}

}

Domain1D* Solver1D::getCurrentDomain() const
{
	if(Dom1isCurrent)
		return Dom1;
	else if(Dom2isCurrent)
		return Dom2;

	return Dom3;
}

Domain1D* Solver1D::getNextDomain() const
{
	if(Dom1isCurrent)
		return Dom2;
	else if(Dom2isCurrent)
		return Dom3;
	return Dom1;
}

Domain1D* Solver1D::getOldDomain() const
{
	if(Dom1isCurrent)
		return Dom3;
	else if(Dom2isCurrent)
		return Dom1;
	return Dom2;
}



double Solver1D::V(int i) const
{
	return 0.;
	return getValueNorm(i);
}

void Solver1D::initPulse()
{
	for(int i=0; i<this->getN();++i)
	{
		long double x(this->getPos(i));
		cmplx j(0,1.);
		this->setValue(i,.5*std::exp(-1.*x*x/4.)*std::exp((Type)100.*j*x));
		if(std::abs(this->getValue(i))<0.01)
			this->setValue(i,0.);
	}
}



Solver1D::~Solver1D()
{
	delete Dom1;
	delete Dom2;
}
