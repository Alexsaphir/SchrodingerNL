#include "schrodingernlint1d.h"

SchrodingerNLInt1D::SchrodingerNLInt1D(int i, GridManagerBase *G, Type dx, Type dt, Type k): FunctionPDEVirtual(G)
{
	m_intIndex = i;
	m_idtdiv2dx = cmplx(0,dt/dx/2.);
	m_idtk = cmplx(0,dt*k);
}


cmplx SchrodingerNLInt1D::f(CoreMatrix *C) const
{
	cmplx tmp(0,0);
	cmplx uipnp(C->at(m_intIndex+1));
	cmplx uimnp(C->at(m_intIndex-1));
	cmplx uinp(C->at(m_intIndex));
	tmp += (uipnp + uimnp - 2.*uinp);
	tmp *= m_idtdiv2dx;
	tmp += m_idtk*(std::norm(uinp)*uinp);
	tmp += m_gridManager->getCurrentGrid()->getValue(m_intIndex) -uinp;

	return tmp;
}

cmplx SchrodingerNLInt1D::df(CoreMatrix *C, int var) const
{
	cmplx tmp(0,0);

	if(var == m_intIndex)
	{
		cmplx uinp(C->at(m_intIndex));
		tmp += m_idtdiv2dx*(-2.);
		Type x = uinp.real();
		Type y = uinp.imag();
		cmplx dnl((3.)*x*x+y*y-(2.)*x*y,);
	}
	else if(var == m_intIndex-1)
	{
		cmplx uimnp(C->at(m_intIndex-1));
		tmp += m_idtdiv2dx;
	}
	else if(var == m_intIndex+1)
	{
		cmplx uipnp(C->at(m_intIndex+1));
		tmp += m_idtdiv2dx;
	}

	return tmp;
}

SchrodingerNLInt1D::~SchrodingerNLInt1D()
{

}
