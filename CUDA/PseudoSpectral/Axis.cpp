#include "Axis.h"



Axis::Axis(double Xmin, double Xmax, int nbPts)
{
	m_Xmax = Xmax;
	m_Xmin = Xmin;
	m_nbPts = nbPts;

	if (Xmin > Xmax)
	{
		m_Xmin = Xmax;
	}

	if (m_nbPts < 2)
	{
		m_nbPts = 2;
	}

	if (m_Xmax == m_Xmin)
	{
		m_nbPts = 1;
	}
}

double Axis::getLinearValueAt(int i) const
{
	if (m_nbPts == 1)
	{
		return m_Xmin;//m_nbPts ==1 => m_nbPts-1 == 0 => divide by 0: error
	}

	return m_Xmin + static_cast<double>(i)*(m_Xmax - m_Xmin) / (static_cast<double>(m_nbPts - 1));
}

double Axis::getChebyshevValueAt(int i) const
{
	return (m_Xmax + m_Xmin) / 2. + (m_Xmax - m_Xmin)*.5*std::cos(static_cast<double>(i)*M_PI / static_cast<double>(m_nbPts));
}

double Axis::getFrequency(int i) const
{
	if (i < m_nbPts / 2)
		return 2.*M_PI*(static_cast<double>(i - m_nbPts)) / (m_Xmax - m_Xmin);
	else if (i > m_nbPts / 2)
		return 2.*M_PI*static_cast<double>(i) / (m_Xmax - m_Xmin);
	else
		return 0.;//k=N/2
	
	
	return 2.*M_PI*static_cast<double>(-m_nbPts+2*i)/2.;
}

Axis::~Axis()
{
}
