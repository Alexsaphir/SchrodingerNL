#include "Axis.h"

Axis::Axis(double Xmin, double Xmax, int nbPts)
{
	m_Xmax = Xmax;
	m_Xmin = Xmin;
	m_nbPts = nbPts;

	if (Xmin > Xmax)
	{
		std::swap(m_Xmax, m_Xmin);
	}

	if (m_nbPts < 1)
	{
		m_nbPts = 1;
	}

	if (m_Xmax == m_Xmin)
	{
		m_nbPts = 1;
	}
}

double Axis::getXmin() const
{
	return m_Xmin;
}

double Axis::getXmax() const
{
	return m_Xmax;
}

int Axis::getN() const
{
	return m_nbPts;
}

double Axis::getValueAt(int i) const
{
	if (m_nbPts == 1)
	{
		return m_Xmin;//m_nbPts ==1 => m_nbPts-1 == 0 => divide by 0: error
	}
	return m_Xmin + static_cast<double>(i)*(m_Xmax - m_Xmin) / (static_cast<double>(m_nbPts));
}

Axis::~Axis()
{
}
