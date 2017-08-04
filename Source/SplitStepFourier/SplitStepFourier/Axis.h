#pragma once

#include <cmath>
#include <algorithm>

class Axis
{
public:
	Axis(double Xmin, double Xmax, int nbPts);
	double getXmin() const;
	double getXmax() const;
	int getN() const;

	double getValueAt(int i) const;
	~Axis();
private:
	double m_Xmin;
	double m_Xmax;
	int m_nbPts;
};

