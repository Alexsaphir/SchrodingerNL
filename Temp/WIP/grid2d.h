#ifndef GRID2D_H
#define GRID2D_H

#include <QVector>


template<typename Type> class Grid2D
{
public:
	Grid2D(double Xmn, double Xmx, long double Xsp, long double Ymn, double Ymx, long double Ysp);

	double getDx() const;
	double getDy() const;
	double getPosX(int i, int j) const;
	double getPosY(int i, int j) const;
	Type getValue(int i,int j) const;
	double getXmax() const;
	double getXmin() const;
	double getYmax() const;
	double getYmin() const;
	int getNx() const;
	int getNy() const;
	void setValue(int i, int j, Type y);

private:
	QVector<Type> V;
	long double Xmin;
	long double Xmax;
	long double Ymin;
	long double Ymax;
	long double Xstep;
	long double Ystep;

	int nbPtsX;
	int nbPtsY;
};



template<class Type> Grid2D<Type>::Grid2D(double Xmn, double Xmx, long double Xsp, long double Ymn, double Ymx, long double Ysp)
{
	Xmin = Xmn;
	Xmax = Xmx;

	Ymin = Ymn;
	Ymax = Ymx;

	Xstep = Xsp;
	Ystep = Ysp;

	nbPtsX = (int)((-Xmin + Xmax)/Xstep)+1;
	nbPtsY = (int)((-Ymin + Ymax)/Ystep)+1;

	V.fill(0.,nbPtsX*nbPtsY);
}

template<class Type> double Grid2D<Type>::getDx() const
{
	return Xstep;
}

template<class Type> double Grid2D<Type>::getDy() const
{
	return Ystep;
}

template<class Type> int Grid2D<Type>::getNx() const
{
	return nbPtsX;
}

template<class Type> int Grid2D<Type>::getNy() const
{
	return nbPtsY;
}

template<class Type> double Grid2D<Type>::getPosX(int i, int j) const
{
	if(i<0)
		return 0;
}

template<class Type> double Grid2D<Type>::getPosY(int i, int j) const
{

}

template<class Type> Type Grid2D<Type>::getValue(int i, int j) const
{

}

template<class Type> double Grid2D<Type>::getXmax() const
{

}

template<class Type> double Grid2D<Type>::getXmin() const
{

}

template<class Type> double Grid2D<Type>::getYmax() const
{

}

template<class Type> double Grid2D<Type>::getYmin() const
{

}

template<class Type> void Grid2D<Type>::setValue(int i, int j, Type y)
{

}




























#endif // GRID2D_H
