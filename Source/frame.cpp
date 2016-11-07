#include "frame.h"

Frame::Frame()
{
	N = 0;
}

Frame::Frame(const Axis *X)
{
	Basis.push_back(X->clone());
	N = 1;
}

Frame::Frame(const Axis *X, const Axis *Y)
{
	Basis.push_back(X->clone());
	Basis.push_back(Y->clone());
	N = 2;
}

Frame::Frame(QVector<Axis*> axes)
{
	N = axes.size();
	if(N==0)
		return;
	for(int i=0; i<N; ++i)
		Basis.push_back(axes.at(i)->clone());

}

Frame::Frame::Frame(const Frame &F)
{
	for(int i=0; i<F.Basis.size(); ++i)
		Basis.push_back(F.Basis.at(i)->clone());
	N = F.size();
}

Axis* Frame::at(int i) const
{
	return getAxis(i);
}

Axis* Frame::getAxis(int i) const
{
	if(i<0 || i>=Basis.size())
		return NULL;
	return Basis.at(i);
}

int Frame::size() const
{
	return N;
}

bool Frame::empty() const
{
	return (N==0);
}

Frame::~Frame()
{
	for(int i=0; i<Basis.size();++i)
		delete Basis.at(i);
}
