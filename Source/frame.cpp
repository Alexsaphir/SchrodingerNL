#include "frame.h"

Frame::Frame()
{
	m_N = 0;
}

Frame::Frame(const Axis *X)
{
	m_Basis.push_back(X->clone());
	m_N = 1;
}

Frame::Frame(const Axis *X, const Axis *Y)
{
	m_Basis.push_back(X->clone());
	m_Basis.push_back(Y->clone());
	m_N = 2;
}

Frame::Frame(QVector<Axis*> axes)
{
	m_N = axes.size();
	if(m_N==0)
		return;
	for(int i=0; i<m_N; ++i)
		m_Basis.push_back(axes.at(i)->clone());

}

Frame::Frame::Frame(const Frame &F)
{
	for(int i=0; i<F.m_Basis.size(); ++i)
		m_Basis.push_back(F.m_Basis.at(i)->clone());
	m_N = F.size();
}

const Axis* Frame::at(int i) const
{
	return m_Basis.at(i);
}

const Axis* Frame::getAxis(int i) const
{
	if(i<0 || i>=m_Basis.size())
		return NULL;
	return m_Basis.at(i);
}

int Frame::size() const
{
	return m_N;
}


bool Frame::empty() const
{
	return (m_N==0);
}

Frame::~Frame()
{
	for(int i=0; i<m_Basis.size();++i)
		delete m_Basis.at(i);
}
