#include "Signal.cuh"


Signal::Signal(double Xmin, double Xmax, int res) :m_Xmin(Xmin), m_Xmax(Xmax), m_nbPts(res), m_d_V(NULL), m_h_V(NULL)
{
	if (Xmin > Xmax)
		return;
	if (m_nbPts <= 0)
		return;

	m_h_V = new (std::nothrow) cmplx[m_nbPts];
	cudaMalloc(&m_d_V, m_nbPts * sizeof(cmplx));
}

cmplx* Signal::getHostData() const
{
	return m_h_V;
}

cmplx* Signal::getDeviceData() const
{
	return m_d_V;
}

double Signal::getXmin() const
{
	return m_Xmin;
}

double Signal::getXmax() const
{
	return m_Xmax;
}

int Signal::getSignalPoints() const
{
	return m_nbPts;
}

void Signal::syncHostToDevice()
{
	cudaMemcpy(m_d_V, m_h_V, m_nbPts * sizeof(cmplx), cudaMemcpyHostToDevice);
}

void Signal::syncDeviceToHost()
{
	cudaMemcpy(m_h_V, m_d_V, m_nbPts * sizeof(cmplx), cudaMemcpyDeviceToHost);
}

void Signal::fillHost(cmplx value)
{
	for (int i = 0; i < m_nbPts; ++i)
	{
		m_h_V[i] = value;
	}
}

void Signal::fillDevice(cmplx value)
{
}

void Signal::fillBoth(cmplx value)
{
	fillHost(value);
	syncHostToDevice();
}

Signal::~Signal()
{
	delete[] m_h_V;
	cudaFree(m_d_V);
}

void SinPulseEqui(Signal * S)
{
	for (int i = 0; i < S->getSignalPoints(); ++i)
	{
		S->getHostData()[i] = make_cuDoubleComplex(std::sin(S->getXmin() + static_cast<double>(i)*(S->getXmax() - S->getXmin()) / (static_cast<double>(S->getSignalPoints()) - 1)), 0);
	}
	S->syncHostToDevice();
}

void CosWTPulseEqui(Signal * S)
{
	for (int i = 0; i < S->getSignalPoints(); ++i)
	{
		S->getHostData()[i] = make_cuDoubleComplex(std::cos(S->getXmin() + static_cast<double>(i)*(S->getXmax() - S->getXmin()) / (static_cast<double>(S->getSignalPoints()) - 1)), 0);
	}
	S->syncHostToDevice();
}

void Trigo(Signal * S,double freq)
{
	for (int i = 0; i < S->getSignalPoints(); ++i)
	{
		
		S->getHostData()[i] = cuCexp(make_cuDoubleComplex(0, freq*static_cast<double>(i)*(S->getXmax() - S->getXmin()) / (static_cast<double>(S->getSignalPoints()) - 1)));
	}
	S->syncHostToDevice();
}

void GaussPulseLinear(Signal *S, double fc, double bw, double bwr, double tpr)
{
	Axis X(S->getXmin(), S->getXmax(), S->getSignalPoints());
	
	double ref = std::pow(10., bwr / 20.);
	double a = -(M_PI*fc*bw)*(M_PI*fc*bw) / 4. / std::log(ref);

	for (int i = 0; i < S->getSignalPoints(); ++i)
	{
		double t = X.getLinearValueAt(i);
		double yenv = std::exp(-a*t*t);
		S->getHostData()[i] = yenv*make_cuDoubleComplex(std::cos(2.*M_PI*fc*t), std::sin(2.*M_PI*fc*t));
	}
	S->syncHostToDevice();
}
