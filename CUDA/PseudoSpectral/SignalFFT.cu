#include "SignalFFT.cuh"

namespace
{
	__global__ void kernelResizeDataFFT(cmplx * d_V, int nbPts)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;
		if (i < nbPts)
		{
			d_V[i] = cuCmul(d_V[i], make_cuDoubleComplex(1. / sqrt(static_cast<double>(nbPts)), 0));
		}
	}

	__global__ void kernelFirstDerivative(cmplx * d_V, int nbPts)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;

		double k = static_cast<double>(i);
		cmplx Freq;


		//Positive Frequency are between 0 and <N/2
		//Negative Frequency are between N/2and <N
		if (i < nbPts / 2)
			Freq = make_cuDoubleComplex(0, k *2.*M_PI*k / static_cast<double>(nbPts));
		else if (i > nbPts / 2)
			Freq = make_cuDoubleComplex(0, (k - static_cast<double>(nbPts))*2.*M_PI*k / static_cast<double>(nbPts));
		else
			Freq = make_cuDoubleComplex(0, 0);//k=N/2
		__syncthreads();
		if (i < nbPts)
			d_V[i] = cuCmul(Freq, d_V[i]);
	}

}

SignalFFT::SignalFFT(int nbPoints) :m_nbPts(nbPoints > 0 ? nbPoints : 0), m_d_V(NULL), m_h_V(NULL)
{
	//create plan for complex to complex
	cufftPlan1d(&m_plan, nbPoints, CUFFT_Z2Z, 1);
	cudaMallocHost(&m_h_V, m_nbPts * sizeof(cmplx));
	cudaMalloc(&m_d_V, m_nbPts * sizeof(cmplx));

	m_thread = 1024;
	m_block = ((m_nbPts % m_thread) == 0) ? (m_nbPts / m_thread) : (1 + m_nbPts / m_thread);
	m_GPUOrder = true;
}

int SignalFFT::getSignalPoints() const
{
	return m_nbPts;
}

cmplx * SignalFFT::getHostData() const
{
	return m_h_V;
}

cmplx * SignalFFT::getDeviceData() const
{
	return m_d_V;
}

void SignalFFT::syncHostToDevice()
{
	cudaMemcpy(m_d_V, m_h_V, m_nbPts * sizeof(cmplx), cudaMemcpyHostToDevice);
}

void SignalFFT::syncDeviceToHost()
{
	cudaMemcpy(m_h_V, m_d_V, m_nbPts * sizeof(cmplx), cudaMemcpyDeviceToHost);
	m_GPUOrder = true;
}

void SignalFFT::computeFFT(Signal * src)
{
	m_GPUOrder = true;
	cufftExecZ2Z(m_plan, src->getDeviceData() , m_d_V, CUFFT_FORWARD);
	kernelResizeDataFFT << < m_block, m_thread >> > (m_d_V, m_nbPts);
}

void SignalFFT::ComputeSignal(Signal * dst)
{
	cufftExecZ2Z(m_plan, m_d_V, dst->getDeviceData(), CUFFT_INVERSE);
	kernelResizeDataFFT << < m_block, m_thread >> > (dst->getDeviceData(), m_nbPts);
}

void SignalFFT::smoothFilterCesaro()
{
	reorderData();
	for (int k = -m_nbPts / 2; k < ( m_nbPts / 2); ++k)
	{
		m_h_V[k + m_nbPts / 2] = m_h_V[k + m_nbPts / 2] * (1. - std::abs(k) / (static_cast<double>(m_nbPts) / 2.));
	}
	cancelReorderData();
}

void SignalFFT::smoothFilterLanczos()
{
	reorderData();
	for (int k = -m_nbPts / 2; k < (m_nbPts / 2); ++k)
	{
		m_h_V[k + m_nbPts / 2] = m_h_V[k + m_nbPts / 2] * std::sin(2.*M_PI*static_cast<double>(k) / static_cast<double>(m_nbPts)) / (2.*M_PI*static_cast<double>(k) / static_cast<double>(m_nbPts));
	}
	cancelReorderData();
}

void SignalFFT::smoothFilterRaisedCosinus()
{
	reorderData();
	for (int k = -m_nbPts / 2; k < (m_nbPts / 2); ++k)
	{
		m_h_V[k + m_nbPts / 2] = m_h_V[k + m_nbPts / 2] * (.5 + .5*std::cos(2.*M_PI*static_cast<double>(k) / static_cast<double>(m_nbPts)));
	}
	cancelReorderData();
}

void SignalFFT::reorderData()
{
	if (!m_GPUOrder)
		return;
	else
	{
		for (int i = 0; i < m_nbPts / 2; ++i)
		{
			cmplx tmp = m_h_V[i];
			m_h_V[i] = m_h_V[i + m_nbPts / 2];
			m_h_V[i + m_nbPts / 2] = tmp;
		}
		m_GPUOrder = false;
	}
}

void SignalFFT::cancelReorderData()
{
	if (m_GPUOrder)
		return;
	else
	{
		for (int i = 0; i < m_nbPts / 2; ++i)
		{
			cmplx tmp = m_h_V[i];
			m_h_V[i] = m_h_V[i + m_nbPts / 2];
			m_h_V[i + m_nbPts / 2] = tmp;
		}
		m_GPUOrder = true;
	}
}

void SignalFFT::firstDerivative()
{
	kernelFirstDerivative << < m_block, m_thread >> > (getDeviceData(), m_nbPts);
}

SignalFFT::~SignalFFT()
{
	cufftDestroy(m_plan);
	cudaFreeHost(m_h_V);
	cudaFree(m_d_V);
}
