#include "GridAugmented.cuh"

GridAugmented::GridAugmented(double Xmin, double Xmax, int N, int N_padding) :m_X(Xmin, Xmax, N - N_padding), m_Xa(Xmin, Xmax + static_cast<double>(N_padding)*(Xmax - Xmin) / static_cast<double>(N - N_padding), N)
{
	m_nbPts = m_X.getN();
	m_nbPtsPadding = N_padding;

	m_h_V = new (std::nothrow) cmplx[m_nbPts + m_nbPtsPadding];
	cudaError Err = cudaMalloc(&m_d_V, (m_nbPts + m_nbPtsPadding) * sizeof(cmplx));
	if (Err != 0)
		throw std::exception("Memory Allocation Error!!!!");

	m_h_Vp = m_h_V + m_nbPts;
	m_d_Vp = m_d_V + m_nbPts;

}

Axis GridAugmented::getAxis() const
{
	return m_Xa;
}

Axis GridAugmented::getAxisData() const
{
	return m_X;
}

cmplx * GridAugmented::getHostData() const
{
	return m_h_V;
}

cmplx * GridAugmented::getDeviceData() const
{
	return m_d_V;
}

cmplx * GridAugmented::getHostPaddedData() const
{
	return m_h_Vp;
}

cmplx * GridAugmented::getDevicePaddedData() const
{
	return m_d_Vp;
}

void GridAugmented::syncDeviceToHost()
{
	cudaMemcpy(m_h_V, m_d_V, (m_nbPts + m_nbPtsPadding) * sizeof(cmplx), cudaMemcpyDeviceToHost);
}

void GridAugmented::syncHostToDevice()
{
	cudaMemcpy(m_d_V, m_h_V, (m_nbPts + m_nbPtsPadding) * sizeof(cmplx), cudaMemcpyHostToDevice);
}

GridAugmented::~GridAugmented()
{
	delete[] m_h_V;
	cudaFree(m_d_V);
}
