#include "Grid.cuh"

Grid::Grid(double Xmin, double Xmax, int N) :m_X(Xmin, Xmax, N)
{
	m_nbPts = m_X.getN();
	m_h_V = new (std::nothrow) cmplx[m_nbPts];
	cudaError Err = cudaMalloc(&m_d_V, m_nbPts * sizeof(cmplx));
	if (Err != 0)
		throw std::exception("Memory Allocation Error!!!!");
}

Axis Grid::getAxis() const
{
	return m_X;
}

cmplx * Grid::getHostData() const
{
	return m_h_V;
}

cmplx * Grid::getDeviceData() const
{
	return m_d_V;
}

void Grid::syncDeviceToHost()
{
	cudaMemcpy(m_h_V, m_d_V, m_nbPts * sizeof(cmplx), cudaMemcpyDeviceToHost);
}

void Grid::syncHostToDevice()
{
	cudaMemcpy(m_d_V, m_h_V, m_nbPts * sizeof(cmplx), cudaMemcpyHostToDevice);
}

Grid::~Grid()
{
	delete[] m_h_V;
	cudaFree(m_d_V);
}
