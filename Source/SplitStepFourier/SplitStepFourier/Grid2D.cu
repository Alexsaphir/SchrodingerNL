#include "Grid2D.cuh"

Grid2D::Grid2D(double Xmin, double Xmax, int Nx, double Ymin, double Ymax, int Ny)
{
	m_X = Axis(Xmin, Xmax, Nx);
	m_Y = Axis(Ymin, Ymax, Ny);

	m_nbPts = Nx*Ny;
	m_nbPtsX = Nx;
	m_nbPtsY = Ny;


}

Axis Grid2D::getAxisX() const
{
	return m_X;
}

Axis Grid2D::getAxisY() const
{
	return m_Y;
}

cmplx * Grid2D::getHostData() const
{
	return m_h_V;
}

cmplx * Grid2D::getDeviceData() const
{
	return m_d_V;
}

void Grid2D::syncDeviceToHost()
{
	cudaMemcpy(m_h_V, m_d_V, m_nbPts * sizeof(cmplx), cudaMemcpyDeviceToHost);
}

void Grid2D::syncHostToDevice()
{
	cudaMemcpy(m_d_V, m_h_V, m_nbPts * sizeof(cmplx), cudaMemcpyHostToDevice);
}

Grid2D::~Grid2D()
{
	delete m_h_V;
	cudaFree(m_d_V);
}
