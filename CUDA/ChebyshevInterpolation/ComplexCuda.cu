#include "ComplexCuda.cuh"



//Operator overloading for cmplx

__host__ __device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex z)
{
	double factor = exp(z.x);
	return make_cuDoubleComplex(factor * cos(z.y), factor * sin(z.y));
}


__device__ __host__ __inline__ cmplx operator+(const cmplx &a, const cmplx &b)
{
	return cuCadd(a, b);
}

__device__ __host__ __inline__ cmplx operator+(const double &a, const cmplx &b)
{
	return make_cuDoubleComplex(a + b.x, b.y);
}

__device__ __host__ __inline__ cmplx operator+(const cmplx  &a, const double &b)
{
	return make_cuDoubleComplex(a.x + b, a.y);
}

__device__ __host__ __inline__ cmplx operator-(const cmplx &a, const cmplx &b)
{
	return cuCsub(a, b);
}

__device__ __host__ __inline__ cmplx operator-(const double &a, const cmplx &b)
{
	return make_cuDoubleComplex(a - b.x, -b.y);
}

__device__ __host__ __inline__ cmplx operator-(const cmplx &a, const double &b)
{
	return make_cuDoubleComplex(a.x - b, a.y);
}

__device__ __host__ __inline__ cmplx operator-(const cmplx &a)
{
	return make_cuDoubleComplex(-a.x, -a.y);
}

__device__ __host__ __inline__ cmplx operator*(const cmplx &a, const cmplx &b)
{
	return cuCmul(a, b);
}

__device__ __host__ __inline__ cmplx operator*(const double &a, const cmplx &b)
{
	return make_cuDoubleComplex(a*cuCreal(b), a*cuCimag(b));
}

__device__ __host__ __inline__ cmplx operator*(const cmplx &a, const double &b)
{
	return make_cuDoubleComplex(b*cuCreal(a), b*cuCimag(a));
}

__device__ __host__ __inline__ cmplx operator/(const cmplx &a, const cmplx &b)
{
	return cuCdiv(a, b);
}

__device__ __host__ __inline__ cmplx operator/(const cmplx &a, const double &b)
{
	return make_cuDoubleComplex(cuCreal(a) / b, cuCimag(a) / b);
}

__device__ __host__ __inline__ cmplx operator/(const double &a, const cmplx &b)
{
	return make_cuDoubleComplex(a, 0) / b;
}

__device__ __host__ __inline__ cmplx iMul(const cmplx &a)
{
	return make_cuDoubleComplex(-cuCimag(a), cuCreal(a));
}

__device__ __host__ __inline__ cmplx iMul(const double &a)
{
	return make_cuDoubleComplex(0, a);
}

