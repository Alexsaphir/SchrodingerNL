#if !defined COMPLEXCUDA_H
#define COMPLEXCUDA_H

#include <cmath>

#include <cuComplex.h>

#define cmplx cuDoubleComplex
#define M_PI 3.14159265358979323846

//Operator overloading for cmplx

__host__ __device__ cuDoubleComplex cuCexp(cuDoubleComplex z);

__device__ __host__ cuDoubleComplex cuCpow(cuDoubleComplex z, double n);

__host__ __device__ cuDoubleComplex cuCexp(double z);

__device__ __host__ cmplx operator+(const cmplx &a, const cmplx &b);

__device__ __host__ cmplx operator+(const double &a, const cmplx &b);

__device__ __host__ cmplx operator+(const cmplx  &a, const double &b);

__device__ __host__ cmplx operator-(const cmplx &a, const cmplx &b);

__device__ __host__ cmplx operator-(const double &a, const cmplx &b);

__device__ __host__ cmplx operator-(const cmplx &a, const double &b);

__device__ __host__ cmplx operator-(const cmplx &a);

__device__ __host__ cmplx operator*(const cmplx &a, const cmplx &b);

__device__ __host__ cmplx operator*(const double &a, const cmplx &b);

__device__ __host__ cmplx operator*(const cmplx &a, const double &b);

__device__ __host__ cmplx operator/(const cmplx &a, const cmplx &b);

__device__ __host__ cmplx operator/(const cmplx &a, const double &b);

__device__ __host__ cmplx operator/(const double &a, const cmplx &b);

__device__ __host__ cmplx iMul(const cmplx &a);

__device__ __host__ cmplx iMul(const double &a);




#endif