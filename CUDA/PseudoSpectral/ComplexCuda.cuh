#if !defined COMPLEXCUDA_H
#define COMPLEXCUDA_H

#include <cuComplex.h>

#define cmplx cuDoubleComplex

//Operator overloading for cmplx

__host__ __device__ cuDoubleComplex cuCexp(cuDoubleComplex z);

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