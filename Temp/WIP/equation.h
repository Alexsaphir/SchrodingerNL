#ifndef EQUATION_H
#define EQUATION_H

#include <complex>

template<class Type> class Solver1D;

template<class Type> class Equation
{
public:
	Equation();

	//U		: (i,t) --> U(i,t)
	//U_t	: value of U at the point wanted at the time t
	//U_l_t	: value of U before U_t at time t
	//U_r_t	: value of U after U_t at time t
	//s=t-1


	static Type Heat1D(Solver1D<Type> const *S, int i);// Return Value for the time t+1
	static Type Schrodinger1D(Solver1D<Type> const *S, int i);
	static Type Schrodinger1DT0(Solver1D<Type> const *S, int i);
};



template<class Type> Equation<Type>::Equation()
{

}

template<class Type> Type Equation<Type>::Heat1D(const Solver1D<Type> *S, int i)
{
	double r(S->alpha*S->getDt()/(S->getDx()*S->getDx()));
	double r2(1.-2.*r);
	return r*S->getValue(i-1)+r2*S->getValue(i)+r*S->getValue(i+1);
}

template<class Type>  Type Equation<Type>::Schrodinger1D(const Solver1D<Type> *S, int j)
{
	std::complex<double> i(0.,1.);
	double two(2.);
	double three(3.);
	double four(4.);
	double five(5.);

	std::complex<double> r0(two*i*S->getDt()/(S->getDx()*S->getDx()));
	std::complex<double> r1(two*i*S->getDt());

	std::complex<double> res(0.);
	res= i*S->getOldValue(j)+r1*(std::pow(std::abs(S->getValue(j)),2.)-std::pow(std::abs(S->getValue(j)),4.))*S->getValue(j);

	//if(j==0)
	//	res+=r0*(two*S->getValue(j)-five*S->getValue(j+1)+four*S->getValue(j+2)-S->getValue(j+3));
	//else if(j==(S->getN()-1))
	//	res+=r0*(two*S->getValue(j)-five*S->getValue(j-1)+four*S->getValue(j-2)-S->getValue(j-3));
	//else
		res+=r0*(S->getValue(j+1)-two*S->getValue(j)+S->getValue(j-1));

	return res;
}

template<class Type>  Type Equation<Type>::Schrodinger1DT0(const Solver1D<Type> *S, int j)
{
	std::complex<double> i(0.,1.);

	std::complex<double> r0(i*S->getDt()/(S->getDx()*S->getDx()));
	std::complex<double> r1(i*S->getDt());

	std::complex<double> res(0.);

	double two(2.);
	double three(3.);
	double four(4.);
	double five(5.);
	res= i*S->getValue(j)+r1*(std::pow(std::abs(S->getValue(j)),2.)-std::pow(std::abs(S->getValue(j)),4.))*S->getValue(j);

	//if(j==0)
	//	res+=r0*(two*S->getValue(j)-five*S->getValue(j+1)+four*S->getValue(j+2)-S->getValue(j+3));
	//else if(j==(S->getN()-1))
	//	res+=r0*(two*S->getValue(j)-five*S->getValue(j-1)+four*S->getValue(j-2)-S->getValue(j-3));
	//else
		res+=r0*(S->getValue(j+1)-two*S->getValue(j)+S->getValue(j-1));

	return res;
}


#endif // EQUATION_H
