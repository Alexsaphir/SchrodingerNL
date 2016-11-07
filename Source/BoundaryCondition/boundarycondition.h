#ifndef BOUNDARYCONDITION_H
#define BOUNDARYCONDITION_H


class BoundaryCondition
{
public:
	BoundaryCondition();
};

enum TypeOfBoundaryCondition
{
	Dirichlet,
	Neumann
};
//Dirichlet		y(Boundary)=alpha()
//Neumann		y'(Boundary)=alpha()

#endif // BOUNDARYCONDITION_H
