#ifndef ROWDATAPROXY_H
#define ROWDATAPROXY_H

#include "../../../Domain/domain.h"
#include "../../corematrix.h"
#include "../../../type.h"

class RowDataProxy: public CoreMatrix
{
public:
	RowDataProxy();
	RowDataProxy(Domain *D);

	void setDomain(Domain *D);

	//Specific method for Row or Column Matrix
	virtual cmplx at(uint i) const;
	virtual void set(uint i, const cmplx &value);
	//Generic method
	virtual cmplx getValue(uint i, uint j) const = 0;
	virtual void setValue(uint i, uint j, const cmplx &value) = 0;

	virtual uint row() const;
	virtual uint column() const;


	~RowDataProxy();
private:
	Domain *m_domain;
};

#endif // ROWDATAPROXY_H
