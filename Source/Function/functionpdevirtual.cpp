#include "functionpdevirtual.h"

FunctionPDEVirtual::FunctionPDEVirtual(): m_domainManager(NULL)
{
}

FunctionPDEVirtual::FunctionPDEVirtual(DomainManagerBase *dmb): m_domainManager(dmb)
{
}

FunctionPDEVirtual::FunctionPDEVirtual(const FunctionPDEVirtual &F): m_domainManager(F.m_domainManager)
{
}

FunctionPDEVirtual::~FunctionPDEVirtual()
{
}
