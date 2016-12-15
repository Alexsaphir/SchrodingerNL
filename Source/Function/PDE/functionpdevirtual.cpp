#include "functionpdevirtual.h"

FunctionPDEVirtual::FunctionPDEVirtual(GridManagerBase *dmb): FunctionVirtual(), m_gridManager(dmb)
{
}

FunctionPDEVirtual::FunctionPDEVirtual(const FunctionPDEVirtual &F): FunctionVirtual(), m_gridManager(F.m_gridManager)
{
}

FunctionPDEVirtual::~FunctionPDEVirtual()
{
}
