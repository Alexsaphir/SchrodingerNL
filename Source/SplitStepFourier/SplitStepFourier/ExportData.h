#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "Axis.h"
#include "Grid.cuh"
#include "Grid2D.cuh"


void exportData(Grid* const S, const std::string &name);

void exportCsvXY(Grid* const S, const std::string &name);

void exportCsvXTY(Grid * const S, const std::string & name, int iter);

void exportCsv2DMatlab(Grid * const S, const std::string & name, int iter);

void exportCsvComplexMatlab(Grid * const S, const std::string & name);

void exportMassOverTime(double E, const std::string & name, int iter);

void exportCsv2DMatlab(Grid2D * const S, const std::string & name, int iter);