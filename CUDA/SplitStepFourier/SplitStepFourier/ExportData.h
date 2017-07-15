#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "Axis.h"
#include "Grid.cuh"


void exportData(Grid* const S, const std::string &name);

void exportCsvXY(Grid* const S, const std::string &name);

void exportCsvXTY(Grid * const S, const std::string & name, int iter);

void exportCsv2DMatlab(Grid * const S, const std::string & name, int iter);

void exportMassOverTime(double E, const std::string & name, int iter);