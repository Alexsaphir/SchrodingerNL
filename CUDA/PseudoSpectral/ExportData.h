#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "Axis.h"
#include "Signal.cuh"
#include "SignalFFT.cuh"

void exportData(const Axis * X, const Signal * S, const std::string & name);
void exportData(const Axis * X, SignalFFT * S, const std::string & name);
void writeInFile(const Axis * X, Signal * S, int fileX);
