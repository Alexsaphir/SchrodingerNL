#include "ExportData.h"



void exportData(const Axis *X, const Signal *S, const std::string &name)
{
	std::cout << std::fixed;
	std::ofstream file;
	file.open(name);
	file << "a" << std::endl;
	for (int i = 0; i < S->getSignalPoints(); ++i)
		file << X->getLinearValueAt(i) << " " << std::setprecision(5) << S->getHostData()[i].x << " " << S->getHostData()[i].y << "\n";//(S->getHostData()[i].x<0?-1.:1)*
	file.close();
}

void exportData(const Axis *X, SignalFFT *S, const std::string &name)
{//Export data of the host
	std::ofstream file;
	file.open(name);
	file << "pa" << std::endl;
	for (int i = 0; i < S->getSignalPoints(); ++i)
		file << X->getFrequency(i) << " " << std::setprecision(5) << cuCabs(S->getHostData()[i]) << " " << 0 << "\n";
	file.close();
}

void writeInFile(const Axis *X, Signal *S, int fileX)
{
	cmplx *V = S->getHostData();
	std::ofstream file;

	file.open("GNUPLOT/data" + std::to_string(fileX) + ".ds");
	for (int i = 0; i < S->getSignalPoints(); ++i)
		file << X->getLinearValueAt(i) << " " << std::setprecision(5) << (V[i].x<0 ? -1. : 1.)*sqrt(V[i].x*V[i].x + V[i].y*V[i].y) << "\n";
	file.close();

}

