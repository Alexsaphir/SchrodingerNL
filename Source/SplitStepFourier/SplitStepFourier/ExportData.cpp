#include "ExportData.h"

void exportData(Grid* const S, const std::string &name)
{
	Axis X = S->getAxis();
	S->syncDeviceToHost();

	std::ofstream file;
	file.open(name);
	file << "a" << std::endl;
	for (int i = 0; i < X.getN(); ++i)
		file << X.getValueAt(i) << " " << std::setprecision(5) << S->getHostData()[i].x << " " << S->getHostData()[i].y << "\n";//(S->getHostData()[i].x<0?-1.:1)*
	file.close();
}

void exportCsvXY(Grid * const S, const std::string & name)
{
	Axis X = S->getAxis();
	S->syncDeviceToHost();

	std::ofstream file;
	file.open(name);
	file << "\"x\",\"y\",\"scalar\"" << std::endl;
	for (int i = 0; i < X.getN(); ++i)
		file << std::fixed << std::setprecision(5) << X.getValueAt(i) << "," << cuCabs(S->getHostData()[i]) << "," << cuCabs(S->getHostData()[i]) << "\n";//(S->getHostData()[i].x<0?-1.:1)*
	file.close();
}

void exportCsvXTY(Grid * const S, const std::string & name, int iter)
{
	Axis X = S->getAxis();
	S->syncDeviceToHost();

	std::ofstream file;
	if (iter == 0)
	{
		//erase file
		file.open(name, std::ios::out | std::ios::trunc);
		file << "\"x\",\"y\",\"z\",\"scalar\"" << std::endl;
	}
	else
	{
		file.open(name, std::ios::out | std::ios::app);
	}

	for (int i = 0; i < X.getN(); ++i)
		file << std::fixed << std::setprecision(5) << X.getValueAt(i) << "," << iter << "," << cuCabs(S->getHostData()[i]) << "," << cuCabs(S->getHostData()[i]) << "\n";//(S->getHostData()[i].x<0?-1.:1)*
	file.close();
}

void exportCsv2DMatlab(Grid * const S, const std::string & name, int iter)
{
	Axis X = S->getAxis();
	S->syncDeviceToHost();

	std::ofstream file;
	if (iter == 0)
	{
		//erase file
		file.open(name, std::ios::out | std::ios::trunc);
	}
	else
	{
		file.open(name, std::ios::out | std::ios::app);
	}

	for (int i = 0; i < X.getN(); ++i)
	{
		file << std::fixed << std::setprecision(5) << cuCabs(S->getHostData()[i]);
		if (i == X.getN() - 1)
		{
			file << "\n";
		}
		else
		{
			file << ", ";
		}
	}
	file.close();
}

void exportCsvComplexMatlab(Grid * const S, const std::string & name)
{
	Axis X = S->getAxis();
	S->syncDeviceToHost();

	std::ofstream file;
	
	//erase file
	file.open(name, std::ios::out | std::ios::trunc);

	for (int i = 0; i < X.getN(); ++i)
	{
		file << std::fixed << X.getValueAt(i) << ", " << std::setprecision(5) << S->getHostData()[i].x << ", " << S->getHostData()[i].y;
		file << "\n";
	}
	file.close();
}

void exportMassOverTime(double E, const std::string & name, int iter)
{

	std::ofstream file;
	if (iter == 0)
	{
		//erase file
		file.open(name, std::ios::out | std::ios::trunc);
	}
	else
	{
		file.open(name, std::ios::out | std::ios::app);
	}
	
	file << std::fixed << std::setprecision(17) << E << "\n";

	file.close();
}
