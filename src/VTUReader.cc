#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>

#include <limits.h> // UINT_MAX

#include "VTUReader.h"
#include "pugixml.h"

using namespace std;

int
VTUReader::getNParts()
{
	pugi::xml_document vtuFile;
	string binaryData;

	if (!vtuFile.load_file(filename.c_str())) {
		// open binary file for reading to check if we have an appended data xml entry
		ifstream binaryFile(filename.c_str(), ifstream::in | ifstream::binary);
		if (!binaryFile.is_open())
			cerr << "Cannot open " <<  filename.c_str() << "!\n";
		// search for <AppendedData
		stringstream bufferss;
		bufferss << binaryFile.rdbuf();
		string buffer(bufferss.str());
		size_t startData = buffer.find("<AppendedData");
		if (startData == string::npos)
			cerr << "FATAL: Could not find <AppendedData in file " << filename << "!\n";
		// now look for the closing ">" of the <AppendData .... > node. This is where the data will start
		startData = buffer.find(">", startData);
		if (startData == string::npos)
			cerr << "FATAL: Could not find closing > of AppendedData node in file " << filename << "!\n";
		// the data starts after the ">" so increase count by 1
		startData++;
		// the data will end at the last occurance of </AppendData so look for it
		size_t endData = buffer.rfind("</AppendedData");
		if (endData == string::npos && startData < endData)
			cerr << "FATAL: Could not identify end of data in file " << filename << "!\n";
		// put the binary data into the binaryData string for access later on
		binaryData = buffer.substr(startData, endData-startData);
		// remove the binary data in the original file to get a valid xml file in the buffer string
		buffer.replace(startData, endData-startData, "");
		// load that xml file with pugixml
		if(!vtuFile.load_string(buffer.c_str(), buffer.size()))
			cerr << "FATAL: Cannot open " << filename << " using the xml parser even after removing binary data.\n";
		binaryFile.close();
	}

	pugi::xml_node vtkFile = vtuFile.child("VTKFile");
	if (!vtkFile)
		cerr << "FATAL: " << filename << " is not a valid vtk file\n";

	pugi::xml_node uGrid = vtkFile.child("UnstructuredGrid");
	if (!uGrid)
		cerr << "FATAL: VTK reader cannot find a UnstructuredGrid node in file " << filename << "!\n";

	pugi::xml_node piece = uGrid.child("Piece");
	if (!piece)
		cerr << "FATAL: VTK reader cannot find a piece node in file " << filename << "!\n";
	if (piece.attribute("NumberOfPoints"))
		npart = piece.attribute("NumberOfPoints").as_int();
	else
		cerr << "FATAL: VTK reader cannot determine number of particles in file " << filename << "!\n";

	return npart;
}

void
VTUReader::read()
{
}
