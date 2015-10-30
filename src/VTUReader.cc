#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <stdexcept>

#include <limits.h> // UINT_MAX

#include "VTUReader.h"

using namespace std;

int
VTUReader::getNParts()
{
	pugi::xml_document vtuFile;
	ostringstream err_msg;

	if (!vtuFile.load_file(filename.c_str())) {
		// open binary file for reading to check if we have an appended data xml entry
		ifstream binaryFile(filename.c_str(), ifstream::in | ifstream::binary);
		if (!binaryFile.is_open()) {
			err_msg << "Cannot open " <<  filename.c_str() << "!\n";
			throw runtime_error(err_msg.str());
		}
		// search for <AppendedData
		stringstream bufferss;
		bufferss << binaryFile.rdbuf();
		string buffer(bufferss.str());
		size_t startData = buffer.find("<AppendedData");
		if (startData == string::npos) {
			err_msg << "FATAL: Could not find <AppendedData in file " << filename << "!\n";
			throw runtime_error(err_msg.str());
		}
		// now look for the closing ">" of the <AppendData .... > node. This is where the data will start
		startData = buffer.find(">", startData);
		if (startData == string::npos) {
			err_msg << "FATAL: Could not find closing > of AppendedData node in file " << filename << "!\n";
			throw runtime_error(err_msg.str());
		}
		// the data starts after the ">" so increase count by 1
		startData++;
		// the data will end at the last occurance of </AppendData so look for it
		size_t endData = buffer.rfind("</AppendedData");
		if (endData == string::npos && startData < endData) {
			err_msg << "FATAL: Could not identify end of data in file " << filename << "!\n";
			throw runtime_error(err_msg.str());
		}
		// remove the binary data in the original file to get a valid xml file in the buffer string
		buffer.replace(startData, endData-startData, "");
		// load that xml file with pugixml
		if(!vtuFile.load_string(buffer.c_str(), buffer.size())) {
			err_msg << "FATAL: Cannot open " << filename << " using the xml parser even after removing binary data.\n";
			throw runtime_error(err_msg.str());
		}
		binaryFile.close();
	}

	pugi::xml_node vtkFile = vtuFile.child("VTKFile");
	if (!vtkFile) {
		err_msg << "FATAL: " << filename << " is not a valid vtk file\n";
		throw runtime_error(err_msg.str());
	}

	pugi::xml_node uGrid = vtkFile.child("UnstructuredGrid");
	if (!uGrid) {
		err_msg << "FATAL: VTK reader cannot find a UnstructuredGrid node in file " << filename << "!\n";
		throw runtime_error(err_msg.str());
	}

	pugi::xml_node piece = uGrid.child("Piece");
	if (!piece) {
		err_msg << "FATAL: VTK reader cannot find a piece node in file " << filename << "!\n";
		throw runtime_error(err_msg.str());
	}
	if (piece.attribute("NumberOfPoints"))
		npart = piece.attribute("NumberOfPoints").as_int();
	else {
		err_msg << "FATAL: VTK reader cannot determine number of particles in file " << filename << "!\n";
		throw runtime_error(err_msg.str());
	}

	return npart;
}

void
VTUReader::read()
{
	// read npart if it was yet uninitialized
	if (npart == UINT_MAX)
		getNParts();

	std::cout << "Reading particle data from the input: " << filename << std::endl;

	// allocating read buffer
	if(buf == NULL)
		buf = new ReadParticles[npart];
	else{
		delete [] buf;
		buf = new ReadParticles[npart];
	}

	pugi::xml_document vtuFile;
	string binaryData;
	ostringstream err_msg;
	// big endian encoding
	bool bigEndian = true;

	if (!vtuFile.load_file(filename.c_str())) {
		// open binary file for reading to check if we have an appended data xml entry
		ifstream binaryFile(filename.c_str(), ifstream::in | ifstream::binary);
		if (!binaryFile.is_open()) {
			err_msg << "Cannot open " <<  filename.c_str() << "!\n";
			throw runtime_error(err_msg.str());
		}
		// search for <AppendedData
		stringstream bufferss;
		bufferss << binaryFile.rdbuf();
		string buffer(bufferss.str());
		size_t startData = buffer.find("<AppendedData");
		if (startData == string::npos) {
			err_msg << "FATAL: Could not find <AppendedData in file " << filename << "!\n";
			throw runtime_error(err_msg.str());
		}
		// now look for the closing ">" of the <AppendData .... > node. This is where the data will start
		startData = buffer.find(">", startData);
		if (startData == string::npos) {
			err_msg << "FATAL: Could not find closing > of AppendedData node in file " << filename << "!\n";
			throw runtime_error(err_msg.str());
		}
		// the data starts after the ">" so increase count by 1
		startData++;
		// the data will end at the last occurance of </AppendData so look for it
		size_t endData = buffer.rfind("</AppendedData");
		if (endData == string::npos && startData < endData) {
			err_msg << "FATAL: Could not identify end of data in file " << filename << "!\n";
			throw runtime_error(err_msg.str());
		}
		// put the binary data into the binaryData string for access later on
		binaryData = buffer.substr(startData, endData-startData);
		// remove the binary data in the original file to get a valid xml file in the buffer string
		buffer.replace(startData, endData-startData, "");
		// load that xml file with pugixml
		if(!vtuFile.load_string(buffer.c_str(), buffer.size())) {
			err_msg << "FATAL: Cannot open " << filename << " using the xml parser even after removing binary data.\n";
			throw runtime_error(err_msg.str());
		}
		binaryFile.close();
	}

	pugi::xml_node vtkFile = vtuFile.child("VTKFile");
	if (!vtkFile) {
		err_msg << "FATAL: " << filename << " is not a valid vtk file\n";
		throw runtime_error(err_msg.str());
	}

	// get byte order
	if (vtkFile.attribute("byte_order")) {
		if (!strcmp(vtkFile.attribute("byte_order").value(), "LittleEndian")) bigEndian = false;
	}
	// Determine byte order on local machine
	union {
		uint i;
		char c[sizeof(uint)];
	} bint = {0x01020304};
	const bool localBigEndian = (bint.c[0] == 1);
	// if the byte order on the machine is different from the one in the file make sure we swap later on
	const bool swapRequired = bigEndian != localBigEndian;

	// Determine header int size
	uint sizeofHeader = 4;
	// this attribute only exists in vtk version 1.0
	// by default it's uint32 (also in vtk version 0.1)
	if (vtkFile.attribute("header_type")) {
		if (!strcmp(vtkFile.attribute("header_type").value(), "UInt64")) sizeofHeader = 8;
	}

	pugi::xml_node uGrid = vtkFile.child("UnstructuredGrid");
	if (!uGrid) {
		err_msg << "FATAL: VTK reader cannot find a UnstructuredGrid node in file " << filename << "!\n";
		throw runtime_error(err_msg.str());
	}

	pugi::xml_node piece = uGrid.child("Piece");
	if (!piece) {
		err_msg << "FATAL: VTK reader cannot find a piece node in file " << filename << "!\n";
		throw runtime_error(err_msg.str());
	}

	pugi::xml_node pointData = piece.child("PointData");
	if (!pointData) {
		err_msg << "FATAL: VTK reader cannot find a PointData node in file " << filename << "!\n";
		throw runtime_error(err_msg.str());
	}

	uint counter = 0;
	void *data0 = NULL;
	void *data1 = NULL;
	void *data2 = NULL;

	for (pugi::xml_node da = pointData.first_child(); da; da = da.next_sibling()) {
		bool doRead = true;
		data0 = NULL;
		data1 = NULL;
		data2 = NULL;
		// set the data0, data1, data2 pointers to the respective struct member in the buf array
		if (!strcmp(da.attribute("Name").value(), "Volume")){
			counter++;
			data0 = &buf[0].Volume;
		}
		else if (!strcmp(da.attribute("Name").value(), "Surface")){
			counter++;
			data0 = &buf[0].Surface;
		}
		else if (!strcmp(da.attribute("Name").value(), "ParticleType")){
			counter++;
			data0 = &buf[0].ParticleType;
		}
		else if (!strcmp(da.attribute("Name").value(), "FluidType")){
			counter++;
			data0 = &buf[0].FluidType;
		}
		else if (!strcmp(da.attribute("Name").value(), "KENT")){
			counter++;
			data0 = &buf[0].KENT;
		}
		else if (!strcmp(da.attribute("Name").value(), "MovingBoundary")){
			counter++;
			data0 = &buf[0].MovingBoundary;
		}
		else if (!strcmp(da.attribute("Name").value(), "AbsoluteIndex")){
			counter++;
			data0 = &buf[0].AbsoluteIndex;
		}
		else if (!strcmp(da.attribute("Name").value(), "Normal")){
			counter++;
			data0 = &buf[0].Normal_0;
			data1 = &buf[0].Normal_1;
			data2 = &buf[0].Normal_2;
		}
		else if (!strcmp(da.attribute("Name").value(), "VertexParticle")){
			counter++;
			data0 = &buf[0].VertexParticle1;
			data1 = &buf[0].VertexParticle2;
			data2 = &buf[0].VertexParticle3;
		}
		else {
			cout << "VTK reader: Unknown name: " << da.attribute("Name").value() << " in DataArray in PointData node" << endl;
			doRead = false;
		}

		if (doRead)
			readData(da, data0, data1, data2, swapRequired, sizeofHeader, binaryData, vtkFile);
	}

	if (counter != 9) {
		err_msg << "FATAL: VTK reader could not find all required data fields in file " << filename << "!\n";
		throw runtime_error(err_msg.str());
	}

	pugi::xml_node points = piece.child("Points");
	if (!points) {
		err_msg << "FATAL: VTK reader cannot find a Points node in file " << filename << "!\n";
		throw runtime_error(err_msg.str());
	}

	pugi::xml_node da = points.child("DataArray");
	if (!da) {
		err_msg << "FATAL: VTK reader cannot find a DataArray node inside the Points node in file " << filename << "!\n";
		throw runtime_error(err_msg.str());
	}

	data0 = (void*) &buf[0].Coords_0;
	data1 = (void*) &buf[0].Coords_1;
	data2 = (void*) &buf[0].Coords_2;

	readData(da, data0, data1, data2, swapRequired, sizeofHeader, binaryData, vtkFile);

	return;
}

void VTUReader::readData(	pugi::xml_node	da,
							void*			data0,
							void*			data1,
							void*			data2,
							bool			swapRequired,
							uint			sizeofHeader,
							string			&binaryData,
							pugi::xml_node	vtkFile)
{
	ostringstream err_msg;
	uint numberOfComponents = 1;
	if (da.attribute("NumberOfComponents"))
		numberOfComponents = da.attribute("NumberOfComponents").as_int();

	// check whether the appropriate number of 
	switch (numberOfComponents) {
	case 1:
		if(!data0) {
			err_msg << "FATAL: VTK reader void array not initialized correctly for 1 component with variable " << da.attribute("Name") << "!\n";
			throw runtime_error(err_msg.str());
		}
		break;
	case 3:
		if(!data0 || !data1 || !data2) {
			err_msg << "FATAL: VTK reader void array not initialized correctly for 3 components with variable " << da.attribute("Name") << "!\n";
			throw runtime_error(err_msg.str());
		}
		break;
	default:
		err_msg << "FATAL: VTK reader found array with " << numberOfComponents << " components at variable " << da.attribute("Name") << "!\n";
		throw runtime_error(err_msg.str());
	}

	string format = "ascii";
	if (da.attribute("format")) {
		format = da.attribute("format").value();
		if (format.compare("ascii") && format.compare("binary") && format.compare("appended")) {
			printf("Fatal: VTK reader discovered unkown format: %s in file %s\n", format.c_str(), filename.c_str());
		}
	}

	string type = "";
	uint sizeofData = 0;
	if (da.attribute("type")) {
		string tmp = da.attribute("type").value();
		if (tmp.find("Float") != string::npos)
			type = "float";
		else if (tmp.find("Int") != string::npos)
			type = "int";
		else {
			err_msg << "FATAL: VTK reader found array with unkown type " << tmp << " at variable " << da.attribute("Name") << "!\n";
			throw runtime_error(err_msg.str());
		}
		if (tmp.find("32") != string::npos) {
			sizeofData = 4;
		} else if (tmp.find("64") != string::npos) {
			sizeofData = 8;
		} else {
			err_msg << "FATAL: VTK reader found array with unkown size " << tmp << " at variable " << da.attribute("Name") << "!\n";
			throw runtime_error(err_msg.str());
		}
	} else {
		err_msg << "FATAL: VTK reader found array without type at variable " << da.attribute("Name") << "!\n";
		throw runtime_error(err_msg.str());
	}

	if (format == "ascii") {
		if (type == "int")
			readAsciiData<int>(	da,
								data0,
								data1,
								data2,
								numberOfComponents,
								swapRequired,
								sizeofHeader,
								sizeofData);
		else
			readAsciiData<double>(	da,
									data0,
									data1,
									data2,
									numberOfComponents,
									swapRequired,
									sizeofHeader,
									sizeofData);

	} else if (format == "binary") {
		if (type == "int")
			readBinaryData<int>(da,
								data0,
								data1,
								data2,
								numberOfComponents,
								swapRequired,
								sizeofHeader,
								sizeofData);
		else
			readBinaryData<double>(	da,
									data0,
									data1,
									data2,
									numberOfComponents,
									swapRequired,
									sizeofHeader,
									sizeofData);

	} else if (format == "appended") {
		if (type == "int") {
			if (sizeofData == 4)
				readAppendedData<int32_t, int>(	da,
												data0,
												data1,
												data2,
												numberOfComponents,
												swapRequired,
												sizeofHeader,
												binaryData,
												vtkFile);
			else
				readAppendedData<int64_t, int>(	da,
												data0,
												data1,
												data2,
												numberOfComponents,
												swapRequired,
												sizeofHeader,
												binaryData,
												vtkFile);
		} else {
			if (sizeofData == 4)
				readAppendedData<float, double>(da,
												data0,
												data1,
												data2,
												numberOfComponents,
												swapRequired,
												sizeofHeader,
												binaryData,
												vtkFile);
			else
				readAppendedData<double, double>(da,
												data0,
												data1,
												data2,
												numberOfComponents,
												swapRequired,
												sizeofHeader,
												binaryData,
												vtkFile);
		}
	}
}

template<typename T>
void
VTUReader::readAsciiData(	pugi::xml_node	da,
							void			*data0,
							void			*data1,
							void			*data2,
							uint			numberOfComponents,
							bool			swapRequired,
							uint			sizeofHeader,
							uint			sizeofData)
{
}

template<typename T>
void
VTUReader::readBinaryData(	pugi::xml_node	da,
							void			*data0,
							void			*data1,
							void			*data2,
							uint			numberOfComponents,
							bool			swapRequired,
							uint			sizeofHeader,
							uint			sizeofData)
{
}

template<typename IN, typename OUT>
void
VTUReader::readAppendedData(pugi::xml_node	da,
							void			*data0,
							void			*data1,
							void			*data2,
							uint			numberOfComponents,
							bool			swapRequired,
							uint			sizeofHeader,
							string			&binaryData,
							pugi::xml_node	vtkFile)
{
	uint offset = 0;
	bool dataEncoded = true;
	ostringstream err_msg;

	pugi::xml_node appData = vtkFile.child("AppendedData");
	if (!appData) {
		err_msg << "Fatal: VTK reader cannot find a AppendedData child in the VTKFile node in file " << filename;
	}

	if (appData.attribute("encoding")) {
		if (!strcmp(appData.attribute("encoding").value(), "raw")) dataEncoded = false;
	}

	if (da.attribute("offset")) {
		offset = da.attribute("offset").as_int();
	}

	// this is the char array full of data starting with a header then a _ and then binary data possibly encoded in base64
	string dataIncHead;
	if (dataEncoded) dataIncHead = appData.text().get();
	else dataIncHead = binaryData;
	string::iterator dataIt = dataIncHead.begin();
	// skip header
	while ((*dataIt != '_') && dataIt < dataIncHead.end()) dataIt++;
	// skip the _ as well
	dataIt++;
	// byte array that contains the decoded data
	vector<BYTE> data;
	// text data is base64 encoded
	if (dataEncoded) {
		// new data array without header
		string encodedData = string(dataIt, dataIncHead.end());
		// convert to byte array
		data = base64_decode(encodedData);
	// data not base64 encoded: can be read raw
	} else {
		// write string into data vector
		while (dataIt < dataIncHead.end()) {
			data.push_back((BYTE) *dataIt);
			dataIt++;
		}
	}

	readBinaryVtkData<IN, OUT> (	data,
									data0,
									data1,
									data2,
									numberOfComponents,
									sizeofHeader,
									offset,
									swapRequired);
}

template<typename IN, typename OUT>
void VTUReader::readBinaryVtkData (	vector<BYTE>	&data,
									void			*data0,
									void			*data1,
									void			*data2,
									uint			numberOfComponents,
									uint			sizeofHeader,
									uint			offset,
									bool			swapRequired)
{
	// dataI is the iterator that tells us where in the data we are
	uint dataI = offset;
	// dataSize is the size of the entry that we are going to read, it is determined in the first sizeofHeader bytes
	int64_t dataSize = 0;
	// read header for 64 bit integer
	if (sizeofHeader == 8) {
		BYTE *iC = (BYTE*) &dataSize;
		if (swapRequired) {
			for (uint ii=0; ii<sizeofHeader; ii++) iC[ii] = data[dataI+sizeofHeader-1-ii];
		} else {
			for (uint ii=0; ii<sizeofHeader; ii++) iC[ii] = data[dataI+ii];
		}
	// read header for 32 bit integer
	} else {
		int32_t tmp;
		BYTE *iC = (BYTE*) &tmp;
		if (swapRequired) {
			for (uint ii=0; ii<sizeofHeader; ii++) iC[ii] = data[dataI+sizeofHeader-1-ii];
		} else {
			for (uint ii=0; ii<sizeofHeader; ii++) iC[ii] = data[dataI+ii];
		}
		dataSize = (int64_t) tmp;
	}
	dataI += sizeofHeader;
	dataSize /= sizeof(IN);
	uint pointsRead = 0;
	while (pointsRead < dataSize) {
		OUT dvalue;
		IN tmp;
		BYTE *dvalueC = (BYTE*) &tmp;
		if (swapRequired) {
			for (uint ii=0; ii<sizeof(IN); ii++) dvalueC[ii] = data[dataI+sizeof(IN)-1-ii];
		} else {
			for (uint ii=0; ii<sizeof(IN); ii++) dvalueC[ii] = data[dataI+ii];
		}
		dataI += sizeof(IN);
		dvalue = (OUT) tmp;

		if (numberOfComponents == 3) {
			if (pointsRead%3 == 0)
				*((OUT*)((char*)data0 + pointsRead/3*sizeof(ReadParticles))) = dvalue;
			else if (pointsRead%3 == 1)
				*((OUT*)((char*)data1 + pointsRead/3*sizeof(ReadParticles))) = dvalue;
			else if (pointsRead%3 == 2)
				*((OUT*)((char*)data2 + pointsRead/3*sizeof(ReadParticles))) = dvalue;
		} else
			*((OUT*)((char*)data0 + pointsRead*sizeof(ReadParticles))) = dvalue;
		pointsRead++;
	}
}
