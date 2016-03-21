#ifndef _VTUREADER_H
#define _VTUREADER_H

#include <string>
#include <iostream>

using namespace std;

#include "Reader.h"
#include "pugixml.h"
#include "base64.h"

class VTUReader : public Reader{
public:
	// returns the number of particles in the vtu file
	int getNParts(void);

	// allocates the buffer and reads the data from the vtu file
	void read(void);

	// read Data array from a node in a vtk file
	void readData(	pugi::xml_node	da,
					void*			data0,
					void*			data1,
					void*			data2,
					bool			swapRequired,
					uint			sizeofHeader,
					string			&binaryData,
					pugi::xml_node	vtkFile);

	// read ascii data from a node in a vtk file
	template<typename T>
	void readAsciiData(	pugi::xml_node	da,
						void			*data0,
						void			*data1,
						void			*data2,
						uint			numberOfComponents,
						bool			swapRequired,
						uint			sizeofHeader,
						uint			sizeofData);

	// read binary data from a node in a vtk file
	template<typename T>
	void readBinaryData(pugi::xml_node	da,
						void			*data0,
						void			*data1,
						void			*data2,
						uint			numberOfComponents,
						bool			swapRequired,
						uint			sizeofHeader,
						uint			sizeofData);

	// read appended data from a node in a vtk file
	template<typename IN, typename OUT>
	void readAppendedData(	pugi::xml_node	da,
							void			*data0,
							void			*data1,
							void			*data2,
							uint			numberOfComponents,
							bool			swapRequired,
							uint			sizeofHeader,
							string			&binaryData,
							pugi::xml_node	vtkFile);

template<typename IN, typename OUT>
void readBinaryVtkData (vector<BYTE>	&data,
						void			*data0,
						void			*data1,
						void			*data2,
						uint			numberOfComponents,
						uint			sizeofHeader,
						uint			offset,
						bool			swapRequired);

};

#endif
