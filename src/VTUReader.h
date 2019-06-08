/*  Copyright (c) 2015-2018 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _VTUREADER_H
#define _VTUREADER_H

#include <string>
#include <iostream>

#include "Reader.h"
#include "pugixml.h"
#include "base64.h"
#include "common_types.h"

class VTUReader : public Reader
{
public:
	// returns the number of particles in the vtu file
	size_t getNParts(void) override;

	// allocates the buffer and reads the data from the vtu file
	void read(void) override;

	// read Data array from a node in a vtk file
	void readData(	pugi::xml_node	da,
					void*			data0,
					void*			data1,
					void*			data2,
					bool			swapRequired,
					uint			sizeofHeader,
					std::string		&binaryData,
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
							std::string		&binaryData,
							pugi::xml_node	vtkFile);

template<typename IN, typename OUT>
void readBinaryVtkData (std::vector<BYTE>	&data,
						void			*data0,
						void			*data1,
						void			*data2,
						uint			numberOfComponents,
						uint			sizeofHeader,
						uint			offset,
						bool			swapRequired);

};

#endif
