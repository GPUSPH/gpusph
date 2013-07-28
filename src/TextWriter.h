/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

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
#ifndef H_TEXTWRITER_H
#define H_TEXTWRITER_H

#include "Writer.h"

using namespace std;

class TextWriter : public Writer
{
public:
	TextWriter(const Problem *problem);
	~TextWriter();


	void write(uint numParts, const float4 *pos, const float4 *vel,
		const particleinfo *info, const float3 *vort, float t, const bool testpoints, const float4 *normals, const float4 *gradGamma = 0,
		const float *tke = 0, const float *turbvisc = 0);
};

#endif
