/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

#ifndef _VTKCPADAPTOR_H
#define _VTKCPADAPTOR_H

class vtkDataObject;

/*!
 * This class is responsible for Paraview Catalyst library interfacing.
 * It creates, stores and manipulates the instance of vtkCPProcessor.
 */
class VTKCPAdaptor
{
public:
	/*!
	 * Constructor.
	 */
	VTKCPAdaptor();

	/*!
	 * Destructor.
	 */
	~VTKCPAdaptor();

	/*!
	 * Initializes vtkCPProcessor instance and adds the pipeline.
	 * \param[in] script_path path to the pipeline Python script
	 */
	void Initialize(const char* script_path);

	/*!
	 * Releases all resources used by Catalyst.
	 */
	void Finalize();

	/*!
	 * Executes the pipeline.
	 * \param[in] data object representing the grids and fields
	 * \param[in] time current simulation time
	 * \param[in] time_step current simulation step
	 */
	void CoProcess(vtkDataObject* data, double time, unsigned int time_step);
};

#endif /* _VTKCPADAPTOR_H */
