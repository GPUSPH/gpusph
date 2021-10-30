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

#include "DisplayWriter.h"

#include "GlobalData.h"

#include "VTKCPAdaptor.h"

#include <vtkNew.h>
#include <vtkCellType.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnsignedShortArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>

#include "ProblemCore.h"

static float zeroes[4];

DisplayWriter::DisplayWriter(const GlobalData *_gdata)
	: Writer(_gdata)
{
	// Create the adaptor
	m_adaptor = new VTKCPAdaptor();

	// Initialize the adaptor with Python pipeline path (got from execution options)
	m_adaptor->Initialize(_gdata->clOptions->pipeline_fpath.c_str());
}


DisplayWriter::~DisplayWriter()
{
	m_adaptor->Finalize();
}

void
DisplayWriter::write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints)
{
	// Build VTK grid
	vtkSmartPointer<vtkUnstructuredGrid> vtkGrid = buildGrid(numParts, buffers, node_offset);

	// Co-process
	m_adaptor->CoProcess(vtkGrid, t, gdata->iterations);
}

vtkSmartPointer<vtkUnstructuredGrid>
DisplayWriter::buildGrid(uint numParts, BufferList const& buffers, uint node_offset)
{
	const double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();
	const hashKey *particleHash = buffers.getData<BUFFER_HASH>();
	const float4 *vel = buffers.getData<BUFFER_VEL>();
	const float4 *vol = buffers.getData<BUFFER_VOLUME>();
	const float *sigma = buffers.getData<BUFFER_SIGMA>();
	const particleinfo *info = buffers.getData<BUFFER_INFO>();
	const float3 *vort = buffers.getData<BUFFER_VORTICITY>();
	const float4 *normals = buffers.getData<BUFFER_NORMALS>();
	const float4 *gradGamma = buffers.getData<BUFFER_GRADGAMMA>();
	const float *tke = buffers.getData<BUFFER_TKE>();
	const float *eps = buffers.getData<BUFFER_EPSILON>();
	const float *turbvisc = buffers.getData<BUFFER_TURBVISC>();
	const float *spsturbvisc = buffers.getData<BUFFER_SPS_TURBVISC>();
	const float4 *eulervel = buffers.getData<BUFFER_EULERVEL>();
	const float *priv = buffers.getData<BUFFER_PRIVATE>();
	const vertexinfo *vertices = buffers.getData<BUFFER_VERTICES>();
	const float *intEnergy = buffers.getData<BUFFER_INTERNAL_ENERGY>();
	const float4 *forces = buffers.getData<BUFFER_FORCES>();

	const neibdata *neibslist = buffers.getData<BUFFER_NEIBSLIST>();

	// Create vtkUnstructuredGrid
	vtkSmartPointer<vtkUnstructuredGrid> vtkGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();

	// Create the points information
	vtkNew<vtkPoints> points;
	points->SetNumberOfPoints(numParts);

	for (uint i = node_offset; i < node_offset + numParts; i++) {
		points->SetPoint(i - node_offset, pos[i].x, pos[i].y, pos[i].z);
	}

	vtkGrid->SetPoints(points.GetPointer());

	// Create the cells information#include <vtkUnsignedShortArray.h>
	vtkGrid->Allocate(numParts);
	for (uint i = 0; i < numParts; ++i) {
		vtkIdType ids[1] = {static_cast<vtkIdType>(i)};
		vtkGrid->InsertNextCell(VTK_VERTEX, 1, ids);
	}

	// Internal Energy
	if (intEnergy) {
		vtkNew<vtkFloatArray> intEnergyArray;
		intEnergyArray->SetName("Internal Energy");
		intEnergyArray->SetArray((float*)intEnergy, numParts, 1);

		vtkGrid->GetPointData()->AddArray(intEnergyArray.GetPointer());
	}

	// Spatial acceleration and Continuity derivative
	if (forces) {
		// Spatial acceleration
		vtkNew<vtkFloatArray> spatialAccArray;
		spatialAccArray->SetName("Spatial accelerationy");
		spatialAccArray->SetNumberOfComponents(3);
		spatialAccArray->SetNumberOfTuples(numParts);

	for (uint i = node_offset; i < node_offset + numParts; i++) {
		const float *value = (float*)(forces + i);
		spatialAccArray->SetTypedTuple(i - node_offset, value);
	}

	vtkGrid->GetPointData()->AddArray(spatialAccArray.GetPointer());

	// Continuity derivative
	vtkNew<vtkFloatArray> continuityArray;
	continuityArray->SetName("Continuity derivative");
	continuityArray->SetNumberOfValues(numParts);

	for (uint i = node_offset; i < node_offset + numParts; i++) {
		const float value = forces[i].w;
		continuityArray->SetValue(i - node_offset, value);
	}

	vtkGrid->GetPointData()->AddArray(continuityArray.GetPointer());
	}

	// Pressure
	vtkNew<vtkFloatArray> pressureArray;
	pressureArray->SetName("Pressure");
	pressureArray->SetNumberOfValues(numParts);

	for (uint i = node_offset; i < node_offset + numParts; i++) {
		float value = 0.0;
		if (TESTPOINT(info[i]))
			value = vel[i].w;
		else
			value = m_problem->pressure(vel[i].w, fluid_num(info[i]));

		pressureArray->SetValue(i - node_offset, value);
	}

	vtkGrid->GetPointData()->AddArray(pressureArray.GetPointer());

	// Density
	vtkNew<vtkFloatArray> densityArray;
	densityArray->SetName("Density");
	densityArray->SetNumberOfValues(numParts);

	for (uint i = node_offset; i < node_offset + numParts; i++) {
		float value = 0.0;
		if (TESTPOINT(info[i]))
			// TODO FIXME: Testpoints compute pressure only
			// In the future we would like to have a#include <vtkUnsignedShortArray.h> density here
			// but this needs to be done correctly for multifluids
			value = NAN;
		else
			value = vel[i].w;

		densityArray->SetValue(i - node_offset, value);
	}

	vtkGrid->GetPointData()->AddArray(densityArray.GetPointer());

	// Mass
	vtkNew<vtkDoubleArray> massArray;
	massArray->SetName("Mass");
	massArray->SetNumberOfValues(numParts);

	for (uint i = node_offset; i < node_offset + numParts; i++) {
		float value = pos[i].w;
		massArray->SetValue(i - node_offset, value);
	}

	vtkGrid->GetPointData()->AddArray(massArray.GetPointer());

	// Gamma
	if (gradGamma) {
		vtkNew<vtkFloatArray> gammaArray;
		gammaArray->SetName("Gamma");
		gammaArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float value = gradGamma[i].w;
			gammaArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(gammaArray.GetPointer());
	}

	// Turbulent kinetic energy
	if (tke) {
		vtkNew<vtkFloatArray> tkeArray;
		tkeArray->SetName("TKE");
		tkeArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float value = tke[i];
			tkeArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(tkeArray.GetPointer());
	}

	// Turbulent epsilon
	if (eps) {
		vtkNew<vtkFloatArray> epsArray;
		epsArray->SetName("Epsilon");
		epsArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float value = eps[i];
			epsArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(epsArray.GetPointer());
	}

	// Eddy viscosity
	if (turbvisc) {
		vtkNew<vtkFloatArray> turbviscArray;
		turbviscArray->SetName("Eddy viscosity");
		turbviscArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float value = turbvisc[i];
			turbviscArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(turbviscArray.GetPointer());
	}

	// SPS turbulent viscosity
	if (spsturbvisc) {
		vtkNew<vtkFloatArray> spsturbviscArray;
		spsturbviscArray->SetName("SPS turbulent viscosity");
		spsturbviscArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float value = spsturbvisc[i];
			spsturbviscArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(spsturbviscArray.GetPointer());
	}

	// Particle info
	if (info) {
		// type + flags
		vtkNew<vtkUnsignedShortArray> typeArray;
		typeArray->SetName("Part type+flags");
		typeArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			ushort value = type(info[i]);
			typeArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(typeArray.GetPointer());

		// fluid number (is only included if there are more than 1)
		const bool write_fluid_num = (gdata->problem->physparams()->numFluids() > 1);

		if (write_fluid_num) {
			vtkNew<vtkUnsignedCharArray> fluidNumberArray;
			fluidNumberArray->SetName("Fluid number");
			fluidNumberArray->SetNumberOfValues(numParts);

			for (uint i = node_offset; i < node_offset + numParts; i++) {
			uchar value = fluid_num(info[i]);
			fluidNumberArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(fluidNumberArray.GetPointer());
	}

	// object number (is only included if there are any)
	// TODO a better way would be for GPUSPH to expose the highest
	// object number ever associated with any particle, so that we
	// could check that
	const bool write_part_obj = (gdata->problem->simparams()->numbodies > 0);

	if (write_part_obj) {
		vtkNew<vtkUnsignedShortArray> objectNumberArray;
		objectNumberArray->SetName("Part object");
		objectNumberArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			ushort value = object(info[i]);
			objectNumberArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(objectNumberArray.GetPointer());
	}

	// part id
	vtkNew<vtkUnsignedIntArray> partIdArray;
	partIdArray->SetName("Part id");
	partIdArray->SetNumberOfValues(numParts);

	for (uint i = node_offset; i < node_offset + numParts; i++) {
		uint value = id(info[i]);
		partIdArray->SetValue(i - node_offset, value);
	}

	vtkGrid->GetPointData()->AddArray(partIdArray.GetPointer());
	}

	// Vertices
	if (vertices) {
		vtkNew<vtkUnsignedIntArray> verticesArray;
		verticesArray->SetName("Vertices");
		verticesArray->SetNumberOfComponents(4);
		verticesArray->SetNumberOfTuples(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			const uint *value = (uint*)(vertices + i);
			verticesArray->SetTypedTuple(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(verticesArray.GetPointer());
	}

	// Linearized cell index (NOTE: particles might be slightly off the belonging cell)
	vtkNew<vtkUnsignedIntArray> cellIndexArray;
	cellIndexArray->SetName("CellIndex");
	cellIndexArray->SetNumberOfValues(numParts);

	for (uint i = node_offset; i < node_offset + numParts; i++) {
		uint value = cellHashFromParticleHash( particleHash[i] );
		cellIndexArray->SetValue(i - node_offset, value);
	}

	vtkGrid->GetPointData()->AddArray(cellIndexArray.GetPointer());

	// Velocity
	vtkNew<vtkFloatArray> velocityArray;
	velocityArray->SetName("Velocity");
	velocityArray->SetNumberOfComponents(3);
	velocityArray->SetNumberOfTuples(numParts);

	for (uint i = node_offset; i < node_offset + numParts; i++) {
		float *value = zeroes;
		value = (float*)(vel + i);
		velocityArray->SetTypedTuple(i - node_offset, value);
	}

	vtkGrid->GetPointData()->AddArray(velocityArray.GetPointer());

	// Eulerian velocity and density
	if (eulervel) {
		// Eulerian velocity
		vtkNew<vtkFloatArray> eulerVelocityArray;
		eulerVelocityArray->SetName("Eulerian velocity");
		eulerVelocityArray->SetNumberOfComponents(3);
		eulerVelocityArray->SetNumberOfTuples(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			value = (float*)(eulervel + i);
			eulerVelocityArray->SetTypedTuple(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(eulerVelocityArray.GetPointer());

		// Eulerian density
		vtkNew<vtkFloatArray> eulerDensityArray;
		eulerDensityArray->SetName("Eulerian density");
		eulerDensityArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float value = eulervel[i].w;
			eulerDensityArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(eulerDensityArray.GetPointer());
	}

	// Gradient gamma
	if (gradGamma) {
		vtkNew<vtkFloatArray> gradGammaArray;
		gradGammaArray->SetName("Gradient Gamma");
		gradGammaArray->SetNumberOfComponents(3);
		gradGammaArray->SetNumberOfTuples(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			value = (float*)(gradGamma + i);
			gradGammaArray->SetTypedTuple(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(gradGammaArray.GetPointer());
	}

	// Vorticity
	if (vort) {
		vtkNew<vtkFloatArray> vorticityArray;
		vorticityArray->SetName("Vorticity");
		vorticityArray->SetNumberOfComponents(3);
		vorticityArray->SetNumberOfTuples(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			if (FLUID(info[i])) {
				value = (float*)(vort + i);
			}
			vorticityArray->SetTypedTuple(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(vorticityArray.GetPointer());
	}

	// Normals and criteria
	if (normals) {
		// normals
		vtkNew<vtkFloatArray> normalsArray;
		normalsArray->SetName("Normals");
		normalsArray->SetNumberOfComponents(3);
		normalsArray->SetNumberOfTuples(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			if (FLUID(info[i])) {
				value = (float*)(normals + i);
			}
			normalsArray->SetTypedTuple(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(normalsArray.GetPointer());

		// criteria
		vtkNew<vtkFloatArray> criteriaArray;
		criteriaArray->SetName("Criteria");
		criteriaArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float value = 0;
			if (FLUID(info[i])) {
				value = normals[i].w;
			}
			criteriaArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(criteriaArray.GetPointer());
	}

	// Private
	if (priv) {
		vtkNew<vtkFloatArray> privateArray;
		privateArray->SetName("Private");
		privateArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float value = priv[i];
			privateArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(privateArray.GetPointer());
	}

	// Volume
	if (vol) {
		vtkNew<vtkFloatArray> volumeArray;
		volumeArray->SetName("Volume");
		volumeArray->SetNumberOfComponents(4);
		volumeArray->SetNumberOfTuples(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float *value = (float*)(vol + i);
			volumeArray->SetTypedTuple(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(volumeArray.GetPointer());
	}

	// Sigma
	if (sigma) {
		vtkNew<vtkFloatArray> sigmaArray;
		sigmaArray->SetName("Sigma");
		sigmaArray->SetNumberOfValues(numParts);

		for (uint i = node_offset; i < node_offset + numParts; i++) {
			float value = sigma[i];
			sigmaArray->SetValue(i - node_offset, value);
		}

		vtkGrid->GetPointData()->AddArray(sigmaArray.GetPointer());
	}

	// Reclaim unused memory
	vtkGrid->Squeeze();

	return vtkGrid;
}
