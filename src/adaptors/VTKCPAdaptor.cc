/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

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


#include "VTKCPAdaptor.h"

#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkCPPythonScriptPipeline.h>
#include <vtkNew.h>

vtkCPProcessor* processor = NULL;

VTKCPAdaptor::VTKCPAdaptor()
{
}


VTKCPAdaptor::~VTKCPAdaptor()
{
}

void VTKCPAdaptor::Initialize(const char* script_path)
{
	// Create co-processor
	if(processor == NULL) {
		processor = vtkCPProcessor::New();
		processor->Initialize();
	} else {
		processor->RemoveAllPipelines();
	}

	// Add pipeline
	vtkNew<vtkCPPythonScriptPipeline> pipeline;
	pipeline->Initialize(script_path);
	processor->AddPipeline(pipeline.GetPointer());
}

void VTKCPAdaptor::Finalize()
{
	// Delete co-procesor
	if(processor) {
		processor->Finalize();
		processor->Delete();
		processor = NULL;
	}
}

void VTKCPAdaptor::CoProcess(vtkDataObject* data, double time, unsigned int time_step)
{
	// Create data description
	vtkNew<vtkCPDataDescription> dataDescription;
	dataDescription->AddInput("input");
	dataDescription->SetTimeData(time, time_step);

	// Set grid
	dataDescription->GetInputDescriptionByName("input")->SetGrid(data);

	// Process the data
	processor->CoProcess(dataDescription.GetPointer());
}
