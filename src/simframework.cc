/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#include "simframework.h"

SimFramework::SimFramework() :
	m_neibsEngine(NULL),
	m_integrationEngine(NULL),
	m_viscEngine(NULL),
	m_forcesEngine(NULL),
	m_bcEngine(NULL),
	m_allocPolicy(NULL),
	m_filterEngines(),
	m_filterFreqList(),
	m_postProcessEngines(),
	m_simparams(NULL)
{}

SimFramework::~SimFramework()
{
	delete m_simparams;

	m_postProcessEngines.clear();
	m_filterFreqList.clear();
	m_filterEngines.clear();

	delete m_allocPolicy;
	delete m_bcEngine;
	delete m_forcesEngine;
	delete m_viscEngine;
	delete m_integrationEngine;
	delete m_neibsEngine;
}

// Filters are run at the beginning of each iteration whose number is an exact
// multiple of the filter frequency. We also want to ensure that filters are run
// in order (from lowest-valued type to highest-valued type). Therefore, whenever a new
// filter is added, we update the map of FilterType to actual FilterEngine, as well as
// the ordered list of (filter, frequency) pairs.

AbstractFilterEngine* SimFramework::addFilterEngine(FilterType filtertype, int frequency)
{
	AbstractFilterEngine *flt = NULL;

	FilterEngineSet::iterator found(m_filterEngines.find(filtertype));
	if (found == m_filterEngines.end()) {
		flt = newFilterEngine(filtertype, frequency);
		m_filterEngines[filtertype] = flt;
		// insert the (filter, freq) pair before any higher-valued filters
		FilterFreqList::iterator place(m_filterFreqList.begin());
		FilterFreqList::iterator ffend(m_filterFreqList.end());
		while (place != ffend && place->first < filtertype)
			++place;
		if (place != ffend)
			--place;
		m_filterFreqList.insert(place, std::make_pair(filtertype, frequency));

	} else {
		flt = found->second;
		// update m_filterFreqList
		FilterFreqList::iterator place(m_filterFreqList.begin());
		while (place->first != filtertype)
			++place;
		place->second = frequency;
	}
	flt->set_frequency(frequency);
	return flt;
}

AbstractPostProcessEngine* SimFramework::addPostProcessEngine(PostProcessType pptype, flag_t options)
{
	AbstractPostProcessEngine *flt = NULL;

	PostProcessEngineSet::iterator found(m_postProcessEngines.find(pptype));
	if (found == m_postProcessEngines.end()) {
		m_postProcessEngines[pptype] = newPostProcessEngine(pptype, options);
	} else {
		std::cerr << "WARNING: tried to re-add post-process filter " <<
			PostProcessName[pptype < INVALID_POSTPROC ? pptype : INVALID_POSTPROC] <<
			" (" << pptype << "), skipped!" << std::endl;
	}
	return flt;
}

AbstractPostProcessEngine* SimFramework::hasPostProcessEngine(PostProcessType pptype) const
{
	PostProcessEngineSet::const_iterator found(m_postProcessEngines.find(pptype));
	return found == m_postProcessEngines.end() ? NULL : found->second;
}

flag_t SimFramework::hasPostProcessOption(PostProcessType pptype, flag_t option) const
{
	PostProcessEngineSet::const_iterator found(m_postProcessEngines.find(pptype));
	return found == m_postProcessEngines.end() ? NO_FLAGS :
		(found->second->get_options() & option);
}
