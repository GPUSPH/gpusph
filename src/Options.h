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

#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include <cmath>
#include <string>
#include <sstream> // for de-serialization of option values
#include <map> // unordered_map would be faster, but it's C++11

// arbitrary problem options are allowed, stored in
// a string -> string map, and deserialized on retrieval.
typedef std::map<std::string, std::string> OptionMap;

class Options {
private:
	// Storage for arbitrary options
	// TODO convert legacy options to new mechanism
	OptionMap m_options;

public:
	// legacy options
	std::string	problem; // problem name
	std::string	resume_fname; // file to resume simulation from
	int		device;  // which device to use
	std::string	dem; // DEM file to use
	std::string	dir; // directory where data will be saved
	double	deltap; // deltap
	float	tend; // simulation end
	float	checkpoint_freq; // frequency of hotstart checkpoints (in simulated seconds)
	int	checkpoints; // number of hotstart checkpoints to keep
	bool	nosave; // disable saving
	bool	gpudirect; // enable GPUDirect
	bool	striping; // enable striping (i.e. compute/transfer overlap)
	bool	asyncNetworkTransfers; // enable asynchronous network transfers
	unsigned int num_hosts; // number of physical hosts to which the processes are being assigned
	bool byslot_scheduling; // by slot scheduling across MPI nodes (not round robin)

	Options(void) :
		m_options(),
		problem(),
		resume_fname(),
		device(-1),
		dem(),
		dir(),
		deltap(NAN),
		checkpoint_freq(NAN),
		checkpoints(-1),
		tend(NAN),
		nosave(false),
		gpudirect(false),
		striping(false),
		asyncNetworkTransfers(false),
		num_hosts(0),
		byslot_scheduling(false)
	{};

	// set an arbitrary option
	// TODO templatize for serialization?
	void
	set(std::string const& key, std::string const& value)
	{
		// TODO set legacy options from here too?
		m_options[key] = value;
	}

	template<typename T> T
	get(std::string const& key, T const& _default) const
	{
		T ret(_default);
		OptionMap::const_iterator found(m_options.find(key));
		if (found != m_options.end()) {
			std::istringstream extractor(found->second);
			extractor >> ret;
		}
		return ret;
	}

	OptionMap::const_iterator
	begin() const
	{ return m_options.begin(); }

	OptionMap::const_iterator
	end() const
	{ return m_options.end(); }
};

// Declare custom specializations which otherwise wouldn't be known to
// Options users
template<>
std::string
Options::get(std::string const& key, std::string const& _default) const;

template<>
bool
Options::get(std::string const& key, bool const& _default) const;


#endif
