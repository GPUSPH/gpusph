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

/*! \file
 * Definitions and classes to handle command-line options
 */

#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include <cmath>
#include <string>
#include <sstream> // for de-serialization of option values
#include <vector>
#include <map> // unordered_map would be faster, but it's C++11

/*! arbitrary problem options are allowed, stored in
 * a string -> string map, and deserialized on retrieval.
 */
typedef std::map<std::string, std::string> OptionMap;

//! parse a string in the form number[,number...] and return a vector of integers
std::vector<int> parse_devices_string(const char *);

//! return the default devices
/*! This parses the environment variable GPUSPH_DEVICE as a comma-separated list
 *  of device numbers (if available), and otherwise returns just the device 0
 */
std::vector<int> get_default_devices();

//! The Options class is used to manage command-line options
class Options {
private:
	//! Storage for arbitrary options
	//! TODO convert legacy options to new mechanism
	OptionMap m_options;

public:

	//! legacy options
	//! @{
	std::string	problem; ///< problem name
	std::string	resume_fname; ///< file to resume simulation from
	std::vector<int> devices; ///< list of devices to be used
	std::string	dem; ///< DEM file to use
	std::string	dir; ///< directory where data will be saved
	double	deltap; ///< deltap
	float	tend; ///< simulation end
	float	dt; ///< fixed time-step
	float	ccsph_min_det; ///< minimum determinant for CCSPH
	unsigned long maxiter; ///< maximum number of iterations to run
	unsigned int repack_maxiter; ///< maximum number of iterations for repacking
	float	checkpoint_freq; ///< frequency of hotstart checkpoints (in simulated seconds)
	int		checkpoints; ///< number of hotstart checkpoints to keep
	bool	nosave; ///< disable saving
	bool	pin_fea_buffers; ///< pin host-side global buffers for FEA
	bool	gpudirect; ///< enable GPUDirect
	bool	striping; ///< enable striping (i.e. compute/transfer overlap)
	bool	asyncNetworkTransfers; ///< enable asynchronous network transfers
	unsigned int num_hosts; ///< number of physical hosts to which the processes are being assigned
	bool byslot_scheduling; ///< by slot scheduling across MPI nodes (not round robin)
	bool no_leak_warning; ///< if true, do not warn if #parts decreased in simulations without outlets
	bool visualization; ///< if true - live visualization via DisplayWriter will be enabled
	double visu_freq; ///< visualization frequency
	std::string pipeline_fpath; ///< path to visualization pipeline file
	bool repack; ///< if true, run the repacking before the simulation
	bool repack_only; ///< if true, run the repacking only and quit
	std::string repack_fname; ///< repack file to resume simulation from
	//! @}

	Options(void) :
		m_options(),
		problem(),
		resume_fname(),
		devices(),
		dem(),
		dir(),
		deltap(NAN),
		tend(NAN),
		dt(NAN),
		ccsph_min_det(NAN),
		maxiter(0),
		repack_maxiter(0),
		checkpoint_freq(NAN),
		checkpoints(-1),
		pin_fea_buffers(true),
		nosave(false),
		gpudirect(false),
		striping(false),
		asyncNetworkTransfers(false),
		num_hosts(0),
		byslot_scheduling(false),
		no_leak_warning(false),
		visualization(false),
		visu_freq(NAN),
		pipeline_fpath(),
		repack(false),
		repack_only(false),
		repack_fname()
	{};

	//! set an arbitrary option
	//! TODO templatize for serialization?
	void
	set(std::string const& key, std::string const& value)
	{
		// TODO set legacy options from here too?
		m_options[key] = value;
	}

	//! get the value of an option, providing default if option is not set
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

//! Declare custom specializations which otherwise wouldn't be known to
//! Options users
//! @{

//! get() overload for string type
template<>
std::string
Options::get(std::string const& key, std::string const& _default) const;

//! get() overload for boolean type
template<>
bool
Options::get(std::string const& key, bool const& _default) const;

//! @}


#endif
