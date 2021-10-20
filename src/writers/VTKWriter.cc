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

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <functional>

#include "VTKWriter.h"
// GlobalData is required for writing the device index. With some order
// of inclusions, a forward declaration might be required
#include "GlobalData.h"
#include "ProblemCore.h"
#include "simparams.h"

#include "vector_print.h"

#include "neibs_list_layout.h"

// for FLT_EPSILON
#include <cfloat>

/// TODO FIXME these should go in a more general place
#include "cuda/tensor.h"
template<>
struct vector_traits<symtensor3>
{
	typedef float component_type;
	enum { components = 6 };
};

using namespace std;

template<typename T>
const char *vtk_type_name(T const*);

template<>
const char *vtk_type_name<uchar>(uchar const*)
{ return "UInt8"; }
template<>
const char *vtk_type_name<ushort>(ushort const*)
{ return "UInt16"; }
template<>
const char *vtk_type_name<uint>(uint const*)
{ return "UInt32"; }
template<>
const char *vtk_type_name<float>(float const*)
{ return "Float32"; }
template<>
const char *vtk_type_name<double>(double const*)
{ return "Float64"; }

// TODO for the time being, we assume no more than 256 devices
// upgrade to UInt16 / ushort if it's ever needed

typedef unsigned char dev_idx_t;

VTKWriter::VTKWriter(const GlobalData *_gdata)
  : Writer(_gdata),
	m_planes_fname(),
	m_blockidx(-1),
	m_neiblist_stride(gdata->allocatedParticles),
	m_neiblist_size(gdata->problem->simparams()->neiblistsize),
	m_neiblist_end(m_neiblist_stride*m_neiblist_size),
	m_neib_bound_pos(gdata->problem->simparams()->neibboundpos)
{
	m_fname_sfx = ".vtp";

	string time_fname = open_data_file(m_timefile, "VTUinp", "", ".pvd");

	// Writing header of VTUinp.pvd file
	if (m_timefile) {
		m_timefile << "<?xml version='1.0'?>\n";
		m_timefile << "<VTKFile type='Collection' version='0.1'>\n";
		m_timefile << " <Collection>\n";
	}
}


VTKWriter::~VTKWriter()
{
	mark_timefile();
	m_timefile.close();
}

void VTKWriter::add_block(string const& blockname, string const& fname)
{
	++m_blockidx;
	m_timefile << "  <DataSet timestep='" << m_current_time << "' group='" << m_blockidx <<
		"' name='" << blockname << "' file='" << fname << "'/>" << endl;
}

void VTKWriter::start_writing(double t, WriteFlags const& write_flags)
{
	Writer::start_writing(t, write_flags);

	ostringstream time_repr;
	time_repr << setprecision(16) << t;
	m_current_time = time_repr.str();

	const int step_num = write_flags.step.number;

	// we append the current integrator step to the timestring,
	// but we need to add a dot if there isn't one already
	string dot = m_current_time.find('.') != string::npos ? "" : ".";
	if (step_num > 0) {
		char buf[10];
		snprintf(buf, 10, "%08u", step_num);
		buf[9] = '\0';
		m_current_time += dot + string(buf);
	}

	m_blockidx = -1;

	const bool has_planes = gdata->s_hPlanes.size() > 0;

	if (has_planes) {
		if (m_planes_fname.size() == 0) {
			save_planes();
		}
		add_block("Planes", m_planes_fname);
	}

	if (gdata->problem->get_dem()) {
		if (m_dem_fname.size() == 0) {
			save_dem();
		}
		add_block("Topography", m_dem_fname);
	}
}

void VTKWriter::mark_written(double t)
{
	mark_timefile();

	Writer::mark_written(t);
}

/* Endianness check: (char*)&endian_int reads the first byte of the int,
 * which is 0 on big-endian machines, and 1 in little-endian machines */
static int endian_int=1;
static const char* endianness[2] = { "BigEndian", "LittleEndian" };

/* auxiliary functions to write data array entrypoints */
inline void
scalar_array_header(ofstream &out, const char *type, const char *name, size_t offset)
{
	out << "	<DataArray type='" << type << "' Name='" << name
		<< "' format='appended' offset='" << offset << "'/>" << endl;
}

inline void
vector_array_header(ofstream &out, const char *type, const char *name, uint dim, size_t offset)
{
	out << "	<DataArray type='" << type << "' Name='" << name
		<< "' NumberOfComponents='" << dim
		<< "' format='appended' offset='" << offset << "'/>" << endl;
}

// A structure to manage appending data at the end of a VTK file
struct VTKAppender
{
	ofstream &out;
	particleinfo const* info;
	GlobalData const* gdata;
	size_t node_offset;
	size_t numParts;
	size_t data_offset;

	vector<function<void(void)>> data_filler;

	VTKAppender(
		ofstream& _out,
		particleinfo const* _info,
		GlobalData const* _gdata,
		size_t _node_offset,
		size_t _numParts)
	:
		out(_out),
		info(_info),
		gdata(_gdata),
		node_offset(_node_offset),
		numParts(_numParts),
		data_offset(0)
	{}

private:

	/// Create the metadata for array data, named name
	template<typename T>
	inline void
	array_header(T const* data, const char *name)
	{
		using traits = vector_traits<T>;
		using S = typename traits::component_type;
		using Sptr = S const*;

		/* Non-vector types have components == 0, but for us they count as having 1 component */
		constexpr auto N0 = vector_traits<T>::components;
		constexpr auto N = N0 > 0 ? N0 : 1;

		if (N == 1) {
			scalar_array_header(out, vtk_type_name(data), name, data_offset);
		} else {
			Sptr dummy(nullptr);
			vector_array_header(out, vtk_type_name(dummy), name, N, data_offset);
		}
		data_offset += N*sizeof(S)*numParts + sizeof(uint);
	}

	/// Create the metadata for a split array, with a name for the xyz part,
	/// and a name for the w part
	template< typename T >
	inline
	enable_if_t<vector_traits<T>::components == 4>
	array_header(T const* data, const char *name_xyz, const char *name_w)
	{
		using traits = vector_traits<T>;
		using S = typename traits::component_type;
		using Sptr = S const*;
		Sptr dummy(nullptr);

		if (name_xyz) {
			vector_array_header(out, vtk_type_name(dummy), name_xyz, 3, data_offset);
			data_offset += 3*sizeof(S)*numParts + sizeof(uint);
		}
		if (name_w) {
			scalar_array_header(out, vtk_type_name(dummy), name_w, data_offset);
			data_offset += sizeof(S)*numParts + sizeof(uint);
		}
	}

	/// Binary dump of (part of) a variable
	template<typename T,
		typename traits = vector_traits<T>,
		typename S = typename traits::component_type,
		size_t N0 = traits::components,
		size_t N = (N0 > 0 ? N0 : 1)
		>
	inline void
	write_var(T const& var, size_t components = N)
	{
		out.write(reinterpret_cast<const char *>(&var), sizeof(S)*components);
	}

	/// Binary dump of an array of nels variables
	template<typename T>
	inline void
	write_array(T const *var, size_t nels)
	{
		out.write(reinterpret_cast<const char *>(var), sizeof(T)*nels);
	}

public:
	/// Write appended data for VTK, without any transformation
	/*! The array is assumed to be local to the node, and node_offset will not
	 * be considered
	 */
	template<typename T>
	inline void
	append_local_data(T const* data, const char *name)
	{
		array_header(data, name);

		// Push back a lambda that does the actual data storage
		data_filler.push_back([this, data]() {
			uint numbytes = sizeof(T)*numParts;
			write_var(numbytes);
			write_array(data, numParts);
		});
	}

	template<typename T, typename Ret>
	using DataTransformFull = Ret (*)(T const&, particleinfo const&, GlobalData const*);
	template<typename T, typename Ret>
	using DataTransformInfo = Ret (*)(T const&, particleinfo const&);
	template<typename T, typename Ret>
	using DataTransform = Ret (*)(T const&);

	/// Write appended data for VTK, transforming an array of T into an array of Ret
	/*! This version is specialized for a transform that takes as input the data,
	 * the particle info for the corresponding element (including the node_offset),
	 * and the GlobalData object.
	 */
	template<typename T, typename Ret>
	inline void
	append_local_data(T const* data, const char *name, DataTransformFull<T, Ret> func)
	{
		Ret *dummy(nullptr);
		array_header(dummy, name);

		data_filler.push_back( [this, data, func]() {
			uint numbytes = sizeof(Ret)*numParts;
			write_var(numbytes);
			for (size_t i = 0; i < numParts; ++i) {
				Ret value = func(data[i], info[i + node_offset], gdata);
				write_var(value);
			}
		});
	}

	/// Write appended data for VTK, transforming an array of T into an array of Ret
	/*! This version is specialized for a transform that takes as input the data,
	 * and the particle info for the corresponding element (including the node_offset).
	 */
	template<typename T, typename Ret>
	inline void
	append_local_data(T const* data, const char *name, DataTransformInfo<T, Ret> func)
	{
		Ret *dummy(nullptr);
		array_header(dummy, name);

		data_filler.push_back( [this, data, func]() {
			uint numbytes = sizeof(Ret)*numParts;
			write_var(numbytes);
			for (size_t i = 0; i < numParts; ++i) {
				Ret value = func(data[i], info[i + node_offset]);
				write_var(value);
			}
		});
	}

	/// Write appended data for VTK, transforming an array of T into an array of Ret
	/*! This version is specialized for a transform that takes as input the data alone.
	 */
	template<typename T, typename Ret>
	inline void
	append_local_data(T const* data, const char *name, DataTransform<T, Ret> func)
	{
		Ret *dummy(nullptr);
		array_header(dummy, name);

		data_filler.push_back( [this, data, func]() {
			uint numbytes = sizeof(Ret)*numParts;
			write_var(numbytes);
			for (size_t i = 0; i < numParts; ++i) {
				Ret value = func(data[i]);
				write_var(value);
			}
		});
	}

	/// Write appended data for VTK, mapping the index to some arbitrary value
	template<typename IndexTransform,
		typename Ret = typename result_of<IndexTransform(size_t)>::type>
	inline void
	append_local_data(const char *name, IndexTransform func)
	{
		Ret *dummy = nullptr;
		array_header(dummy, name);

		data_filler.push_back( [this, func]() {
			uint numbytes = sizeof(Ret)*numParts;
			write_var(numbytes);
			for (size_t i = 0; i < numParts; ++i) {
				Ret value = func(i);
				write_var(value);
			}
		});
	}

	/// Write a (local) split array to a VTK.
	/*! No transformation can be applied in this case.
	 */
	template<typename T>
	inline
	enable_if_t<vector_traits<T>::components == 4>
	append_local_data(T const* data, const char *name_xyz, const char *name_w)
	{
		using S = typename vector_traits<T>::component_type ;

		array_header(data, name_xyz, name_w);

		data_filler.push_back( [this, data, name_xyz, name_w]() {
			if (name_xyz) {
				uint numbytes = 3*sizeof(S)*numParts;
				write_var(numbytes);
				for (size_t i = 0; i < numParts; ++i)
					write_var(data[i], 3);
			}
			if (name_w) {
				uint numbytes = sizeof(S)*numParts;
				write_var(numbytes);
				for (size_t i = 0; i < numParts; ++i)
					write_var(data[i].w);
			}
		});
	}

	/// Write appended data for VTK, applying node_offset
	template<typename T, typename ...Args>
	inline void
	append_data(T const* var, Args...args)
	{
		append_local_data(var + node_offset, args...);
	}

	void write_appended_data(void)
	{
		for (auto& func : data_filler)
			func();
	}
};

/// Binary dump of (part of) a tensor
template<>
inline void
VTKAppender::write_var<symtensor3>(symtensor3 const& var, size_t components)
{
	out.write(reinterpret_cast<const char *>(&var.xx), sizeof(float));
	out.write(reinterpret_cast<const char *>(&var.yy), sizeof(float));
	out.write(reinterpret_cast<const char *>(&var.zz), sizeof(float));
	out.write(reinterpret_cast<const char *>(&var.xy), sizeof(float));
	out.write(reinterpret_cast<const char *>(&var.xz), sizeof(float));
	out.write(reinterpret_cast<const char *>(&var.yz), sizeof(float));
}



float get_pressure(float4 const& pvel, particleinfo const& pinfo, GlobalData const* gdata)
{
	return TESTPOINT(pinfo) ? pvel.w : gdata->problem->pressure(pvel.w, fluid_num(pinfo));
}

// Here we store the physical density to visualize
float get_density(float4 const& pvel, particleinfo const& pinfo, GlobalData const* gdata)
{
	// TODO FIXME: Testpoints compute pressure only
	// In the future we would like to have a density here but this needs to be
	// done correctly for multifluids
	return TESTPOINT(pinfo) ? NAN : gdata->problem->physical_density(pvel.w,fluid_num(pinfo)) ;
}

// Return the last component of a float4 
float get_last_component(float4 const& pvel )
{ return pvel.w; }

// Return the last component of a double4, demoted to a float
float demote_w(double4 const& data)
{ return data.w; }

symtensor3 fetch_tensor(float2 const* const* data, size_t index)
{
	float2 const* tau0 = data[0];
	float2 const* tau1 = data[1];
	float2 const* tau2 = data[2];
	symtensor3 ret;
	ret.xx = tau0[index].x;
	ret.xy = tau0[index].y;
	ret.xz = tau1[index].x;
	ret.yy = tau1[index].y;
	ret.yz = tau2[index].x;
	ret.zz = tau2[index].y;
	return ret;
}

uchar get_part_type(particleinfo const& pinfo)
{ return PART_TYPE(pinfo); }

ushort get_part_flags(particleinfo const& pinfo)
{ return PART_FLAGS(pinfo) >> PART_FLAG_SHIFT; }

uchar get_fluid_num(particleinfo const& pinfo)
{ return fluid_num(pinfo); }

ushort get_object(particleinfo const& pinfo)
{ return object(pinfo); }

uchar get_object_few(particleinfo const& pinfo)
{ return object(pinfo); }

uint get_cellindex(hashKey const& phash)
{ return cellHashFromParticleHash(phash); }

// to debug the neighbors list
static char3 cell_to_offset[27] = {};

//! Names of the particle types
// TODO these should probably go in particleinfo.h in a similar manner to the particledefine.h names
const char* ParticleTypeName[PT_NONE]
= {
	"fluid",
	"boundary",
	"vertex",
	"testpoint"
};

void
VTKWriter::write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints)
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
	const float *effvisc = buffers.getData<BUFFER_EFFVISC>();
	const float *turbvisc = buffers.getData<BUFFER_TURBVISC>();
	const float *spsturbvisc = buffers.getData<BUFFER_SPS_TURBVISC>();
	const float *effpres = buffers.getData<BUFFER_EFFPRES>();
	const float4 *eulervel = buffers.getData<BUFFER_EULERVEL>();
	const float *priv = buffers.getData<BUFFER_PRIVATE>();
	const float2 *priv2 = buffers.getData<BUFFER_PRIVATE2>();
	const float4 *priv4 = buffers.getData<BUFFER_PRIVATE4>();
	const vertexinfo *vertices = buffers.getData<BUFFER_VERTICES>();
	const float *intEnergy = buffers.getData<BUFFER_INTERNAL_ENERGY>();
	const float4 *forces = buffers.getData<BUFFER_FORCES>();

	const float *cspm_wcoeff = buffers.getData<BUFFER_WCOEFF>();
	const float2* const* cspm_fcoeff = buffers.getRawPtr<BUFFER_FCOEFF>();

	const neibdata *neibslist = buffers.getData<BUFFER_NEIBSLIST>();

	// TODO debugging
	const uint *nextIDs = buffers.getData<BUFFER_NEXTID>();

	ushort *neibsnum = new ushort[numParts];

	if (neibslist) {
		const uint *cellStart = buffers.getData<BUFFER_CELLSTART>();

		// initialize cell_to_offset array
		if (cell_to_offset[0].x == 0) {
			for(char z=-1; z<=1; z++) {
				for(char y=-1; y<=1; y++) {
					for(char x=-1; x<=1; x++) {
						int i = (x + 1) + (y + 1)*3 + (z + 1)*9;
						cell_to_offset[i] =  make_char3(x, y, z);
					}
				}
			}
		}

		ofstream neibs;
		open_data_file(neibs, "neibs", current_filenum(), ".txt");

		for (uint i = 0; i < numParts; ++i) {
			/* The neighbors list is split in three sections; if F denotes
			 * fluid neighbors, B boundary neighbors and V vertex neighbors,
			 * the layout for a single particle is:
			 * FFFF...*...*...BBBVVVV...*
			 * →→→→→→→→   ←←←←←←←→→→→→→→→
			 * where the * symbols represent end markers (NEIBS_END).
			 *
			 * F neighbors are stored from position 0 upwards,
			 * B neighbors are stored from position m_neib_bound_pos downwards,
			 * V neighbors are stored from position m_neib_bound_pos+1 upwards.
			 *
			 * Since our purpose here is debugging the neighbors list, for each
			 * particle we output multiple lines: a RAW line simply printing all values
			 * for the given particle, followed by one line 'decoding' the neighbors list
			 * of each type for the particle.
			 */
			neibs	<< i << "\t" << id(info[i])
				<< " (" << particle_type_sym(info[i]) << ")";
			// raw output first
#if NL_INTERLEAVED
			for (uint index = i; index < m_neiblist_end; index += m_neiblist_stride) {
#else
			for (uint index = i*m_neiblist_size; index < (i+1)*m_neiblist_size; index += 1) {
#endif
				neibdata neib = neibslist[index];
				if (neib == NEIBS_END)
					neibs << "\t*";
				else
					neibs << "\t" << neib;
			}

			// now each type, with decoding
			neibsnum[i] = 0;
			const uint3 gridPos = gdata->calcGridPosFromCellHash(particleHash[i]);
			// SA_BOUNDARY should check up to PT_VERTEX, all other boundaries
			// up to PT_BOUNDARY
			static const ParticleType last_pt =
				gdata->problem->simparams()->boundarytype == SA_BOUNDARY ?
				PT_VERTEX : PT_BOUNDARY;

			for (ParticleType pt = PT_FLUID; pt <= last_pt; pt = (ParticleType)(pt+1)) {
				neibs << "\n\t";
				uint num_neibs = 0;
				const uint nl_stride = NL_INTERLEAVED ? m_neiblist_stride : 1;
				const uint start = i*(NL_INTERLEAVED ? 1 : m_neiblist_size) + (
					(pt == PT_FLUID)	? 0 :
					(pt == PT_BOUNDARY)	? m_neib_bound_pos :
					/*pt == PT_VERTEX */	  m_neib_bound_pos+1)
					*nl_stride;
				const uint stride = (pt == PT_BOUNDARY ? -1 : 1)*nl_stride;
#if NL_INTERLEAVED
				const uint end = (pt == PT_BOUNDARY ? i : m_neiblist_end);
#else
				const uint end = (pt == PT_BOUNDARY ? i*m_neiblist_size : (i+1)*m_neiblist_size);
#endif

				uint neib_cell_base_index = UINT_MAX;

				for (uint index = start; index < end; index += stride) {
					neibdata neib = neibslist[index];
					if (neib == NEIBS_END)
						break;
					++num_neibs;
					neibs << "\t" << neib << " (";
					if (neib >= CELLNUM_ENCODED) {
						int neib_cellnum = DECODE_CELL(neib);
						neib = neib & NEIBINDEX_MASK;
						neibs << neib_cellnum << "|" << neib << " = ";

						int3 neib_grid_pos =
							make_int3(gridPos.x, gridPos.y, gridPos.z) +
							cell_to_offset[neib_cellnum];
						uint neib_grid_hash = gdata->calcGridHashPeriodic(neib_grid_pos);
						neib_cell_base_index = cellStart[neib_grid_hash];
					}
					uint neib_idx = neib_cell_base_index + neib;
					neibs << neib_idx << ")";
				}
				neibs << "\t[" << num_neibs << "]";
				neibsnum[i] += num_neibs;
			}
			neibs << "\n";
		}
		neibs.close();
	}

	string filename;

	ofstream fid;
	if (gdata->run_mode == REPACK)
		filename = open_data_file(fid, "REPACK", current_filenum());
	else
		filename = open_data_file(fid, "PART", current_filenum());
	VTKAppender appender(fid, info, gdata, node_offset, numParts);

	// Header
	//====================================================================================
	fid << "<?xml version='1.0'?>" << endl;
	fid << "<VTKFile type='PolyData'  version='0.1'  byte_order='" <<
		endianness[*(char*)&endian_int & 1] << "'>" << endl;
	fid << " <PolyData>" << endl;
	fid << "  <Piece NumberOfPoints='" << numParts << "' NumberOfVerts='" << numParts << "'>" << endl;

	// position
	fid << "   <Points>" << endl;
	appender.append_data(pos, "Position", nullptr);
	fid << "   </Points>" << endl;

	fid << "   <PointData Scalars='" << (neibslist ? "Neibs" : "Pressure") << "' Vectors='Velocity'>" << endl;

	// neibs
	if (neibslist) {
		appender.append_local_data(neibsnum, "Neibs");
	}

	if (nextIDs) {
		appender.append_data(nextIDs, "NextID");
	}

	if (intEnergy) {
		appender.append_data(intEnergy, "Internal Energy");
	}

	if (forces) {
		appender.append_data(forces,
			"Spatial acceleration", "Continuity derivative");
	}

	// pressure
	appender.append_data(vel, "Pressure", get_pressure);

	// velocity
	appender.append_data(vel, "Velocity", nullptr);

	// density
	appender.append_data(vel, "Density", get_density);

	if (g_debug.numerical_density) {
		appender.append_data(vel, "Relative Density", get_last_component);
	}

	// mass
	appender.append_data(pos, "Mass", demote_w);

	// gamma and its gradient
	if (gradGamma) {
		appender.append_data(gradGamma, "Gradient Gamma", "Gamma");
	}

	// Effective viscosity
	if (effvisc) {
		appender.append_data(effvisc, "Effective viscosity");
	}

	// turbulent kinetic energy
	if (tke) {
		appender.append_data(tke, "TKE");
	}

	// turbulent epsilon
	if (eps) {
		appender.append_data(eps, "Epsilon");
	}

	// eddy viscosity
	if (turbvisc) {
		appender.append_data(turbvisc, "Eddy viscosity");
	}

	// SPS eddy viscosity
	if (spsturbvisc) {
		appender.append_data(spsturbvisc, "SPS turbulent viscosity");
	}

	// effective pressure
	if (effpres) {
		appender.append_data(effpres, "Effective pressure");
	}

	/* Fluid number is only included if there are more than 1 */
	const bool write_fluid_num = (gdata->problem->physparams()->numFluids() > 1);

	/* Object number is only included if there are any */
	// TODO a better way would be for GPUSPH to expose the highest
	// object number ever associated with any particle, so that we
	// could check that
	const uint numbodies = gdata->problem->simparams()->numbodies;
	const uint numOpenBoundaries = gdata->problem->simparams()->numOpenBoundaries;
	const bool write_part_obj = (numbodies > 0 || numOpenBoundaries > 0);
	// does the number of objects fit in an unsigned char?
	const bool write_few_obj = (numbodies + numOpenBoundaries < UCHAR_MAX);

	// particle info
	if (info) {
		// type
		appender.append_data(info, "Part type", get_part_type);

		// flag
		appender.append_data(info, "Part flags", get_part_flags);

		// fluid number
		if (write_fluid_num) {
			// Limit to 256 fluids
			appender.append_data(info, "Fluid number", get_fluid_num);
		}

		// object number
		if (write_part_obj) {
			if (write_few_obj)
				appender.append_data(info, "Part object", get_object_few);
			else
				appender.append_data(info, "Part object", get_object);
		}

		// id
		appender.append_data(info, "Part id", id);
	}

	if (vertices) {
		appender.append_data(vertices, "Vertices");
	}

	// cell index
	appender.append_data(particleHash, "CellIndex", get_cellindex);

	if (eulervel) {
		// Eulerian velocity
		appender.append_data(eulervel, "Eulerian velocity", nullptr);

		// Eulerian density
		appender.append_data(eulervel, "Eulerian density", get_density);
	}

	// vorticity
	if (vort) {
		appender.append_data(vort, "Vorticity");
	}

	// normals
	if (normals) {
		appender.append_data(normals, "Normals", "Criteria");
	}

	// private
	if (priv) {
		appender.append_data(priv, m_problem->get_private_name(BUFFER_PRIVATE).c_str());
		if (priv2)
			appender.append_data(priv2, m_problem->get_private_name(BUFFER_PRIVATE2).c_str());
		if (priv4)
			appender.append_data(priv4, m_problem->get_private_name(BUFFER_PRIVATE4).c_str());
	}

	// volume
	if (vol) {
		appender.append_data(vol, "Volume");
	}

	// sigma
	if (sigma) {
		appender.append_data(sigma, "Sigma");
	}

	// device index
	if (MULTI_DEVICE) {
		appender.append_local_data("DeviceIndex", [this](size_t i) -> dev_idx_t {
			GlobalData const *gdata(this->gdata);
			uint numdevs = gdata->devices;
			for (uint d = 0; d < numdevs; ++d) {
				uint partsInDevice = gdata->s_hPartsPerDevice[d];
				if (i < partsInDevice)
					return gdata->GLOBAL_DEVICE_ID(gdata->mpi_rank, d);
				i -= partsInDevice;
			}
			// If we got here, the sum of all device particles is less than i,
			// which is an error
			throw runtime_error("unable to find device particle belongs to");
		});
	}

	if (cspm_wcoeff) {
		appender.append_data(cspm_wcoeff, "CSPM W");
		appender.append_local_data("CSPM gradW", [this, cspm_fcoeff](size_t i) -> symtensor3 {
			return fetch_tensor(cspm_fcoeff, i);
		});
	}

	fid << "   </PointData>" << endl;

	// Cells data
	fid << "   <Verts>" << endl;

	// connectivity
	appender.append_local_data("connectivity", [](size_t i)->uint { return i; });
	// offsets
	appender.append_local_data("offsets", [](size_t i)->uint { return i+1; });

	fid << "   </Verts>" << endl;
	fid << "  </Piece>" << endl;

	fid << " </PolyData>" << endl;
	fid << " <AppendedData encoding='raw'>\n_";
	//====================================================================================

	appender.write_appended_data();

	fid << " </AppendedData>" << endl;
	fid << "</VTKFile>" << endl;

	fid.close();

	add_block("Particles", filename);

	delete[] neibsnum;
}

void
VTKWriter::write_WaveGage(double t, GageList const& gage)
{
	static const int ndims = space_dimensions_for(gdata->problem->simparams()->dimensions);

	// For gages without points, z will be NaN, and we'll set
	// it to match the lowest world coordinate
	static const double worldBottom = ndims == 2 ? gdata->worldOrigin.y : gdata->worldOrigin.z;

	ofstream fp;
	string filename = open_data_file(fp, "WaveGage", current_filenum(), ".vtu");

	size_t num = gage.size();

	// Header
	fp << "<?xml version='1.0'?>" << endl;
	fp << "<VTKFile type='UnstructuredGrid'  version='0.1'  byte_order='" <<
		endianness[*(char*)&endian_int & 1] << "'>" << endl;
	fp << " <UnstructuredGrid>" << endl;
	fp << "  <Piece NumberOfPoints='" << num << "' NumberOfCells='" << num << "'>" << endl;

	//Writing Position
	fp << "   <Points>" << endl;
	fp << "	<DataArray type='Float32' NumberOfComponents='3' format='ascii'>" << endl;
	for (size_t i=0; i <  num; i++) {
		double3 g = gage[i]->get_3D_pos();
		if (ndims == 2 && !isfinite(g.y)) g.y = worldBottom;
		if (ndims == 3 && !isfinite(g.z)) g.z = worldBottom;
		fp << g.x << "\t" << g.y << "\t" << g.z << "\t";
	}
	fp << endl;
	fp << "	</DataArray>" << endl;
	fp << "   </Points>" << endl;

	// Cells data
	fp << "   <Cells>" << endl;
	fp << "	<DataArray type='Int32' Name='connectivity' format='ascii'>" << endl;
	for (size_t i = 0; i < num; i++)
		fp << i << "\t" ;
	fp << endl;
	fp << "	</DataArray>" << endl;
	fp << "" << endl;

	fp << "	<DataArray type='Int32' Name='offsets' format='ascii'>" << endl;
	for (size_t i = 0; i < num; i++)
		fp << (i+1) << "\t" ;
	fp << endl;
	fp << "	</DataArray>" << endl;

	fp << "" << endl;
	fp << "	<DataArray type='Int32' Name='types' format='ascii'>" << endl;
	for (size_t i = 0; i < num; i++)
		fp << 1 << "\t" ;
	fp << endl;
	fp << "	</DataArray>" << endl;

	fp << "   </Cells>" << endl;

	fp << "  </Piece>" << endl;
	fp << " </UnstructuredGrid>" << endl;
	fp << "</VTKFile>" <<endl;

	fp.close();

	add_block("WaveGages", filename);
}

static inline void chomp(double3 &pt, double eps=FLT_EPSILON)
{
		if (fabs(pt.x) < eps)
			pt.x = 0;
		if (fabs(pt.y) < eps)
			pt.y = 0;
		if (fabs(pt.z) < eps)
			pt.z = 0;
}

// check that pt is between inf and sup, with FLT_EPSILON relative tolerange
static inline bool bound(float pt, float inf, float sup)
{
	// when inf or sup is zero, the tolerance must be absolute, not relative
	// Also note the use of absolue value to ensure the limits are expanded
	// in the right direction
	const float lower = inf ? inf - FLT_EPSILON*fabs(inf) : -FLT_EPSILON;
	const float upper = sup ? sup + FLT_EPSILON*fabs(sup) : FLT_EPSILON;
	return (pt > lower) && (pt < upper);
}

void
VTKWriter::save_planes()
{
	ofstream fp;
	m_planes_fname = open_data_file(fp, "PLANES", "", ".vtu");

	fp << set_vector_fmt(" ");

	PlaneList const& planes = gdata->s_hPlanes;
	const double3 wo = gdata->problem->get_worldorigin();
	const double3 ow = wo + gdata->problem->get_worldsize();

	typedef vector<pair<double4, int> > CheckList;
	typedef vector<double3> CoordList;

	// We want to find the intersection of the planes defined in the boundary
	// with the bounding box of the plane (wo to ow). We do this by finding the intersection
	// with each pair of planes of the bounding box. The CheckList is composed of such pairs,
	// ordered such that the intersections are returned in sequence (otherwise the resulting
	// planes in the VTK would come out butterfly-shaped.
	// The number associated with each pair of planes is the index of the coordinate that must
	// be found by the intersection.
	CheckList checks;

	checks.push_back(make_pair(
			make_double4(wo.x, wo.y, 0, 1), 2));
	checks.push_back(make_pair(
			make_double4(wo.x, 0, wo.z, 1), 1));
	checks.push_back(make_pair(
			make_double4(wo.x, ow.y, 0, 1), 2));
	checks.push_back(make_pair(
			make_double4(wo.x, 0, ow.z, 1), 1));

	checks.push_back(make_pair(
			make_double4(ow.x, ow.y, 0, 1), 2));
	checks.push_back(make_pair(
			make_double4(ow.x, 0, ow.z, 1), 1));

	checks.push_back(make_pair(
			make_double4(ow.x, wo.y, 0, 1), 2));
	checks.push_back(make_pair(
			make_double4(0, wo.y, wo.z, 1), 0));
	checks.push_back(make_pair(
			make_double4(0, wo.y, ow.z, 1), 0));

	checks.push_back(make_pair(
			make_double4(0, ow.y, ow.z, 1), 0));

	checks.push_back(make_pair(
			make_double4(ow.x, 0, wo.z, 1), 1));
	checks.push_back(make_pair(
			make_double4(0, ow.y, wo.z, 1), 0));

	CoordList centers;
	CoordList normals;
	vector< CoordList > all_intersections;

	// we will store one point per plane (center)
	// followed by the intersections for each plane with the domain bounding box
	size_t npoints = planes.size();

	// find the intersection of each plane with the domain bounding box
	PlaneList::const_iterator plane(planes.begin());
	for (; plane != planes.end(); ++plane) {
		centers.push_back(gdata->calcGlobalPosOffset(plane->gridPos, plane->pos) + wo);
		double3 &cpos = centers.back();
		chomp(cpos);

		normals.push_back(make_double3(plane->normal));
		chomp(normals.back());
		double3 const& normal = normals.back();

		double4 implicit = make_double4(normal, -dot(cpos, normal));

#if DEBUG_VTK_PLANES
		cout << "plane through " << cpos << " normal " << normal << endl;
		cout << "\timplicit " << implicit << endl;
#endif

		all_intersections.push_back( vector<double3>() );

		vector<double3> & intersections = all_intersections.back();

		CheckList::const_iterator check(checks.begin());
		for (; check != checks.end(); ++check) {
			const double4 &ref = check->first;
			const int coord = check->second;
			double3 pt = make_double3(ref);
			switch (coord) {
			case 0:
				if (!normal.x) continue;
				pt.x = -dot(implicit, ref)/normal.x;
				if (!bound(pt.x, wo.x, ow.x)) continue;
				break;
			case 1:
				if (!normal.y) continue;
				pt.y = -dot(implicit, ref)/normal.y;
				if (!bound(pt.y, wo.y, ow.y)) continue;
				break;
			case 2:
				if (!normal.z) continue;
				pt.z = -dot(implicit, ref)/normal.z;
				if (!bound(pt.z, wo.z, ow.z)) continue;
				break;
			}
			chomp(pt);
			intersections.push_back(pt);
#if DEBUG_VTK_PLANES
			cout << "\t(" << (check-checks.begin()) << ")" << endl;
			cout << "\tcheck " << ref << " from " << coord << endl;
			cout << "\t\tpoint " << intersections.back() << endl;
#endif
		}
		npoints += intersections.size();
	}

	size_t offset = 0;

	fp << "<?xml version='1.0'?>" << endl;
	fp << "<VTKFile type='UnstructuredGrid'  version='0.1'  byte_order='" <<
		endianness[*(char*)&endian_int & 1] << "'>" << endl;
	fp << " <UnstructuredGrid>" << endl;
	fp << "  <Piece NumberOfPoints='" << npoints
		<< "' NumberOfCells='" << planes.size() << " '>" << endl;

	fp << "   <Points>" << endl;

	fp << "<DataArray type='Float64' NumberOfComponents='3'>" << endl;

	// intersection points
	for (vector<CoordList>::const_iterator pl(all_intersections.begin());
		pl < all_intersections.end(); ++pl) {
		CoordList const& pts = *pl;
		for (CoordList::const_iterator pt(pts.begin()); pt != pts.end(); ++pt)
			fp << *pt << endl;
	}

	// center points
	for (CoordList::const_iterator pt(centers.begin()); pt != centers.end(); ++pt)
		fp << *pt << endl;

	fp << "</DataArray>" << endl;

	fp << "   </Points>" << endl;

	fp << "   <Cells>" << endl;
	fp << "<DataArray type='Int32' Name='connectivity'>" << endl;
	// intersection points
	offset = 0;
	for (vector<CoordList>::const_iterator pl(all_intersections.begin());
		pl < all_intersections.end(); ++pl) {
		CoordList const& pts = *pl;
		for (size_t i = 0; i < pts.size(); ++i) {
			fp << " " << offset + i;
		}
		offset += pts.size();
		fp << endl;
	}
	fp << "</DataArray>" << endl;
	fp << "<DataArray type='Int32' Name='offsets'>" << endl;
	offset = 0;
	for (size_t i = 0; i < planes.size(); ++i) {
		offset += all_intersections[i].size();
		fp << offset << endl;
	}
	fp << "</DataArray>" << endl;
	fp << "<DataArray type='Int32' Name='types'>" << endl;
	for (size_t i = 0; i < planes.size(); ++i) {
		fp << 7 << " "; // POLYGON
	}
	fp << endl;
	fp << "</DataArray>" << endl;
	fp << "   </Cells>" << endl;

	fp << "   <PointData />" << endl;

	fp << "   <CellData Normals='Normals'>" << endl;
	fp << "<DataArray type='Float64' Name='Normals' NumberOfComponents='3'>" << endl;
	for (CoordList::const_iterator pt(normals.begin()); pt != normals.end(); ++pt)
		fp << *pt << endl;
	fp << "</DataArray>" << endl;
	fp << "   </CellData>" << endl;

	fp << "  </Piece>" << endl;
	fp << " </UnstructuredGrid>" << endl;
	fp << "</VTKFile>" <<endl;

	fp.close();
}

void
VTKWriter::save_dem()
{
	ofstream fp;
	m_dem_fname = open_data_file(fp, "DEM", "", ".vts");

	const float *dem = gdata->problem->get_dem();
	const int cols = gdata->problem->get_dem_ncols();
	const int rows = gdata->problem->get_dem_nrows();

	const float ewres = gdata->problem->physparams()->ewres;
	const float nsres = gdata->problem->physparams()->nsres;

	const string extent="1 " + to_string(cols) + " 1 " + to_string(rows) + " 0 0";

	// TODO support DEM offset

	fp	<< "<?xml version='1.0'?>\n"
		<< "<VTKFile type='StructuredGrid'  version='0.1'  byte_order='" << endianness[*(char*)&endian_int & 1] << "'>\n"
		<< " <StructuredGrid WholeExtent='" + extent + "'>\n"
		<< "  <Piece Extent='" + extent + "'>\n"
		<< "   <Points><DataArray type='Float32' NumberOfComponents='3' format='appended' offset='0'/></Points>\n"
		<< "  </Piece>\n"
		<< " </StructuredGrid>\n"
		<< " <AppendedData encoding='raw'>_";

	uint numbytes = 3*sizeof(float)*cols*rows;
	fp.write(reinterpret_cast<char*>(&numbytes), sizeof(numbytes));

	uint i = 0;
	float3 pt;
	for (int row = 0; row < rows; ++row) {
		pt.y = row*nsres;
		for (int col = 0; col < cols; ++col) {
			pt.x = col*ewres;
			pt.z = dem[i];
			fp.write(reinterpret_cast<char*>(&pt), sizeof(pt));
			++i;
		}
	}

	fp	<< " <AppendedData>\n</VTKFile>" << endl;

	fp.close();
}

void
VTKWriter::mark_timefile()
{
	if (!m_timefile)
		return;
	// Mark the current position, close the XML, go back
	// to the marked position
	ofstream::pos_type mark = m_timefile.tellp();
	m_timefile << " </Collection>\n";
	m_timefile << "</VTKFile>" << endl;
	m_timefile.seekp(mark);
}
