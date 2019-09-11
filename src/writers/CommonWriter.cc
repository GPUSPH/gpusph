/*  Copyright (c) 2014-2019 INGV, EDF, UniCT, JHU

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

#include <fstream>

#include "CommonWriter.h"

#include "GlobalData.h"
#include "simflags.h"
#include "vector_print.h"

#include "gpusph_version.opt"
#include "make_show.opt"
#include "git_info.opt"

using namespace std;

CommonWriter::CommonWriter(const GlobalData *_gdata)
	: Writer(_gdata)
{
	m_fname_sfx = ".txt";

	// special value denoting default behavior of writing every time
	// any other writer does
	m_writefreq = NAN;

	write_summary();

	// TODO only do this if energy writing is enabled
	string energy_fn = open_data_file(m_energyfile, "energy");
	if (m_energyfile) {
		m_energyfile << "time";
		uint fluid = 0;
		for (; fluid < m_problem->physparams()->numFluids(); ++fluid)
			m_energyfile	<< "\tkinetic" << fluid
							<< "\tpotential" << fluid
							<< "\tinternal" << fluid;
		/* non-fluid */
		m_energyfile	<< "\tkineticNF"
						<< "\tpotentialNF"
						<< "\tinternalNF";
		m_energyfile << "\ttotal";
		m_energyfile << endl;
		m_energyfile << set_vector_fmt("\t");
		m_energyfile.precision(16);
	}

	size_t ngages = m_problem->simparams()->gage.size();
	if (ngages > 0) {
		string WaveGage_fn = open_data_file(m_WaveGagefile, "WaveGage");
		if (m_WaveGagefile) {
			m_WaveGagefile << "time";
			for (size_t gage = 0; gage < ngages; ++gage)
				m_WaveGagefile << "\tzgage" << gage;
			m_WaveGagefile << endl;
			m_WaveGagefile.precision(9);
		}
	}

	// TODO only do this if object data writing is enabled
	size_t nbodies = m_problem->simparams()->numbodies;
	if (nbodies > 0) {
		string rbdata_fn = open_data_file(m_objectfile, "rbdata");
		if (m_objectfile) {
			m_objectfile << "time";
			for (size_t obj = 0; obj < nbodies; ++obj) {
				// object index
				m_objectfile << "\tindex";
				// center of mass
				m_objectfile << "\tCM" << obj << "_X";
				m_objectfile << "\tCM" << obj << "_Y";
				m_objectfile << "\tCM" << obj << "_Z";
				// quaternion
				m_objectfile << "\tQ" << obj << "_1";
				m_objectfile << "\tQ" << obj << "_I";
				m_objectfile << "\tQ" << obj << "_J";
				m_objectfile << "\tQ" << obj << "_K";
			}
			m_objectfile << endl;
			m_objectfile.precision(9);
			m_objectfile << set_vector_fmt("\t");
			m_objectfile << scientific;
		}
	}

	nbodies = m_problem->simparams()->numforcesbodies;
	if (nbodies) {
		string objforce_fn = open_data_file(m_objectforcesfile, "objectforces");
		if (m_objectforcesfile) {
			m_objectforcesfile << "time";
			for (size_t obj = 0; obj < nbodies; ++obj) {
				// object index
				m_objectforcesfile << "\tindex";
				// computed forces
				m_objectforcesfile << "\tComputed_F" << obj << "_X";
				m_objectforcesfile << "\tComputed_F" << obj << "_Y";
				m_objectforcesfile << "\tComputed_F" << obj << "_Z";
				// computed torques
				m_objectforcesfile << "\tComputed_M" << obj << "_X";
				m_objectforcesfile << "\tComputed_M" << obj << "_Y";
				m_objectforcesfile << "\tComputed_M" << obj << "_Z";
				// applied forces
				m_objectforcesfile << "\tApplied_F" << obj << "_X";
				m_objectforcesfile << "\tApplied_F" << obj << "_Y";
				m_objectforcesfile << "\tApplied_F" << obj << "_Z";
				// applied torques
				m_objectforcesfile << "\tApplied_M" << obj << "_X";
				m_objectforcesfile << "\tApplied_M" << obj << "_Y";
				m_objectforcesfile << "\tApplied_M" << obj << "_Z";
			}
			m_objectforcesfile << endl;
			m_objectforcesfile.precision(9);
			m_objectforcesfile << set_vector_fmt("\t");
			m_objectforcesfile << scientific;
		}
	}

	PostProcessEngineSet const& enabledPostProcess = gdata->simframework->getPostProcEngines();
	for (PostProcessEngineSet::const_iterator flt(enabledPostProcess.begin());
		flt != enabledPostProcess.end(); ++flt) {
		switch (flt->first) {
		case FLUX_COMPUTATION: {
			uint numOB = m_problem->simparams()->numOpenBoundaries;
			if (numOB) {
				string flux_fn = open_data_file(m_fluxfile, "IOflux");
				if (m_fluxfile) {
					m_fluxfile << "time";
					for (uint i=0; i<numOB; i++)
						m_fluxfile << "\tFlux_" << i;
					m_fluxfile << endl;
				}
			}
			break;
		}
		default:
			break;
		}
	}
}

CommonWriter::~CommonWriter()
{
	if (m_energyfile)
		m_energyfile.close();
	if (m_WaveGagefile)
		m_WaveGagefile.close();
	if (m_objectfile)
		m_objectfile.close();
	if (m_objectforcesfile)
		m_objectforcesfile.close();
}

/// Write testpoints to CSV file
void
CommonWriter::write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints)
{
	if (!testpoints)
		return;

	const double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();
	const hashKey *particleHash = buffers.getData<BUFFER_HASH>();
	const float4 *vel = buffers.getData<BUFFER_VEL>();
	const particleinfo *info = buffers.getData<BUFFER_INFO>();
	const float *tke = buffers.getData<BUFFER_TKE>();
	const float *eps = buffers.getData<BUFFER_EPSILON>();

	if (!info)
		return; // this shouldn't happen, but whatever

	ofstream testpoints_file;
	string testpoints_fname = open_data_file(testpoints_file, "testpoints/testpoints", current_filenum(), ".csv");

	// 9 decimal digits
	testpoints_file.precision(9);
	// output vectors without parenthesis and with a single comma as separator
	testpoints_file << set_vector_fmt(",");

	// write CSV header
	testpoints_file << "T,ID,Pressure,Object,CellIndex,PosX,PosY,PosZ,VelX,VelY,VelZ,Tke,Eps" << endl;

	for (uint i=node_offset; i < node_offset + numParts; i++) {
		if (!TESTPOINT(info[i]))
			continue;

		const float tkeVal = tke ? tke[i] : 0;
		const float epsVal = eps ? eps[i] : 0;

		testpoints_file << t << ","
			<< id(info[i]) << ","
			<< vel[i].w << ","
			<< object(info[i]) << ","
			<< cellHashFromParticleHash( particleHash[i] ) << ","
			<< as_double3(pos[i]) << ","
			<< as_float3(vel[i]) << ","
			<< tkeVal << ","
			<< epsVal << endl;
	}
	testpoints_file.close();
}

void
CommonWriter::write_energy(double t, double4 *energy)
{
	double total = 0;
	if (m_energyfile) {
		m_energyfile << t;
		uint fluid = 0;
		for (; fluid < m_problem->physparams()->numFluids(); ++fluid) {
			m_energyfile	<< "\t" << as_double3(energy[fluid]);
			total += energy[fluid].x + energy[fluid].y + energy[fluid].z;
		}
		m_energyfile	<< "\t" << as_double3(energy[MAX_FLUID_TYPES]);
		total += energy[MAX_FLUID_TYPES].x + energy[MAX_FLUID_TYPES].y + energy[MAX_FLUID_TYPES].z;
		m_energyfile << "\t" << total;
		m_energyfile << endl;
	}
}

void
CommonWriter::write_WaveGage(double t, GageList const& gage)
{
	if (m_WaveGagefile) {
		m_WaveGagefile << t;
		for (size_t i=0; i < gage.size(); i++) {
			m_WaveGagefile << "\t" << gage[i].z;
		}
		m_WaveGagefile << endl;
	}
}

void
CommonWriter::write_objects(double t)
{
	if (m_objectfile) {
		m_objectfile << t;
		const MovingBodiesVect & mbvect = m_problem->get_mbvect();
		for (vector<MovingBodyData *>::const_iterator it = mbvect.begin(); it != mbvect.end(); ++it) {
			const MovingBodyData *mbdata = *it;
			m_objectfile << "\t" << mbdata->index
				<< "\t" << mbdata->kdata.crot
				<< "\t" << mbdata->kdata.orientation.params();
		}
		m_objectfile << endl;
	}
}

void
CommonWriter::write_objectforces(double t, uint numobjects,
		const float3* computedforces, const float3* computedtorques,
		const float3* appliedforces, const float3* appliedtorques)
{
	if (m_objectforcesfile) {
		const MovingBodiesVect & mbvect = m_problem->get_mbvect();
		m_objectforcesfile << t;
		for (uint i=0; i < numobjects; i++) {
			m_objectforcesfile << "\t" << mbvect[i]->index;
			m_objectforcesfile << "\t" << computedforces[i];
			m_objectforcesfile << "\t" << computedtorques[i];
			m_objectforcesfile << "\t" << appliedforces[i];
			m_objectforcesfile << "\t" << appliedtorques[i];
		}
		m_objectforcesfile << endl;
	}
}

void
CommonWriter::write_flux(double t, float *fluxes)
{
	uint numOB = m_problem->simparams()->numOpenBoundaries;
	if (m_fluxfile) {
		m_fluxfile << t;
		for (uint i=0; i<numOB; i++)
			m_fluxfile << "\t" << fluxes[i];
		m_fluxfile << endl;
	}
}


bool
CommonWriter::need_write(double t) const
{
	if (m_writefreq < 0)
		return false; // special
	return Writer::need_write(t);
}

static const char* TF[] = { "false", "true" };
static const char* ED[] = { "disabled", "enabled" };

// TODO mark params overridden by options with a *
void
CommonWriter::write_simparams(ostream &out)
{
	const SimParams *SP = m_problem->simparams();

	out << "Simulation parameters:" << endl;

	out << " deltap = " << m_problem->get_deltap() << endl;
	out << " sfactor = " << SP->sfactor << endl;
	out << " slength = " << SP->slength << endl;
	out << " kerneltype: " << SP->kerneltype << " (" << KernelName[SP->kerneltype] << ")" << endl;
	out << " kernelradius = " << SP->kernelradius << endl;
	out << " influenceRadius = " << SP->influenceRadius << endl;
	out << " SPH formulation: " << SP->sph_formulation << " (" << SPHFormulationName[SP->sph_formulation] << ")" << endl;
	out << " multi-fluid support: " << ED[!!(SP->simflags & ENABLE_MULTIFLUID)] << endl;
	out << " Rheology: " << RheologyName[SP->rheologytype] << endl;
	if (SP->rheologytype != INVISCID) {
		out << "\tTurbulence model: " << TurbulenceName[SP->turbmodel] << endl;
		out << "\tComputational viscosity type: " << ComputationalViscosityName[SP->compvisc] << endl;
		out << "\tViscous model operator: " << ViscousModelName[SP->viscmodel];
		if (SP->viscmodel == MONAGHAN)
			out << ",\tcoefficient: " << m_problem->physparams()->monaghan_visc_coeff;
		out << endl;
		out << "\tViscous averaging operator: " << AverageOperatorName[SP->viscavgop] << endl;
		if (SP->is_const_visc)
			out << "\t(constant viscosity optimizations)" << endl;
	}

	if (SP->rheologytype == GRANULAR) {
		out << " Granular rheology: effective pressure Jacobi solver parameters: " << endl;
		out << "\tMaximum number of iterations: " << SP->jacobi_maxiter << endl;
		out << "\tBackward error threshold (boundary convergence): " << SP->jacobi_backerr << endl;
		out << "\tResidual threshold (fluid convergence): " << SP->jacobi_residual << endl;
	}

	out << " periodicity: " << SP->periodicbound << " (" << PeriodicityName[SP->periodicbound] << ")" << endl;

	out << " initial dt = " << SP->dt << endl;
	out << " simulation end time = " << SP->tend << endl;
	out << " neib list construction every " << SP->buildneibsfreq << " iterations" << endl;

	/* Iterate over enabled filters, showing their name and frequency */
	FilterFreqList const& enabledFilters = gdata->simframework->getFilterFreqList();
	FilterFreqList::const_iterator flt(enabledFilters.begin());
	FilterFreqList::const_iterator flt_end(enabledFilters.end());
	while (flt != flt_end) {
		out << " " << FilterName[flt->first] << " filter every "
			<< flt->second << " iterations" << endl;
		++flt;
	}

	out << " adaptive time stepping " << ED[!!(SP->simflags & ENABLE_DTADAPT)] << endl;
	if (SP->simflags & ENABLE_DTADAPT)
		out << "    safety factor for adaptive time step = " << SP->dtadaptfactor << endl;
	out << " internal energy computation " << ED[!!(SP->simflags & ENABLE_INTERNAL_ENERGY)] << endl;

	out << " XSPH correction " << ED[!!(SP->simflags & ENABLE_XSPH)] << endl;

	out << " Corrective Smoothed Particle Method (CSPM) " << ED[!!(SP->simflags & ENABLE_CSPM)] << endl;
	switch (SP->densitydiffusiontype) {
	case FERRARI:
		out << " Ferrari density diffusion enabled" << endl;
		out << "    Ferrari length scale = " ;
		if (isnan(SP->ferrariLengthScale))
			out << "unset";
		else
			out << SP->ferrariLengthScale;
		out << endl;
		out << "    Diffusion coefficient = " << SP->densityDiffCoeff << endl;
		break;
	case BREZZI:
		out << " Brezzi density diffusion enabled" << endl;
		out << "    Diffusion coefficient = " << SP->densityDiffCoeff << endl;
		break;
	case COLAGROSSI:
		out << " Colagrossi density diffusion enabled" << endl;
		// recompute the input xi
		out << "    ξ = " << SP->densityDiffCoeff/(2.0f*SP->slength) << endl;
		break;
	default:
		break;
	}
	out << " moving bodies " << ED[!!(SP->simflags & ENABLE_MOVING_BODIES)] << endl;
	out << " open boundaries " << ED[!!(SP->simflags & ENABLE_INLET_OUTLET)] << endl;
	out << " water depth computation " << ED[!!(SP->simflags & ENABLE_WATER_DEPTH)] << endl;
	out << " time-dependent gravity " << ED[!!(SP->gcallback)] << endl;

	const bool has_dem = !!(SP->simflags & ENABLE_DEM);
	const bool has_planes = !!(SP->simflags & ENABLE_PLANES);

	out << " geometric boundaries: " << endl;
	out << "   DEM: " << ED[has_dem];
	if (has_dem)
		out << " (" << (!m_problem->get_dem() ? "NOT" : "") << "present)";
	out << endl;

	out << "   planes: " << ED[has_planes];
	if (has_planes)
		out << ", " << gdata->s_hPlanes.size() << " defined";
	out << endl;

	/* Iterate over enabled postprocessing engines, showing their name and options */
	PostProcessEngineSet const& postProcs(gdata->simframework->getPostProcEngines());
	PostProcessEngineSet::const_iterator pp(postProcs.begin());
	PostProcessEngineSet::const_iterator pp_end(postProcs.end());
	while (pp != pp_end) {
		out << " " << PostProcessName[pp->first] << " post-processing enabled" << endl;
		if (pp->first == SURFACE_DETECTION)
			out << "    normals saving is " << ED[!!(pp->second->get_options() & BUFFER_NORMALS)] << endl;
		++pp;
	}

#undef SP
}

void
CommonWriter::write_physparams(ostream &out)
{
	const SimParams *SP = m_problem->simparams();
	const PhysParams *PP = m_problem->physparams();

	out << "Physical parameters:" << endl;

#define g (PP->gravity)
	out << " gravity = " << PP->gravity << " [" << length(g) << "] "
		<< (SP->gcallback ? "time-dependent" : "fixed") << endl;
#undef g
	out << " numFluids = " << PP->numFluids() << endl;
	for (uint f = 0; f < PP->numFluids(); ++f) {
		out << " rho0[ " << f << " ] = " << PP->rho0[f] << endl;
		out << " B[ " << f << " ] = " << PP->bcoeff[f] << endl;
		out << " gamma[ " << f << " ] = " << PP->gammacoeff[f] << endl;
		out << " sscoeff[ " << f << " ] = " << PP->sscoeff[f] << endl;
		out << " sspowercoeff[ " << f << " ] = " << PP->sspowercoeff[f] << endl;
		out << " sound speed[ " << f << " ] = " << m_problem->soundspeed(PP->rho0[f],f) << endl;
	}
	if (PP->numFluids() > 1 && SP->sph_formulation == SPH_GRENIER)
		out << " interface epsilon = " << PP->epsinterface << endl;

	out << " partsurf = " << PP->partsurf << endl;

	out << " " << BoundaryName[SP->boundarytype] << " boundary parameters:" << endl;
	out << "\tr0 = " << PP->r0 << endl;
	switch (SP->boundarytype) {
		case LJ_BOUNDARY:
			out << "\td = " << PP->dcoeff << endl;
			out << "\tp1 = " << PP->p1coeff << endl;
			out << "\tp2 = " << PP->p2coeff << endl;
			break;
		case MK_BOUNDARY:
			out << "\tK = " << PP->MK_K << endl;
			out << "\td = " << PP->MK_d << endl;
			out << "\tbeta = " << PP->MK_beta << endl;
			break;
		default:
			/* nothing else */
			break;
	}

	out << RheologyName[SP->rheologytype] << " rheology with "
		<< TurbulenceName[SP->turbmodel] << " turbulence model. Parameters:" << endl;
	if (SP->turbmodel == ARTIFICIAL) {
		out << "\tartvisccoeff = " << PP->artvisccoeff << "" << endl;
		out << "\tepsartvisc = " << PP->epsartvisc << "" << endl;
	} else if (SP->turbmodel == SPS) {
		out << "\tSmagfactor = " << PP->smagfactor << endl;
		out << "\tkSPSfactor = " << PP->kspsfactor << endl;
	}
	if (NEEDS_EFFECTIVE_VISC(SP->rheologytype))
		out << "\tlimiting visc = " << PP->limiting_kinvisc << endl;

	for (uint f  = 0; f < PP->numFluids(); ++f) {
		out << "\tkinematicvisc[ " << f << " ] = " << PP->kinematicvisc[f] << " (m^2/s)" << endl;
		out << "\tvisc_consistency[ " << f << " ] = " << PP->visc_consistency[f] << " (Pa^n s)" << endl;

		if (NONLINEAR_RHEOLOGY(SP->rheologytype))
			out << "\tvisc_nonlinear_param[ " << f << " ] = " << PP->visc_nonlinear_param[f] << endl;
		if (YIELDING_RHEOLOGY(SP->rheologytype))
			out << "\tyield_strength[ " << f << " ] = " << PP->yield_strength[f] << " (Pa s)" << endl;
		if (REGULARIZED_RHEOLOGY(SP->rheologytype))
			out << "\tvisc_regularization_param[ " << f << " ] = " << PP->visc_regularization_param[f] << " (Pa s)" << endl;
	}
	for (uint f = 0; f < PP->numFluids(); ++f) {
		if (SP->rheologytype == GRANULAR) {
			out << "\tsinpsi[ " << f << " ] =  " << PP->sinpsi[f] << endl;
			out << "\tcohesion[ " << f << " ] =  " << PP->cohesion[f] << endl;
			out << "\tmin dynamic effvisc[ " << f << " ] =  " << PP->rho0[f]*PP->visccoeff[f] << endl;
			out << "\tmax dynamic effvisc[ " << f << " ] =  " << PP->rho0[f]*PP->limiting_kinvisc << endl;
		}
	}
	for (uint f  = 0; f < PP->numFluids(); ++f)
		out << "\tvisccoeff[ " << f << " ] = " << PP->visccoeff[f]
			<< (SP->compvisc == KINEMATIC ? " (m^2/s)" : " (Pa s)") <<endl;

	if (SP->simflags & ENABLE_XSPH) {
		out << " epsxsph = " << PP->epsxsph << endl;
	}

	if (SP->simflags & ENABLE_DEM) {
		out << " DEM resolution EW = " << PP->ewres << ", NS = " << PP->nsres << endl;
		out << " DEM displacement for normal computation dx = " << PP->demdx << ", dy = " << PP->demdy << endl;
		out << " DEM zmin = " << PP->demzmin << endl;
	}
}

void
CommonWriter::write_options(ostream &out)
{
	const Options *OP = gdata->clOptions;
	out << "Comman-line options:" << endl;
	out << " problem: " << OP->problem << endl;
	out << " dem: " << OP->dem << endl;
	out << " dir: " << OP->dir << endl;
	out << " deltap: " << OP->deltap << endl;
	out << " tend: " << OP->tend << endl;
	out << " dt: "<< OP->tend << endl;
	out << " hosts: " << OP->num_hosts << endl;
	out << " saving " << ED[!OP->nosave] << endl;
	out << " GPUDirect " << ED[OP->gpudirect] << endl;
	out << " striping " << ED[OP->striping] << endl;
	out << " async network transfers " << ED[OP->asyncNetworkTransfers] << endl;

	out << " Other options:" << endl;
	OptionMap::const_iterator opt(OP->begin());
	while (opt != OP->end()) {
		out << "  '" << opt->first << "' = '" << opt->second << "'" << endl;
		++opt;
	}
}

void
CommonWriter::write_summary(void)
{
	ofstream out;
	out.exceptions(ofstream::failbit | ofstream::badbit);
	out.open((m_problem->get_dirname() + "/summary.txt").c_str());

	write_simparams(out);
	out << endl;
	write_physparams(out);
	out << endl;
	write_options(out);
	out.close();

	// Writing out make show result
	out.open((m_problem->get_dirname() + "/make_show.txt").c_str());
	out << MAKE_SHOW_OUTPUT << endl;
	out.close();

	out.open((m_problem->get_dirname() + "/git_branch.txt").c_str());
	out << GPUSPH_VERSION << endl << GIT_INFO_OUTPUT << endl;
	out.close();
}


