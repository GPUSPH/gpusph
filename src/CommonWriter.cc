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

#include <fstream>

#include "CommonWriter.h"

#include "GlobalData.h"

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
		for (; fluid < m_problem->get_physparams()->numFluids; ++fluid)
			m_energyfile	<< "\tkinetic" << fluid
							<< "\tpotential" << fluid
							<< "\telastic" << fluid;
		m_energyfile << endl;
	}

	size_t ngages = m_problem->get_simparams()->gage.size();
	if (ngages > 0) {
		string WaveGage_fn = open_data_file(m_WaveGagefile, "WaveGage");
		if (m_WaveGagefile) {
			m_WaveGagefile << "time";
			for (size_t gage = 0; gage < ngages; ++gage)
				m_WaveGagefile << "\tzgage" << gage;
			m_WaveGagefile << endl;
		}
	}

	// TODO only do this if object data writing is enabled
	size_t nbodies = m_problem->get_simparams()->numODEbodies;
	if (nbodies > 0) {
		string rbdata_fn = open_data_file(m_objectfile, "rbdata");
		if (m_objectfile) {
			m_objectfile << "time";
			for (size_t obj = 0; obj < nbodies; ++obj) {
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
		}

		string objforce_fn = open_data_file(m_objectforcesfile, "objectforces");
		if (m_objectfile) {
			m_objectfile << "time";
			for (size_t obj = 0; obj < nbodies; ++obj) {
				// computed forces
				m_objectfile << "\tComputed_F" << obj << "_X";
				m_objectfile << "\tComputed_F" << obj << "_Y";
				m_objectfile << "\tComputed_F" << obj << "_Z";
				// computed torques
				m_objectfile << "\tComputed_M" << obj << "_X";
				m_objectfile << "\tComputed_M" << obj << "_Y";
				m_objectfile << "\tComputed_M" << obj << "_Z";
				// applied forces
				m_objectfile << "\tApplied_F" << obj << "_X";
				m_objectfile << "\tApplied_F" << obj << "_Y";
				m_objectfile << "\tApplied_F" << obj << "_Z";
				// applied torques
				m_objectfile << "\tApplied_M" << obj << "_X";
				m_objectfile << "\tApplied_M" << obj << "_Y";
				m_objectfile << "\tApplied_M" << obj << "_Z";
			}
			m_objectfile << endl;
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
}

void
CommonWriter::write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints)
{ /* do nothing */ }

void
CommonWriter::write_energy(double t, float4 *energy)
{
	if (m_energyfile) {
		m_energyfile << t;
		uint fluid = 0;
		for (; fluid < m_problem->get_physparams()->numFluids; ++fluid)
			m_energyfile	<< "\t" << energy[fluid].x
							<< "\t" << energy[fluid].y
							<< "\t" << energy[fluid].z;
		m_energyfile << endl;
		m_energyfile.flush();
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
		m_WaveGagefile.flush();
	}
}

void
CommonWriter::write_objects(double t, Object const* const* bodies)
{
	if (m_objectfile) {
		m_objectfile << t;
		size_t nbodies = m_problem->get_simparams()->numODEbodies;
		for (size_t obj = 0; obj < nbodies; ++obj) {
			const dReal *cg = dBodyGetPosition(bodies[obj]->m_ODEBody);
			const dReal *quat = dBodyGetQuaternion(bodies[obj]->m_ODEBody);
			m_objectfile
				<< "\t" << cg[0] << "\t" << cg[1] << "\t" << cg[2]
				<< "\t" << quat[0] << "\t" << quat[1]
				<< "\t" << quat[2] << "\t" << quat[3];
		}
		m_objectfile << endl;
		m_objectfile.flush();
	}
}

void
CommonWriter::write_objectforces(double t, uint numobjects,
		const float3* computedforces, const float3* computedtorques,
		const float3* appliedforces, const float3* appliedtorques)
{
	if (m_objectforcesfile) {
		m_objectforcesfile << t;
		for (int i=0; i < numobjects; i++) {
			m_objectforcesfile << "\t" << i;
			m_objectforcesfile
				<< "\t" << computedforces[i].x
				<< "\t" << computedforces[i].y
				<< "\t" << computedforces[i].z;
			m_objectforcesfile
				<< "\t" << computedtorques[i].x
				<< "\t" << computedtorques[i].y
				<< "\t" << computedtorques[i].z;
			m_objectforcesfile
				<< "\t" << appliedforces[i].x
				<< "\t" << appliedforces[i].y
				<< "\t" << appliedforces[i].z;
			m_objectforcesfile
				<< "\t" << appliedtorques[i].x
				<< "\t" << appliedtorques[i].y
				<< "\t" << appliedtorques[i].z;
		}
		m_objectforcesfile << endl;
		m_objectforcesfile.flush();
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
	const SimParams *SP = m_problem->get_simparams();

	out << "Simulation parameters:" << endl;

	out << " deltap = " << m_problem->get_deltap() << endl;
	out << " sfactor = " << SP->sfactor << endl;
	out << " slength = " << SP->slength << endl;
	out << " kerneltype: " << SP->kerneltype << " (" << KernelName[SP->kerneltype] << ")" << endl;
	out << " kernelradius = " << SP->kernelradius << endl;
	out << " influenceRadius = " << SP->influenceRadius << endl;
	out << " initial dt = " << SP->dt << endl;
	out << " simulation end time = " << SP->tend << endl;
	out << " neib list construction every " << SP->buildneibsfreq << " iterations" << endl;
	out << " Shepard filter every " << SP->shepardfreq << " iterations" << endl;
	out << " MLS filter every " << SP->mlsfreq << " iterations" << endl;
	out << " adaptive time stepping " << ED[SP->dtadapt] << endl;
	if (SP->dtadapt)
		out << " safety factor for adaptive time step = " << SP->dtadaptfactor << endl;
	out << " XSP-> correction " << ED[SP->xsph] << endl;
	out << " SP-> formulation: " << SP->sph_formulation << " (" << SPHFormulationName[SP->sph_formulation] << ")" << endl;
	out << " viscosity type: " << SP->visctype << " (" << ViscosityName[SP->visctype] << ")" << endl;
	out << " moving boundaries " << ED[SP->mbcallback] << endl;
	out << " time-dependent gravity " << ED[SP->gcallback] << endl;
	out << " periodicity: " << SP->periodicbound << " (" << PeriodicityName[SP->periodicbound] << ")" << endl;
	out << " DEM: " << TF[SP->usedem] << endl;
#undef SP
}

void
CommonWriter::write_physparams(ostream &out)
{
	const SimParams *SP = m_problem->get_simparams();
	const PhysParams *PP = m_problem->get_physparams();

	out << "Physical parameters:" << endl;

#define g (PP->gravity)
	out << " gravity = (" << g.x << ", " << g.y << ", " << g.z << ") [" << length(g) << "] "
		<< (SP->gcallback ? "time-dependent" : "fixed") << endl;
#undef g
	out << " numFluids = " << PP->numFluids << endl;
	for (uint f = 0; f < PP->numFluids ; ++f) {
		out << " rho0[ " << f << " ] = " << PP->rho0[f] << endl;
		out << " B[ " << f << " ] = " << PP->bcoeff[f] << endl;
		out << " gamma[ " << f << " ] = " << PP->gammacoeff[f] << endl;
		out << " sscoeff[ " << f << " ] = " << PP->sscoeff[f] << endl;
		out << " sspowercoeff[ " << f << " ] = " << PP->sspowercoeff[f] << endl;
		out << " sound speed[ " << f << " ] = " << m_problem->soundspeed(PP->rho0[f],f) << endl;
	}

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

	out << " " << ViscosityName[SP->visctype] << " viscosity parameters:" << endl;
	if (SP->visctype == ARTVISC) {
		out << "\tartvisccoeff = " << PP->artvisccoeff << "" << endl;
		out << "\tepsartvisc = " << PP->epsartvisc << "" << endl;
	} else
		out << "\tkinematicvisc = " << PP->kinematicvisc << " (m^2/s)" << endl;
	if (SP->visctype == SPSVISC) {
		out << "\tSmagfactor = " << PP->smagfactor << endl;
		out << "\tkSPSfactor = " << PP->kspsfactor << endl;
	}
	out << "\tvisccoeff = " << PP->visccoeff << endl;

	if (SP->xsph) {
		out << " epsxsph = " << PP->epsxsph << endl;
	}

	if (SP->usedem) {
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
	out << " hosts: " << OP->num_hosts << endl;
	out << " saving " << ED[!OP->nosave] << endl;
	out << " GPUDirect " << ED[OP->gpudirect] << endl;
	out << " striping " << ED[OP->striping] << endl;
	out << " async network transfers " << ED[OP->asyncNetworkTransfers] << endl;
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
}


