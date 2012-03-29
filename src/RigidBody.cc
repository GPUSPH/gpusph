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

#include "RigidBody.h"

int RigidBody::m_bodies_number = 0;


/// Constructor
RigidBody::RigidBody(void)
{
	m_body_number = m_bodies_number;
	m_bodies_number++;

	// Allocating storage for integrator
	m_ep = new EulerParameters[2];
	m_cg = new double[6];
	m_vel = new double[6];
	m_omega = new double[6];
	
	m_parts.reserve(1000);
	m_object = NULL;
	
	m_current_ep = &m_ep[0];
}


/// Destructor
RigidBody::~RigidBody(void)
{
	delete [] m_ep;
	delete [] m_cg;
	delete [] m_vel;
	delete [] m_omega;

	m_parts.clear();
}


/// Add new particles to the rigid body
/*! /param new_points : points to be added */
void 
RigidBody::AddParts(const PointVect & parts)
{
	for (int i=0; i < parts.size(); i++)
		m_parts.push_back(parts[i]);
}


/*! Setting inertial frame data  */
void RigidBody::SetInertialFrameData(const Point& cg, const double* SetInertia,
									const double mass, const EulerParameters& ep)
{
	m_cg[0] = cg(0);
	m_cg[1] = cg(1);
	m_cg[2] = cg(2);

	m_current_cg = cg;
	
	for (int i = 0; i < 3; i++)
		m_inertia[i] = SetInertia[i];

	m_mass = mass;

	m_ep[0] = ep;
	m_current_ep = &m_ep[0];
	m_ep[0].ComputeRot();
}


/*! Setting inertial frame data  */
void 
RigidBody::AttachObject(Object* object)
{
	m_object = object;
	object->GetInertialFrameData(&m_cg[0], m_mass, &m_inertia[0], m_ep[0]);
	m_current_cg.SetCoord(&m_cg[0]);
	m_current_ep = &m_ep[0];
	m_ep[0].ComputeRot();
}


/*! Translate rigid body particles */
void 
RigidBody::Translate(const Vector &v)
{
	for (int i=0; i < m_parts.size(); i++)
		m_parts[i] += v;
}


/*! Rotate rigid body particles */
void 
RigidBody::Rotate(const Point &center, const EulerParameters & rot)
{
	// TODO
}


/*! Setting initial values for integration */
void 
RigidBody::SetInitialValues(const Vector &init_vel, const Vector &init_omega)
{
	m_vel[0] = init_vel(0);
	m_vel[1] = init_vel(1);
	m_vel[2] = init_vel(2);

	m_omega[0] = init_omega(0);
	m_omega[1] = init_omega(1);
	m_omega[2] = init_omega(2);
}


PointVect&
RigidBody::GetParts(void)
{
	return m_parts;
}


void 
RigidBody::TimeStep(const float3 &force, const float3 &gravity, const float3 &global_torque, const int step,
						const double dt, float3 * cg, float3 * trans, float * steprot)
{
	if (step == 1) {
		double dt2 = dt/2.0;
		double dt4 = dt/4.0;
		EulerParameters& ep = m_ep[0];
		EulerParameters& ep_pred = m_ep[1];

		// ep.ComputeRot() is computed at step 2 or before start
		float3 torque = ep.TransposeRot(global_torque);

		m_omega[3] = m_omega[0] + ((double) torque.x - (m_inertia[2] - m_inertia[1])*m_omega[2]*m_omega[1])*dt2/m_inertia[0];
		m_omega[4] = m_omega[1] + ((double) torque.y - (m_inertia[0] - m_inertia[2])*m_omega[0]*m_omega[2])*dt2/m_inertia[1];
		m_omega[5] = m_omega[2] + ((double) torque.z - (m_inertia[1] - m_inertia[0])*m_omega[0]*m_omega[1])*dt2/m_inertia[2];

		ep_pred(0) = ep(0) + (-ep(1)*m_omega[0] - ep(2)*m_omega[1] - ep(3)*m_omega[2])*dt4;
		ep_pred(1) = ep(1) + (ep(0)*m_omega[0] - ep(3)*m_omega[1] + ep(2)*m_omega[2])*dt4;
		ep_pred(2) = ep(2) + (ep(3)*m_omega[0] + ep(0)*m_omega[1] + ep(1)*m_omega[2])*dt4;
		ep_pred(3) = ep(3) + (-ep(2)*m_omega[0] + ep(1)*m_omega[1] + ep(0)*m_omega[2])*dt4;

		ep_pred.Normalize();
		ep_pred.ComputeRot();
		ep_pred.StepRotation(ep, steprot);
				
		m_vel[3] = m_vel[0] + (force.x/m_mass + gravity.x)*dt2;
		m_vel[4] = m_vel[1] + (force.y/m_mass + gravity.y)*dt2;
		m_vel[5] = m_vel[2] + (force.z/m_mass + gravity.z)*dt2;

		double vxdt = m_vel[3]*dt2;
		double vydt = m_vel[4]*dt2;
		double vzdt = m_vel[5]*dt2;

		m_cg[3] = m_cg[0] + vxdt;
		m_cg[4] = m_cg[1] + vydt;
		m_cg[5] = m_cg[2] + vzdt;

		cg->x = (float) m_cg[3];
		cg->y = (float) m_cg[4];
		cg->z = (float) m_cg[5];

		trans->x = (float) vxdt;
		trans->y = (float) vydt;
		trans->z = (float) vzdt;

		m_current_cg.SetCoord(&m_cg[3]);
		m_current_ep = &ep_pred;
	}
	else if (step == 2) {
		double dt2 = dt/2.0;
		EulerParameters& ep = m_ep[0];
		EulerParameters& ep_pred = m_ep[1];

		// ep_pred.ComputeRot() is computed at step 1
		float3 torque = ep_pred.TransposeRot(global_torque);

		m_omega[0] = m_omega[0] + ((double) torque.x - (m_inertia[2] - m_inertia[1])*m_omega[5]*m_omega[4])*dt/m_inertia[0];
		m_omega[1] = m_omega[1] + ((double) torque.y - (m_inertia[0] - m_inertia[2])*m_omega[3]*m_omega[5])*dt/m_inertia[1];
		m_omega[2] = m_omega[2] + ((double) torque.z - (m_inertia[1] - m_inertia[0])*m_omega[3]*m_omega[4])*dt/m_inertia[2];

		EulerParameters ep0 = ep;
		
		ep(0) = ep(0) + (-ep_pred(1)*m_omega[0] - ep_pred(2)*m_omega[1] - ep_pred(3)*m_omega[2])*dt2;
		ep(1) = ep(1) + (ep_pred(0)*m_omega[0] - ep_pred(3)*m_omega[1] + ep_pred(2)*m_omega[2])*dt2;
		ep(2) = ep(2) + (ep_pred(3)*m_omega[0] + ep_pred(0)*m_omega[1] + ep_pred(1)*m_omega[2])*dt2;
		ep(3) = ep(3) + (-ep_pred(2)*m_omega[0] + ep_pred(1)*m_omega[1] + ep_pred(0)*m_omega[2])*dt2;

		ep.Normalize();
		ep.ComputeRot();
		ep.StepRotation(ep0, steprot);

		m_vel[0] = m_vel[0] + (force.x/m_mass + gravity.x)*dt;
		m_vel[1] = m_vel[1] + (force.y/m_mass + gravity.y)*dt;
		m_vel[2] = m_vel[2] + (force.z/m_mass + gravity.z)*dt;

		double vxdt = m_vel[0]*dt;
		double vydt = m_vel[1]*dt;
		double vzdt = m_vel[2]*dt;

		m_cg[0] = m_cg[0] + vxdt;
		m_cg[1] = m_cg[1] + vydt;
		m_cg[2] = m_cg[2] + vzdt;

		cg->x = (float)m_cg[0];
		cg->y = (float)m_cg[1];
		cg->z = (float)m_cg[2];

		trans->x = (float) (vxdt);
		trans->y = (float) (vydt);
		trans->z = (float) (vzdt);

		m_current_cg.SetCoord(&m_cg[0]);
		m_current_ep = &ep;
	}

}


void 
RigidBody::GetCG(float3& cg) const
{
	cg.x = (float) m_current_cg(0);
	cg.y = (float) m_current_cg(1);
	cg.z = (float) m_current_cg(2);
}


const Point& 
RigidBody::GetCG(void) const
{
	return m_current_cg;
}


const EulerParameters& 
RigidBody::GetEulerParameters(void) const
{
	return *m_current_ep;
}


void 
RigidBody::GLDraw(void) const
{
	if (m_object)
		m_object->GLDraw(*m_current_ep, m_current_cg);
}


void 
RigidBody::Write(const float t, FILE* fid) const
{
	EulerParameters &ep = *m_current_ep;
	fprintf(fid, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", m_body_number, t, m_current_cg(0), 
			m_current_cg(1), m_current_cg(2), ep(0), ep(1), ep(2), ep(3));
}