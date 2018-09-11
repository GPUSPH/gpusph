void
GenericProblem::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
		const float3& force, const float3& torque, const KinematicData& initial_kdata,
		KinematicData& kdata, double3& dx, EulerParameters& dr)
{
	dx = make_double3(0.0);
	dr.Identity();
	kdata.avel = make_double3(0.0);
	if (t0 >= piston_tstart & t1 <= piston_tend) {
		kdata.lvel = make_double3(0.0, 0.0, piston_vel);
		dx.z = -piston_vel*(t0 - t1);
	} else {
		kdata.lvel = make_double3(0.0);
	}
}

