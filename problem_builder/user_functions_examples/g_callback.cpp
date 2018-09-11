
float3 GenericProblem::g_callback(const double t)
{
#if ISENUM_EQ(physics,variable_gravity,disable)
  return XProblem::g_callback(t);
#else
	if(t > m_gtstart && t < m_gtend) {
		// Program your own gravity variation here
		// For example:
		physparams()->gravity=make_float3(2.*sin(9.8*(t-m_gtstart)), 0.0, -9.81f);
	} else {
		physparams()->gravity=make_float3(0.,0.,-9.81f);
	}
	return physparams()->gravity;
#endif
}
