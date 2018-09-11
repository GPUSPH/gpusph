void
GenericProblem::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
      const float3& force, const float3& torque, const KinematicData& initial_kdata,
      KinematicData& kdata, double3& dx, EulerParameters& dr)
{
  // Create two arrays to store the moving bodies' velocities
  // They could be of lower size
  // Here we create these arrays as many times as the callback is called
  // This should be improved
#ifdef GPUSPH_special_boundary_SECTIONS
  #if ISDEF(special_boundary,start_time_VALS) || ISDEF(special_boundary,end_time_VALS)
    double3 translationVelocity[NB_SECTIONS(special_boundary)];
    double3 rotationVelocity[NB_SECTIONS(special_boundary)];
    // Initialisation
    for (uint i = 0; i < NB_SECTIONS(special_boundary); i++) {
      translationVelocity[i] = make_double3(0.0,0.0,0.0);
      rotationVelocity[i] = make_double3(0.0,0.0,0.0);
    }

    //SET_OBJECTS_VELOCITIES
    const char* aBndType[] = { PSTRVALS( special_boundary, type ) };
    double vel_x[NB_SECTIONS(special_boundary)] = {
  #if ISDEF(special_boundary,translation_vel_x_VALS)
      GPUSPH_special_boundary_translation_vel_x_VALS__ };
  #else
      0 };
  #endif

    double vel_y[NB_SECTIONS(special_boundary)] = {
  #if ISDEF(special_boundary,translation_vel_y_VALS)
      GPUSPH_special_boundary_translation_vel_y_VALS__ };
  #else
      0 };
  #endif

    double vel_z[NB_SECTIONS(special_boundary)] = {
  #if ISDEF(special_boundary,translation_vel_z_VALS)
      GPUSPH_special_boundary_translation_vel_z_VALS__ };
  #else
      0 };
  #endif

      double rot_x[NB_SECTIONS(special_boundary)] = {
  #if ISDEF(special_boundary,rotation_vel_x_VALS)
      GPUSPH_special_boundary_rotation_vel_x_VALS__ };
  #else
      0 };
  #endif

    double rot_y[NB_SECTIONS(special_boundary)] = {
  #if ISDEF(special_boundary,rotation_vel_y_VALS)
      GPUSPH_special_boundary_rotation_vel_y_VALS__ };
  #else
      0 };
  #endif

    double rot_z[NB_SECTIONS(special_boundary)] = {
  #if ISDEF(special_boundary,rotation_vel_z_VALS)
      GPUSPH_special_boundary_rotation_vel_z_VALS__ };
  #else
      0 };
  #endif

    for (uint i=0; i<NB_SECTIONS(special_boundary); i++) {
      if ( aBndType[i][0] == 'm' ) // moving_body
      {
        translationVelocity[i] = make_double3(vel_x[i], vel_y[i], vel_z[i]);
        rotationVelocity[i] = make_double3(rot_x[i],rot_y[i],rot_z[i]);
      }
    }

    bool isTime = true;
    #if ISDEF(special_boundary,start_time_VALS)
    isTime = (t1 >= m_bndtstart[index]);
    #endif
    #if ISDEF(special_boundary,end_time_VALS)
    isTime = isTime && (t1 <= m_bndtend[index]);
    #endif
    if ( isTime )
    {
      // Set the body's translation velocity
      kdata.lvel = translationVelocity[index];
      dx = (t1-t0)*translationVelocity[index];
      // Set the body's rotation velocity
      kdata.avel = rotationVelocity[index];
      EulerParameters dqdt = 0.5*EulerParameters(kdata.avel)*kdata.orientation;
      dr = EulerParameters::Identity() + (t1-t0)*dqdt*kdata.orientation.Inverse();
      dr.Normalize();
      kdata.orientation = kdata.orientation + (t1 - t0)*dqdt;
      kdata.orientation.Normalize();
    }
    else
  #endif

  {
    kdata.lvel = make_double3(0.0,0.0,0.0);
    dx = make_double3(0.0,0.0,0.0);
    kdata.avel = make_double3(0.0,0.0,0.0);
    kdata.orientation = initial_kdata.orientation;
    dr.Identity();
  }
#endif

}


