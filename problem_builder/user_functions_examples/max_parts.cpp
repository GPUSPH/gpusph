uint GenericProblem::max_parts(uint numpart)
{
  uint aNumParts = numpart;
  if ( PVAL( sph, particles_max_factor ) > 0 )
    aNumParts = (uint)((float)numpart*(PVAL( sph, particles_max_factor )));
  return aNumParts;
}

