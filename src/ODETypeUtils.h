/*
 * ODETypeUtils.h
 *
 *  Created on: 14 juin 2012
 *      Author: alexis
 */

#ifndef _ODETYPEUTILS_H
#define _ODETYPEUTILS_H


#include "ode/ode.h"
#include "common_types.h"

float3 make_float3(const dVector3 v)
{
	return make_float3(v[0], v[1], v[2]);
}


float4 make_float4(const dVector4 v)
{
	return make_float4(v[0], v[1], v[2], v[3]);
}


void make_dvector3(const float3 & v, dVector3 vec)
{
	vec[0] = v.x;
	vec[1] = v.y;
	vec[2] = v.z;
}


void make_dvector4(const float4 & v, dVector4 vec)
{
	vec[0] = v.x;
	vec[1] = v.y;
	vec[2] = v.z;
	vec[3] = v.w;
}
#endif /* _ODETYPEUTILS_H */
