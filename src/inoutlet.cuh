/* vim: set ft=cuda: */

// outlet coordinates: a point is in the outlet if its xyz coordinates
// are between the xyz min coordinates and the xyz max coordinates
__constant__ uint	d_outlets;
__device__	float4	d_outlet_min[MAXOUTLETS];
__device__	float4	d_outlet_max[MAXOUTLETS];
// vector as long as the outlet in the outflow direction
__device__	float4	d_outlet_disp[MAXOUTLETS];
// symmetry plane for the outlet ghosts
__device__	float4	d_outlet_plane[MAXOUTLETS];


#ifndef NO_INLET
// inlet coordinates: a point is in the inlet if its xyz coordinates
// are between the xyz min coordinates and the xyz max coordinates
__constant__ uint	d_inlets;
__device__	float4	d_inlet_min[MAXOUTLETS];
__device__	float4	d_inlet_max[MAXOUTLETS];
// displacement added to clone particles when they exit the inlet
__device__	float4	d_inlet_disp[MAXOUTLETS];
// assigned velocity inside the inlet
__device__	float4	d_inlet_vel[MAXOUTLETS];
#endif

/******************** Functions for inlet/outlet management *****************************/

// check if pos is inside outlet with given index
__device__ __forceinline__ bool
inside_outlet(int index, const float4 &pos)
{
	return
		pos.x >= d_outlet_min[index].x && pos.x <= d_outlet_max[index].x &&
		pos.y >= d_outlet_min[index].y && pos.y <= d_outlet_max[index].y &&
		pos.z >= d_outlet_min[index].z && pos.z <= d_outlet_max[index].z;
}

// find the outlet pos is in. returns -1 if pos is not inside an outlet
__device__ __forceinline__
int
find_outlet(float4 const& pos)
{
	int outlet = d_outlets;
	while (--outlet >= 0 && !inside_outlet(outlet, pos));
	return outlet;
}

#ifndef NO_INLET
// check if pos is inside inlet with given index
__device__ __forceinline__ bool
inside_inlet(int index, const float4 &pos)
{
	return
		pos.x >= d_inlet_min[index].x && pos.x <= d_inlet_max[index].x &&
		pos.y >= d_inlet_min[index].y && pos.y <= d_inlet_max[index].y &&
		pos.z >= d_inlet_min[index].z && pos.z <= d_inlet_max[index].z;
}

// find the inlet pos is in. returns -1 if pos is not inside an inlet
__device__ __forceinline__
int
find_inlet(float4 const& pos)
{
	int inlet = d_inlets;
	while (--inlet >= 0 && !inside_inlet(inlet, pos));
	return inlet;
}
#endif


/************************************************************************************************************/
