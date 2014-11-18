/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/* Support for import of STL mesh files */

#include <cfloat>
#include <cstring>

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdexcept>

#include "STLMesh.h"

using namespace std;

void
STLMesh::reset_bounds(void)
{
	m_minbounds.x = m_minbounds.y = m_minbounds.z = INFINITY;
	m_maxbounds.x = m_maxbounds.y = m_maxbounds.z = -INFINITY;
	m_minres = INFINITY;
	m_maxres = -INFINITY;
}

void
STLMesh::expand_bounds(const float4 &pt)
{
	m_minbounds.x = fmin(pt.x, m_minbounds.x);
	m_maxbounds.x = fmax(pt.x, m_maxbounds.x);
	m_minbounds.y = fmin(pt.y, m_minbounds.y);
	m_maxbounds.y = fmax(pt.y, m_maxbounds.y);
	m_minbounds.z = fmin(pt.z, m_minbounds.z);
	m_maxbounds.z = fmax(pt.z, m_maxbounds.z);
}

// update resolution considering a new triangle
void
STLMesh::update_resolution(const float3 v[3])
{
	// work in double precision
	// vector sides
	double3	d1 = make_double3(v[1].x - v[0].x, v[1].y - v[0].y, v[1].z - v[0].z),
			d2 = make_double3(v[2].x - v[1].x, v[2].y - v[1].y, v[2].z - v[1].z),
			d3 = make_double3(v[0].x - v[2].x, v[0].y - v[2].y, v[0].z - v[2].z);

	// squared lengths
	double	l1 = dot(d1, d1), l2 = dot(d2, d2), l3 = dot(d3, d3);

	// min and max
	double dmin = sqrt(fmin(l1, fmin(l2, l3)));
	double dmax = sqrt(fmax(l1, fmax(l2, l3)));

	m_minres = fmin(m_minres, dmin);
	m_maxres = fmax(m_maxres, dmax);
}

STLMesh::STLMesh(uint meshsize) :
	Object(),
	m_ODETriMeshData(0)
{
	reset_bounds();
	// we assume there will be about half as many vertices as triangles
	m_vertices.reserve(meshsize/2);
	m_triangles.reserve(meshsize);
	m_normals.reserve(meshsize);
}

STLMesh::~STLMesh(void)
{
	m_vertices.clear();
	m_triangles.clear();
	m_normals.clear();
}

STLMesh *
STLMesh::load_stl(const char *fname)
{
	// TODO support compressed and ASCII STL files
	char buf[81] = {0};
	STLMesh *stl = NULL;
	STLTriangle tr;

	memset(&tr, -1, sizeof(tr));

	ifstream fstl(fname, ios::binary);
	if (!fstl.good()) {
		stringstream err_msg;
		err_msg	<< "failed to open STL " << fname;

		throw runtime_error(err_msg.str());
	}

	fstl.read(buf, 80);

	if (!strncmp(buf, "solid", 5)) {
		stringstream err_msg;
		err_msg	<< "Unsupported ASCII STL " << fname;

		throw runtime_error(err_msg.str());
	}

	// TODO endianness
	uint32_t meshsize = 0;
	fstl.read((char *)&meshsize, 4);

	printf("STL %s (%s): %u triangles with size %zu\n",
		fname, buf, meshsize, sizeof(STLTriangle));

	stl = new STLMesh(meshsize);

	/* We load each triangle, into tr, and then add it to the mesh.
	 * Note that the STL format is packed, so it is actually not
	 * particularly efficient for load/store, as the STLTriangle
	 * struct has a natural size/alignment of 52 bytes, but the
	 * data (on disk) is only 50, so we couldn't load the whole
	 * bunch of triangles at once in a single array anyway. */

#define STL_TRIANGLE_BYTES 50
	for (uint32_t i = 0; i < meshsize; ++i) {
		fstl.read((char *)(&tr), STL_TRIANGLE_BYTES);
		stl->add(tr, i);
	}
	(stl->m_vmap).clear();
#undef STL_TRIANGLE_BYTES

	double3 minb = stl->get_minbounds();
	double3 maxb = stl->get_maxbounds();
	printf("STL %s loaded, %zu triangles, %zu normals, %zu vertices\n"
		"\tbounds (%g, %g, %g) -- (%g, %g, %g)\n",
		fname,
		(stl->m_triangles).size(),
		(stl->m_normals).size(),
		(stl->m_vertices).size(),
		minb.x, minb.y, minb.z,
		maxb.x, maxb.y, maxb.z);
	printf("\tresolution min %g, max %g\n", stl->get_minres(), stl->get_maxres());

	return stl;
}

/* adding a triangle to the mesh follows these steps:
 * + add the vertices that are new
 * + compute the vertex indices
 * + check that the normal is 'correct'
 * + update the boundaries of the mesh
 * + update the barycenter
 */
void
STLMesh::add(STLTriangle const& t, uint tnum)
{
	const float3 *v = t.vertex;
	float3 cnormal; // computed normal
	float4 normal; // normal that will be added
	float3 avg_pos; // barycenter of the triangle

	uint  vidx[3]; // indices of vertices in the array of vertices
	uint4 triangle;

	// update the min/max mesh resolution from this triangle
	update_resolution(v);

	// barycenter of the triangle
	avg_pos = (v[0] + v[1] + v[2])/3;

	// (unscaled) barycenter of the mesh
	m_barysum += avg_pos;

	// check normal
	cnormal = cross(v[1] - v[0], v[2] - v[1]);
	cnormal /= length(cnormal);

	// the stored normal is assumed to be 'correct' if it introduces an error of less than
	// FLT_EPSILON relative to the triangle barycenter
	bool normal_match = (length(cnormal - t.normal) < max(1,length(avg_pos)) * FLT_EPSILON);

	// we add the original normal if it matches,
	// our own if it doesn't
	if (!normal_match) {
		if (t.normal.x || t.normal.y || t.normal.z) {
			fprintf(stderr, "fixing normal for triangle %u:\n"
				"\t(%.8g, %.8g, %.8g)--(%.8g, %.8g, %.8g)--(%.8g, %.8g, %.8g)\n"
				"\t(%.8g, %.8g, %.8g) -> (%.8g, %.8g, %.8g) (err: %.8g)\n",
				tnum,
				t.vertex[0].x, t.vertex[0].y, t.vertex[0].z,
				t.vertex[1].x, t.vertex[1].y, t.vertex[1].z,
				t.vertex[2].x, t.vertex[2].y, t.vertex[2].z,
				t.normal.x, t.normal.y, t.normal.z,
				cnormal.x, cnormal.y, cnormal.z,
				length(t.normal - cnormal));
			throw;
		}
		normal.x = cnormal.x;
		normal.y = cnormal.y;
		normal.z = cnormal.z;
	} else {
		normal.x = t.normal.x;
		normal.y = t.normal.y;
		normal.z = t.normal.z;
	}
	m_normals.push_back(normal);

	// finally, add the missing the vertices to the
	// vertices list and find the indices of all vertices
	for (uint i = 0; i < 3; ++i) {
		float4 vi = make_float4(v[i]);
		vi.w = 0;
		VertMap::const_iterator f = m_vmap.find(vi);
		if (f == m_vmap.end()) {
			vidx[i] = m_vertices.size();
			m_vmap[vi] = vidx[i];
			m_vertices.push_back(vi);
			expand_bounds(vi);
		} else {
			vidx[i] = f->second;
		}
	}

	triangle.x = vidx[0];
	triangle.y = vidx[1];
	triangle.z = vidx[2];
	triangle.w = 0;
	m_triangles.push_back(triangle);

	m_barycenter(0) = avg_pos.x/get_meshsize();
	m_barycenter(1) = avg_pos.y/get_meshsize();
	m_barycenter(2) = avg_pos.z/get_meshsize();
}

/* FIXME might need something more sophisticated  */
double STLMesh::SetPartMass(const double dp, const double rho)
{
	double mass = dp*dp*dp*rho;
	SetPartMass(mass);
	return mass;
}

void STLMesh::SetPartMass(const double mass)
{
	m_barycenter(3) = mass;
	F4Vect::iterator f = m_vertices.begin();
	F4Vect::iterator e = m_vertices.end();
	for (; f != e; ++f)
		f->w = mass;
}

void STLMesh::FillBorder(PointVect& parts, double)
{
	// start by placing a particle on each vertex

	F4Vect::const_iterator f = m_vertices.begin();
	F4Vect::const_iterator e = m_vertices.end();
	for (; f != e; ++f) {
		parts.push_back(Point(*f));
	}
}

void STLMesh::ODEGeomCreate(dSpaceID ODESpace, const double dx, const double density)
{
	m_ODETriMeshData = dGeomTriMeshDataCreate();
	// TODO FIXME sanity checks on data type (use *Single1 if data is floats,
	// *Double1 if data is doubles)
	dGeomTriMeshDataBuildSingle(m_ODETriMeshData,
			&(m_vertices[0]), sizeof(m_vertices[0]), m_vertices.size(),
			&(m_triangles[0]), 3*m_triangles.size(), sizeof(m_triangles[0]));

	// use the default callbacks
	m_ODEGeom = dCreateTriMesh(ODESpace, m_ODETriMeshData, NULL, NULL, NULL);

	if (m_ODEBody) {
		/* Now we want to compute the body CG, mass and inertia tensor, assuming
		 * constant density. They are all computed by ODE for a generic mesh in
		 * dMassSetTrimesh(). For some obscure reason, ODE requires the CG to be
		 * at (0,0,0) in the object coordinate system for a correct inertia
		 * computation; however, we want dMassSetTrimesh() itself to estimated
		 * the CG. Therefore, we call it twice: the first time we'll read only
		 * the CG; the second, also the inertia. The mass should be practically
		 * identical in both calls.
		 *
		 * See: https://groups.google.com/d/msg/ode-users/SUQzotZNIZU/wMpXpXIk4MMJ
		 */

		// associate the geometry to the body
		dGeomSetBody(m_ODEGeom, m_ODEBody);

		// here we are interested only in the CG; inertia is wrong
		dMassSetTrimesh(&m_ODEMass, (dReal)density, m_ODEGeom);

		// save the CG in the local m_center class member
		m_center(0) = m_ODEMass.c[0];
		m_center(1) = m_ODEMass.c[1];
		m_center(2) = m_ODEMass.c[2];

		// CG != origin until now; for correct inertia computation, we shift the geometry to
		// make the CG coincide with the (ODE object local) origin
		dGeomSetOffsetPosition(m_ODEGeom, -m_ODEMass.c[0], -m_ODEMass.c[1], -m_ODEMass.c[2] );

		// compute again CG, mass, inertia (correct this time)
		dMassSetTrimesh(&m_ODEMass, (dReal)density, m_ODEGeom);

		// CG is now very close to zero, except for numerical leftovers which we manually reset
		m_ODEMass.c[0] = m_ODEMass.c[1] = m_ODEMass.c[2] = 0;

		// we worked on the m_ODEMass class member; tell ODE that's the new ODEMass
		dBodySetMass(m_ODEBody, &m_ODEMass);

		// once the inertia matrix is correctly computed, we can move back the ODE obj to its global position
		dBodySetPosition(m_ODEBody,m_center(0), m_center(1), m_center(2));

		// store inertia and mass in local class members
		m_inertia[0] = m_ODEMass.I[0];
		m_inertia[1] = m_ODEMass.I[5];
		m_inertia[2] = m_ODEMass.I[10];
		m_mass = m_ODEMass.mass;

		// reset the numerical leftovers in inertia matrix
		m_ODEMass.I[1] = m_ODEMass.I[2] = m_ODEMass.I[4] = 0;
		m_ODEMass.I[6] = m_ODEMass.I[8] = m_ODEMass.I[9] = 0;

		// show final computed position, CG, mass, inertia, bbox
		ODEPrintInformation();
	}
	else {
		dGeomSetPosition(m_ODEGeom, m_center(0), m_center(1), m_center(2));
		dGeomSetRotation(m_ODEGeom, m_ODERot);
	}
}

void STLMesh::ODEBodyCreate(dWorldID ODEWorld, const double dx, const double density, dSpaceID ODESpace)
{
	const double m_lx = m_maxbounds.x - m_minbounds.x;
	const double m_ly = m_maxbounds.y - m_minbounds.y;
	const double m_lz = m_maxbounds.z - m_minbounds.z;

	m_ODEBody = dBodyCreate(ODEWorld);

	dMassSetZero(&m_ODEMass);

	if (ODESpace)
		ODEGeomCreate(ODESpace, dx, density);
	else {
		// In case we don't have a geometry we can make ODE believe it is a box.
		// This works because all we need in this case are center of gravity and the
		// tensor of inertia together with the mass to compute the movement of the object.
		dMassSetBoxTotal(&m_ODEMass, m_mass, m_lx + dx, m_ly + dx, m_lz + dx);
		dBodySetMass(m_ODEBody, &m_ODEMass);
		dBodySetPosition(m_ODEBody, m_center(0), m_center(1), m_center(2));
		dBodySetRotation(m_ODEBody, m_ODERot);
	}
}

/* TODO */

int STLMesh::Fill(PointVect&, double, bool)
{}

void STLMesh::Fill(PointVect&, const double)
{}

bool STLMesh::IsInside(const Point&, double) const
{}

double STLMesh::Volume(const double dx) const
{}

void STLMesh::SetInertia(double)
{}

void STLMesh::SetInertia(const double*)
{}
