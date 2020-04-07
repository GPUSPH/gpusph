/*  Copyright (c) 2013-2017 INGV, EDF, UniCT, JHU

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

/* Support for import of STL mesh files */

#include <cfloat>
#include <cstring>

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdexcept>

#include <stdio.h>

#include "chrono_select.opt"
#if USE_CHRONO == 1
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChMeshFileLoader.h"
#include "chrono/fea/ChNodeFEAxyz.h"
#endif

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
	Object()
{
	reset_bounds();
	// we assume there will be about half as many vertices as triangles
	m_vertices.reserve(meshsize/2);
	m_triangles.reserve(meshsize);
	m_normals.reserve(meshsize);

	m_origin = Point(0,0,0);
	m_center = Point(0,0,0);
	m_ep.ComputeRot();
	// TODO : default initialization of  chrono triangle mesh is needed. the function used to have m_ODETriMeshData = 0;

	m_objfile = "";
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

// FIXME we are uploading the chrono fea mesh. This function cannot be simply called load_Tet file
STLMesh *
STLMesh::load_TetFile(const char *nodes, const char *elems, const double z_frame)
{

	STLMesh *stl = NULL;

#if USE_CHRONO == 1
	cout << "loading tet file" << endl;
	stl = new STLMesh(2); // use 2 just to init. It will be resized once the number of nodes is known

	auto mmaterial = chrono_types::make_shared<::chrono::fea::ChContinuumElastic>();
	// TODO FIXME set material from problem
	mmaterial->Set_E(10e6);
	mmaterial->Set_v(0.4);
	//mmaterial->Set_RayleighDampingK(0.01);
	mmaterial->Set_density(1100);


	stl->m_fea_mesh = chrono_types::make_shared<::chrono::fea::ChMesh>();
	//the Chrono function FromTetGenFile accepts shared pointers
	//shared_ptr<::chrono::fea::ChMesh> mesh_sh(ch_mesh);

	// We load the mesh from file using chrono methods
	try {
		::chrono::fea::ChMeshFileLoader::FromTetGenFile(stl->m_fea_mesh, ::chrono::GetChronoDataFile(nodes).c_str(),
			::chrono::GetChronoDataFile(elems).c_str(), mmaterial);
	} catch (::chrono::ChException ch_err) {
		::chrono::GetLog() << ch_err.what();
		throw;
	}

	cout << "building STL mesh" << endl;
	// now we take the mesh nodes in order to create the associated SPH particles
	//vector<shared_ptr<::chrono::fea::ChNodeFEAxyz>>	mesh_nodes(dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyz>(mesh_sh->GetNodes()));
	//vector<shared_ptr<::chrono::fea::ChNodeFEAbase>>::iterator nodes_it;

	uint nodes_num = stl->m_fea_mesh->GetNnodes();

	stl->m_vertices.resize(nodes_num);

	float4 coords; //FIXME should be double precision
	shared_ptr<::chrono::fea::ChNodeFEAxyz> node;

	for (int i = 0; i < nodes_num; ++i) {

		node = dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyz>(stl->m_fea_mesh->GetNode(i));

		if (!node) throw std::runtime_error("Error: unsupported node type in mesh");

		coords.x = node->GetPos().x();
		coords.y = node->GetPos().y();
		coords.z = node->GetPos().z();

		stl->m_vertices[i] = coords;

		cout << "Added FEA node " << coords.x << " " << coords.y << " " << coords.z << endl;
		//ground nodes to the frame
		if(coords.z > z_frame) node->SetFixed(true);

	}

	cout << "loaded tet file" << endl;
#else
	cout << "ERROR: cannot load Tet files without chrono" << endl;
#endif
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
	bool normal_match = (length(cnormal - t.normal) < max(1.0f, length(avg_pos)) * FLT_EPSILON);

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

	// here assign m_minbounds to m_origin to make the lower corner of the bbox align with world origin
	// m_origin = Point(m_minbounds);
	m_center = m_origin + Point(m_minbounds + (m_maxbounds - m_minbounds) / 2.0);
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
	m_center(3) = mass;
	F4Vect::iterator f = m_vertices.begin();
	F4Vect::iterator e = m_vertices.end();
	for (; f != e; ++f)
		f->w = mass;
}

void STLMesh::FillBorder(PointVect& parts, double)
{
	// No OBJ file: a STL mesh was loaded. Iterate on triangles.
	if (m_objfile == "") {
		// place a particle on each vertex
		F4Vect::const_iterator f = m_vertices.begin();
		F4Vect::const_iterator e = m_vertices.end();
		for (; f != e; ++f) {
			// translate from STL coords to GPUSPH ones
			Point p_in_global_coords = Point(*f) + m_origin;
			// rotate around m_center
			Point rotated = m_ep.Rot(p_in_global_coords - m_center) + m_center;
			parts.push_back(rotated);
		}
	} else {
		// OBJ file: reload the file, apparently easier than getting the list of triangles from Chrono
		// Inspired to http://goo.gl/qcMOrZ
		float cx, cy, cz;
		uint vcount = 0;
		FILE * file = fopen(m_objfile.c_str(), "r");
		if( file == NULL )
			throw runtime_error("STLMesh::Fill OBJ file unreadable!");
		while( 1 ){
			char lineHeader[128];
			// read the first word of the line
			int res = fscanf(file, "%s", lineHeader);
			// file end?
			if (res == EOF)
				break;
			if ( strcmp( lineHeader, "f" ) == 0 )
				break;
			if ( strcmp( lineHeader, "v" ) == 0 ){
				fscanf(file, "%f %f %f\n", &cx, &cy, &cz );
				// create point
				Point p_in_global_coords = Point(cx, cy, cz) + m_origin;
				// rotate around m_center
				Point rotated = m_ep.Rot(p_in_global_coords - m_center) + m_center;
				// reset point mass
				rotated(3) = m_center(3);
				// enqueue it
				parts.push_back(rotated);
				vcount++;
			} else {
				char ignore[1024];
				fgets(ignore, sizeof(ignore), file);
			}
		} // while(1)
	}
}

// load OBJ file only to update bbox
void STLMesh::loadObjBounds()
{
	float cx, cy, cz;
	FILE * file = fopen(m_objfile.c_str(), "r");
	if( file == NULL )
		throw runtime_error("STLMesh::Fill OBJ file unreadable!");
	while( 1 ){
		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		// file end?
		if (res == EOF)
			break;
		// faces section?
		if ( strcmp( lineHeader, "f" ) == 0 )
			break;
		// vertex?
		if ( strcmp( lineHeader, "v" ) == 0 ){
			fscanf(file, "%f %f %f\n", &cx, &cy, &cz );
			// create point
			Point p_in_global_coords = Point(cx, cy, cz, m_center(3)) + m_origin;
			// rotate around m_center
			Point rotated = m_ep.Rot(p_in_global_coords - m_center) + m_center;
			// update bounds
			expand_bounds( make_float4(rotated(0), rotated(1), rotated(2), 0) );
		} else {
			// anything else? (vt, normals, etc.) ignore it
			char ignore[1024];
			fgets(ignore, sizeof(ignore), file);
		}
	} // while(1)
}

int STLMesh::Fill(PointVect&, double, bool)
{ throw runtime_error("STLMesh::Fill not implemented yet"); }

void STLMesh::Fill(PointVect&, const double)
{ throw runtime_error("STLMesh::Fill not implemented yet"); }

void STLMesh::FillIn(PointVect&, const double, const int)
{ throw runtime_error("STLMesh::Fill not implemented yet"); }

// NOTE: checking the bounding box (incl. orientation), not the actual mesh space
bool STLMesh::IsInside(const Point& p, double dx) const
{
	const Point rotated_point = m_ep.TransposeRot(p - m_center);
	const Point half_size = Point( (m_maxbounds - m_minbounds) / 2.0 + dx );

	bool inside = true;
	for (uint coord = 0; coord < 3; coord++)
		if ( fabs(rotated_point(coord)) >= half_size(coord) )
			inside =  false;

	return inside;
}

double STLMesh::Volume(const double dx) const
{
	const double dp_offset = 0; // or: dx
	const double m_lx = m_maxbounds.x - m_minbounds.x + dp_offset;
	const double m_ly = m_maxbounds.y - m_minbounds.y + dp_offset;
	const double m_lz = m_maxbounds.z - m_minbounds.z + dp_offset;
	return (m_lx * m_ly * m_lz);
}

void STLMesh::SetInertia(double)
{}

void STLMesh::SetInertia(const double*)
{}

// set the given EulerParameters
void STLMesh::setEulerParameters(const EulerParameters &ep)
{
	m_ep = ep;
	// Before doing any rotation with Euler parameters the rotation matrix associated with
	// m_ep is computed.
	m_ep.ComputeRot();

	// TODO FIXME: check if here should make something, or applying m_ODERot upon body creation is enough
	// Should probably update m_center and m_barycenter
	// m_center = m_origin + 0.5*m_ep.Rot(Vector(m_lx, m_ly, m_lz));
	printf("WARNING: STLMesh::setEulerParameters() is incomplete, rotation might be wrong\n");
}

// get the object bounding box
void STLMesh::getBoundingBox(Point &output_min, Point &output_max)
{
	output_min = m_minbounds - make_double3(m_origin(0), m_origin(1), m_origin(2));
	output_max = output_min + Vector(m_maxbounds - m_minbounds);
}

void STLMesh::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_center += poff;
	m_origin += poff;
	// NOTE: not shifting m_barycenter, since it is in mesh coordinates
}

#if USE_CHRONO == 1
void STLMesh::BodyCreate(::chrono::ChSystem *bodies_physical_system, const double dx, const bool collide,
			const ::chrono::ChQuaternion<> & orientation_diff)
{
	if (m_objfile == "")
		throw std::runtime_error("Object::BodyCreate called but no obj file specified in constructor!");

	/* NOTE
	 * Volume() computes the volume of the bounding box and not the actual one. for primitive shapes
	 * we use volume and mass to get the density and Chrono uses the density to set the mass. In
	 * GPUSPH problems, who loads a mesh usually directly knows its mass. So we create the body with
	 * a standard density value and after that we explicitly set the mass. Mass / density will be
	 * inconsistent only between body creation and SetMass.
	 */

	// Creating a new Chrono object. Parames: filename, density, compute_mass, collide...)
	m_body = chrono_types::make_shared< ::chrono::ChBodyEasyMesh > (m_objfile, 1000, false, collide);

	// retrieve the bounding box
	::chrono::ChVector<> bbmin, bbmax;
	m_body->GetTotalAABB(bbmin, bbmax);
	expand_bounds( make_float4( bbmin.x(), bbmin.y(), bbmin.z(), 0 ) );
	expand_bounds( make_float4( bbmax.x(), bbmax.y(), bbmax.z(), 0 ) );

	m_body->SetPos(::chrono::ChVector<>(m_center(0), m_center(1), m_center(2)));
	m_body->SetRot(orientation_diff*m_ep.ToChQuaternion());

	m_body->SetMass(m_mass);
	// Set custom inertia, if given. TODO: we should check if Chrono needs any explicit method call
	// to update the inertia after SetMass has been called.
	if (isfinite(m_inertia[0]) && isfinite(m_inertia[1]) && isfinite(m_inertia[2]))
		m_body->SetInertiaXX(::chrono::ChVector<>(m_inertia[0], m_inertia[1], m_inertia[2]));

	m_body->SetCollide(collide);
	m_body->SetBodyFixed(m_isFixed);

	// Add the body to the physical system
	bodies_physical_system->AddBody(m_body);
}
#endif
