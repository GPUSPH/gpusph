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

#ifndef _STLMESH_H
#define _STLMESH_H

#include <stdint.h>

#include <vector>
#include <map>
#include <string>

#include "Object.h"

/* A triangle in the mesh. It features a normal,
 * three vertices and a generic unsigned integer attribute
 * (typically used for material/color, which we don't use).
 * The member order matches what is found in a binary file,
 * allowing for quick loads.
 */
typedef struct STLTriangle {
	float3 normal;
	float3 vertex[3];
	uint16_t attribute;
} STLTriangle;

/* The STL format repeats vertices, which is extremely
 * inefficient. Our approach instead is to store all
 * information in three different arrays:
 *   * an array of vertices (without repetitions)
 *   * an array of triangles, expressed as
 *     triples of indices in the array of vertices
 *   * an array of normals
 */

// vector of float4s, used both for vertices
// and normals.
// TODO check efficiency of float4 vs float3
typedef std::vector<float4> F4Vect;
// vector of uint4s, used for the triangles.
// TODO check efficiency of uint4 vs uint3
typedef std::vector<uint4> U4Vect;

// During insertion, we will actually do a lot of look-ups
// to deduplicate the vertex array, so we store them
// in a map instead. We need to define a comparator
// for floa4s though
// TODO should be an unordered_map, requires C++11
// instead, we have to define a float4 comparator

// TODO templatize, might be useful in other cases
class F4Lexical {
public:
	bool operator()(float4 const& a, float4 const& b) const {
		if (a.x < b.x) return true;
		if (a.x > b.x) return false;
		if (a.y < b.y) return true;
		if (a.y > b.y) return false;
		if (a.z < b.z) return true;
		if (a.z > b.z) return false;
		if (a.w < b.w) return true;
		return false;
	}
};

// the actual map
typedef std::map<float4, uint, F4Lexical> VertMap;

class STLMesh: public Object {
private:
	F4Vect m_vertices; // vertices of triangles
	U4Vect m_triangles; // triangles
	F4Vect m_normals; // normals

	// obj filename for chrono
	std::string m_objfile;

	// insertion-time only
	VertMap m_vmap; // vertex map for dedup

	// sum of the barycenters of all triangles.
	// to get the actual barycenter, divide by the
	// number of triangles
	float3 m_barysum;

	// actual barycenter
	// NOTE: barycenter is in mesh coordinates
	Point m_barycenter;

	// origin = position of the smallest corner in world coordinates
	// NOTE: m_origin and m_center are in GPUSPH world coordinates
	Point m_origin;

	// minimum coordinates
	double3	m_minbounds, m_maxbounds;

//	// ODE-related stuff
//	dTriMeshDataID m_ODETriMeshData;
	// TODO: Chrono triangle mesh need to be added here

	// minimum and maximum distance between vertices
	double m_minres, m_maxres;

	void reset_bounds(void); // reset the bounds

	// expand the bounds to include the given point
	void expand_bounds(const float4&);
	// update the resolution considering the given triangle
	void update_resolution(const float3[3]);

	// add an STLTriangle to the mesh
	void add(STLTriangle const& tr, uint tnum);

public:
	STLMesh(uint meshsize = 0);
	virtual ~STLMesh(void);

	size_t get_meshsize(void) const
	{ return m_triangles.size(); }

	double3 const& get_minbounds(void) const
	{ return m_minbounds; }

	double3 const& get_maxbounds(void) const
	{ return m_maxbounds; }

	double get_minres(void) const
	{ return m_minres; }

	double get_maxres(void) const
	{ return m_maxres; }

	void setObjectFile(std::string fname) {m_objfile = fname;}

	static STLMesh *load_stl(const char *fname);

	// load OBJ file only to update bbox
	void loadObjBounds();

	void FillBorder(PointVect&, double);
	int Fill(PointVect&, double, bool);
	void Fill(PointVect&, const double);
	void FillIn(PointVect &, const double, const int);
	bool IsInside(const Point&, double) const;

	void setEulerParameters(const EulerParameters &ep);
	void getBoundingBox(Point &output_min, Point &output_max);
	void shift(const double3 &offset);

	double SetPartMass(const double, const double);
	void SetPartMass(const double);
	double Volume(const double dx) const;
	void SetInertia(double);
	void SetInertia(const double*);

#if USE_CHRONO == 1
		void BodyCreate(::chrono::ChSystem *bodies_physical_system, const double dx, const bool collide,
			const ::chrono::ChQuaternion<> & orientation_diff);
#else
		void BodyCreate(void *p1, const double p2, const bool p3)
		{ Object::BodyCreate(p1, p2, p3); }
#endif
};

#endif // _STLMESH_H
