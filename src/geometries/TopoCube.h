/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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
/*
 * File:   TopoCube.h
 * Author: alexis
 *
 * Created on 14 juin 2008, 18:04
 */

#ifndef _TOPOCUBE_H
#define	_TOPOCUBE_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"

/*! The TopoCube represents a parallelepiped with a bottom described by a natural topography.
 *
 *  The topography (DEM = Digital Elevation Model) is described as a matrix of values that
 *  span the xy plane, sampled at regular intervals, with the first/last samples located
 *  on the edge of the box.
 *
 *  As such, the essential components for the definition of the DEM are:
 *  * information about the span covered by the DEM, i.e. the lengths of the sides in the x and y direction.
 *  * information about the sampling step in each direction (we need at least two samples in each direction),
 *  * the actual data.
 *  Conventionally, the x axis is mapped to the “east-west" direction,
 *  and the y axis is mapped to the “north-south” direction,
 *  with spatial coordinates increasing from  west to east, and from south to north.
 *
 *  Note: the DEM description format used internally by GPUSPH is vertex-based,
 *  in the sense that we hold the information about the height of each vertex of a regular mesh covering
 *  the domain. Some on-disk formats follow this convention (e.g. XYZ and VTS),
 *  while others do not: for example, the GRASS ASCII grid format holds cell-based information:
 *  the height is associated with the center of each DEM cell, rather than the vertex.
 *  When loading cell-based data, we give the user the choice on how to handle this discrepancy:
 *  * in RELAXED mode (the default), the DEM data is interpreted as vertex data,
 *    which corresponds to stretching the DEM so that the vertices of the edge cells
 *    are located at the north/south/east/west coordinates defined by the DEM metadata;
 *  * in STRICT mode, the DEM data is interpreted as cell data, and the DEM span is restricted
 *    by half a cell in each direction so that each data point is correctly georeferenced.
 *  Note that the discrepancy between the two modes is small:
 *  the scale factor is cols/(cols-1) in the x direction). and rows/(rows-1) in the y direction.
 *  In most applications this is going to be less than 1%.
 *
 *  The DEM can have optional geolocation data associated with it. This is not used by GPUSPH,
 *  but the user can access this information to georeference additional elements/objects with
 *  respect to the DEM. This information is preserved from the on-disk metadata (if present),
 *  so its interpretation with respect to edge/center information is up to the user.
 */
class TopoCube: public Object {
	private:
		Point	m_origin;
		Vector	m_vx, m_vy, m_vz;
		float*	m_dem;
		int		m_ncols, m_nrows;
		double	m_nsres, m_ewres;
		double	m_H;
		double	m_filling_offset;

		/* Geolocation data (optional) */
		double	m_north, m_south, m_east, m_west;
		double	m_voff; // vertical offset

		//! Find the height of the DEM at DEM-relative coordinates (x, y)
		/*! x should in the range [0, (ncols-1)*ewres], y in the range [0, (nrows-1)*nsres]
		 */
		double	DemInterpolInternal(const double x, const double y) const;

		//! Find the distance between the DEM and the point with DEM-relative coordinates (x, y, z)
		//! dx is the distance at which to prove for additional points when building the tangent plane
		double	DemDistInternal(const double x, const double y, const double z, const double dx) const;

	public:
		TopoCube(void);
		virtual ~TopoCube(void);

		/// return list of planes in implicit form as double4
		std::vector<double4> get_planes() const;

		void SetCubeDem(const float *dem,
				double sizex, double sizey, double H,
				int ncols, int nrows, double voff = 0, bool restrict = false);

		/* methods to retrieve the DEM geometry */
		int get_nrows()	const { return m_nrows; }
		int get_ncols()	const { return m_ncols; }
		double get_ewres() const { return m_ewres; }
		double get_nsres() const{ return m_nsres; }
		const Point& get_origin() const { return m_origin; }
		double get_H() const { return m_H; }
		Vector const& get_vx() const { return m_vx; }
		Vector const& get_vy() const { return m_vy; }
		Vector const& get_vz() const { return m_vz; }

		/* allows direct read-only access to the DEM data */
		const float *get_dem() const { return m_dem; }

		/*! set the height to which the fluid will be filled over the DEM */
		void SetCubeHeight(double H);
		/*! set the offset from the DEM starting from which the filling will be done */
		void SetFillingOffset(double dx);

		/* Geolocation data (optional) */
		void SetGeoLocation(double north, double south,
				double east, double west);

		double get_north()	{ return m_north; }
		double get_south()	{ return m_south; }
		double get_east()	{ return m_east; }
		double get_west()	{ return m_west; }
		double get_voff()	{ return m_voff; }

		//! Supported file formats
		enum Format {
			DEM_FMT_ASCII, ///< GRASS ASCII Grid format
			DEM_FMT_VTK, ///< Legacy ASCII VTK Structured Grid format
			DEM_FMT_XYZ ///< XYZ file with header
		};

		enum FormatOptions {
			RELAXED, STRICT
		};

		//! Load a topography from the file, given the file name and format
		//! \seealso TopoCube::Format
		static TopoCube* load_file(const char *fname, Format fmt, FormatOptions fmt_options);

		//! Format-specific implementation of topography loader
		template<Format>
		static TopoCube* load_file(const char *fname, FormatOptions fmt_options);

		/* Legacy loader names */
		static TopoCube* load_ascii_grid(const char *fname);
		static TopoCube* load_vtk_file(const char *fname);
		static TopoCube* load_xyz_file(const char *fname);

		using Object::SetPartMass;
		double SetPartMass(const double dx, const double rho) override;
		double Volume(const double dx) const override
		{
			return 0.0;
		}
		void SetInertia(const double dx) override
		{
			m_inertia[0] = 0.0;
			m_inertia[1] = 0.0;
			m_inertia[2] = 0.0;
		}

		void setEulerParameters(const EulerParameters &ep) override;
		void getBoundingBox(Point &output_min, Point &output_max) override;
		void shift(const double3 &offset) override;

		//! Fill a single face of the cube, optionally including its edges
		void FillBorder(PointVect&, const double dx, const int face_num, const bool fill_edges);

		void FillBorder(PointVect& points, const double dx) override
		{
			FillBorder(points, dx, 0, true);
			FillBorder(points, dx, 1, false);
			FillBorder(points, dx, 2, true);
			FillBorder(points, dx, 3, false);
		}

		void FillDem(PointVect&, const double);
		//! Find the height of the DEM at the given global coordinates
		double DemInterpol(const double, const double) const;
		double DemDist(const double, const double, const double, const double) const;

		int Fill(PointVect&, const double, const double, const bool, const bool);
		int Fill(PointVect& points, const double H, const double dx, const bool faces_filled)
		{
			return Fill(points, H, dx, faces_filled, true);
		}
		int Fill(PointVect& points, const double dx, const bool fill = true) override
		{
			return Fill(points, m_H, dx, false, fill);
		}

		void FillIn(PointVect& points, const double dx, const int layers) override
		{ throw std::runtime_error("TopoCube::FillIn not implemented !"); }

		bool IsInside(const Point&, const double) const override;

};

#endif	/* _CUBE_H */

