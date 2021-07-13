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

#include <cstring> // memcpy
#include <climits>

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdexcept>

#include "TopoCube.h"
#include "Point.h"
#include "Vector.h"
#include "Rect.h"

#include "fastdem_select.opt"

using namespace std;

TopoCube::TopoCube(void)
{
	m_origin = Point(0, 0, 0);
	m_vx = Vector(0, 0, 0);
	m_vy = Vector(0, 0, 0);
	m_vz = Vector(0, 0, 0);

	m_dem = NULL;

	m_ncols = m_nrows = 0;
	m_nsres = m_ewres = NAN;
	m_H = 0;
	m_filling_offset = NAN;

	m_north = m_south = m_east = m_west = NAN;
	m_voff = 0;
}


TopoCube::~TopoCube(void)
{
	delete [] m_dem;
}

void TopoCube::SetCubeHeight(double H)
{
	m_H = H;
	m_vz = Vector(0.0, 0.0, H);
}

void TopoCube::SetFillingOffset(double dx)
{
	m_filling_offset = dx;
}

void TopoCube::SetGeoLocation(double north, double south,
		double east, double west)
{
	m_north = north;
	m_south = south;
	m_east = east;
	m_west = west;
}

/* Add wall planes to the planes array; the
 * given array should be able to hold at least 4 elements each
 */
vector<double4> TopoCube::get_planes() const
{

	vector<double4> planes;
	// north wall
	planes.push_back( make_double4(0, 1.0, 0, 0) );
	// south wall
	planes.push_back( make_double4(0, -1.0, 0, m_vy(1)) );
	// west wall
	planes.push_back( make_double4(1.0, 0, 0, 0) );
	// east wall
	planes.push_back( make_double4(-1.0, 0, 0, m_vx(0)) );

	return planes;
}

/* set the cube DEM to have
 * sizex in the x (east-west) direction,
 * sizey in the y (north-south) direction,
 * 'wall' height H,
 * ncols samples in the x direction,
 * nrows samples in the y direction,
 * associated height array dem.
 *
 * The array is a linearized 2D array, standard C layout
 * (row-major) with row 0 holding the southernmost samples
 * and row nrows-1 hoding the northernmost samples.
 * Column 0 holds the westernmost samples, column ncols-1
 * holds the easternmost samples.
 *
 * An optional vertical offset to be added to all values
 * can also be specified (the wall height H is _not_
 * corrected).
 *
 * If the optional restrict parameter is set to true,
 * the x and y size are reduced by an amount equal to
 * the corresponding resolution. This is used to support
 * DEM formats whose georeferencing information is cell-based
 * rather than vertex-based.
 */
void TopoCube::SetCubeDem(const float *dem,
		double sizex, double sizey, double H,
		int ncols, int nrows, double voff, bool restrict)
{
	// On first use of SetCubeDem, print if we're using the fast or accurate DEM interpolation
	static bool logged_fastdem_option = false;
	if (!logged_fastdem_option) {
		printf("DEM will use %s interpolation\n", (FASTDEM ? "fast" : "symmetrized"));
	}

	// Due to our usage, ncols and nrows must be at least 2
	if (ncols < 2)
		throw std::invalid_argument("DEM must have at least two datapoints in the horiziontal direction");
	if (nrows < 2)
		throw std::invalid_argument("DEM must have at least two datapoints in the vertical direction");

	m_ncols = ncols;
	m_nrows = nrows;
	m_ewres = sizex/(ncols-1);
	m_nsres = sizey/(nrows-1);

	m_origin = Point(0, 0, 0, 0);
	m_vx = Vector(sizex - (restrict ? m_ewres : 0.0), 0.0, 0.0);
	m_vy = Vector(0.0, sizey - (restrict ? m_nsres : 0.0) ,0.0);
	SetCubeHeight(H);

	m_voff = voff;

	size_t numels = ncols*nrows;
	m_dem = new float[numels];

	if (voff) {
		for (uint i = 0; i < numels; ++i) {
			m_dem[i] = dem[i] + voff;
		}
	} else {
		/* quicker if no offset must be applied */
		memcpy(m_dem, dem, sizeof(float)*numels);
	}
}

/*! Create a TopoCube from a GRASS ASCII Grid file
 *
 * Supported format options are FormatOptions are either RELAXED or STRICT.
 *
 * In RELAXED mode, the data is interpreted as vertex data, stretching it to cover the defined domain
 * in STRICT mode, the data is interpreted as cell data, and the domain is restricted by half a cell
 * in each direction.
 * */
template<>
TopoCube *TopoCube::load_file<TopoCube::DEM_FMT_ASCII>(const char* fname, FormatOptions fmt_options)
{
	ifstream fdem(fname);
	if (!fdem.good()) {
		stringstream err_msg;
		err_msg	<< "failed to open DEM " << fname;

		throw runtime_error(err_msg.str());
	}

	string s;

	double north, south, east, west;
	int nrows, ncols;

	for (int i = 1; i <= 6; i++) {
		fdem >> s;
		if (s.find("north:") != string::npos) fdem >> north;
		else if (s.find("south:") != string::npos) fdem >> south;
		else if (s.find("east:") != string::npos) fdem >> east;
		else if (s.find("west:") != string::npos) fdem >> west;
		else if (s.find("cols:") != string::npos) fdem >> ncols;
		else if (s.find("rows:") != string::npos) fdem >> nrows;
	}

	double zmin = NAN, zmax = NAN;

	float *dem = new float[ncols*nrows];

	double z;

	// DEM data is stored on disk with the north as the first row,
	// but we want south to be the first array data
	for (int row = nrows-1; row >= 0; --row) {
		for (int col = 0; col < ncols; ++col) {
			fdem >> z;
			zmax = max(z, zmax);
			zmin = min(z, zmin);
			dem[row*ncols+col] = z;
		}
	}
	fdem.close();

	TopoCube *ret = new TopoCube();

	ret->SetCubeDem(dem, east-west, north-south, zmax-zmin,
		ncols, nrows, -zmin, fmt_options == STRICT);

	ret->SetGeoLocation(north, south, east, west);

	delete [] dem;

	return ret;
}

TopoCube *TopoCube::load_ascii_grid(const char *fname)
{ return load_file<DEM_FMT_ASCII>(fname, RELAXED); }

/* Create a TopoCube from a VTK Structure Points datafile. Format options are ignored */
template<>
TopoCube *TopoCube::load_file<TopoCube::DEM_FMT_VTK>(const char* fname, FormatOptions /* fmt_options */)
{
	string s;
	stringstream err_msg;

	double north(NAN), south(NAN), east(NAN), west(NAN);
	double nsres(NAN), ewres(NAN), zbase(NAN);
	int nrows(0), ncols(0), nz(0), nels(0);
	double z(NAN), zmin(NAN), zmax(NAN);

	ifstream fdem(fname);

	if (!fdem.good()) {
		err_msg	<< "failed to open DEM " << fname;
		throw runtime_error(err_msg.str());
	}

	getline(fdem, s);
	if (s.compare(0, 22, "# vtk DataFile Version") != 0) {
		err_msg	<< fname << " is not a VTK data file";
		throw runtime_error(err_msg.str());
	}

	getline(fdem, s);
	cout << "Trying to load " << fname << ": \"" << s << "\"" << endl;

	getline(fdem, s); // ASCII or BINARY
	if (s != string("ASCII")) {
		// TODO FIXME support binary
		err_msg << fname << " has unsupported " << s <<  " data" << endl;
		throw runtime_error(err_msg.str());
	}

	getline(fdem, s);
	if (s != string("DATASET STRUCTURED_POINTS")) {
		err_msg << fname << " is not a STRUCTURED_POINTS dataset" << endl;
		throw runtime_error(err_msg.str());
	}

	while (fdem >> s) {
		if (s == "DIMENSIONS") {
			fdem >> ncols;
			fdem >> nrows;
			fdem >> nz;
			if (nz > 1) {
				err_msg << fname << " has " << nz << " > 1 layers. not supported";
				throw runtime_error(err_msg.str());
			}
		} else if (s == "ORIGIN") {
			fdem >> west;
			fdem >> south;
			fdem >> zbase;
		} else if (s == "SPACING") {
			fdem >> ewres;
			fdem >> nsres;
			fdem >> z; // skip
		} else if (s == "POINT_DATA") {
			fdem >> nels;
			if (nels != nrows*ncols) {
				err_msg << fname << " has " << nels << " point data but " << (nrows*ncols) << " elements";
				throw runtime_error(err_msg.str());
			}
		} else if (s == "SCALARS") {
			fdem >> s;
			cout << "Reading " << s << " scalar field from " << fname << endl;
			fdem.ignore(UINT_MAX, '\n'); // skip to EOL
		} else if (s == "LOOKUP_TABLE") {
			fdem.ignore(UINT_MAX, '\n'); // skip to EOL
			break; // next, start reading the data
		}
	}

	float *dem = new float[nels];

	for (int el = 0; el < nels; ++el) {
		fdem >> z;
		zmax = fmax(z, zmax);
		zmin = fmin(z, zmin);
		dem[el] = z;
	}
	fdem.close();

	const double sizex = ewres*(ncols-1);
	const double sizey = nsres*(nrows-1);

	north = south + sizey;
	east = west + sizex;

	TopoCube *ret = new TopoCube();

	ret->SetCubeDem(dem, sizex, sizey, zmax - zmin, ncols, nrows, -zmin);
	ret->SetGeoLocation(north, south, east, west);

	delete [] dem;

	return ret;
}

TopoCube *TopoCube::load_vtk_file(const char* fname)
{ return load_file<DEM_FMT_VTK>(fname, RELAXED); }


/* Create a TopoCube from a (xyz) sorted ASCII elevation file.
 * The format of the file must be:
 *
 * cols: number of columns
 * rows: number of rows
 * x y z
 *
 * - The total number of line of the file must be cols*rows+2.
 * - Data must be sorted by x (first key) and by y (second key)
 * values.
 * - The underlying grid must be regular : constant and same
 * resolution along x and y.
 *
 * Format options are presently ignored.
 */
template<>
TopoCube *TopoCube::load_file<TopoCube::DEM_FMT_XYZ>(const char* fname, FormatOptions /* fmt_options */)
{
	ifstream fdem(fname);
	if (!fdem.good()) {
		stringstream err_msg;
		err_msg	<< "failed to open DEM " << fname;

		throw runtime_error(err_msg.str());
	}

	string s;

	double north, south, east, west;
	int ncols, nrows;

	// TODO FIXME: xyz files may not have cols or rows specification,
	// and we need to ensure that the data is in the correct (south to north)
	// order
	for (int i = 1; i <= 2; i++) {
		fdem >> s;
		if (s.find("cols:") != string::npos) fdem >> ncols;
		else if (s.find("rows:") != string::npos) fdem >> nrows;
	}

	double zmin = NAN, zmax = NAN;
	float *dem = new float[ncols*nrows];
	// resolution in x and y directions
	double xres = 0, yres = 0;
	double x, y, z;
	for (int col = 0; col < ncols; ++col) {
		for (int row = 0; row < nrows; ++row) {
			fdem >> x >> y >> z;
			if (row == 0 && col == 0) {
				west = x;
				south = y;
			} else if (row == 1 && col == 0)
				yres = y - south;
			else if (row == 0 && col == 1)
				xres = x - west;
			zmax = max(z, zmax);
			zmin = min(z, zmin);
			dem[row*ncols+col] = z;
		}
	}
	fdem.close();

	const double sizex = xres*(ncols-1);
	const double sizey = yres*(nrows-1);

	north = south + sizey;
	east = west + sizex;

	TopoCube *ret = new TopoCube();

	ret->SetCubeDem(dem, sizex, sizey, zmax-zmin, ncols, nrows, -zmin);
	ret->SetGeoLocation(north, south, east, west);

	delete [] dem;

	return ret;
}

TopoCube *TopoCube::load_xyz_file(const char* fname)
{ return load_file<DEM_FMT_XYZ>(fname, RELAXED); }

TopoCube *TopoCube::load_file(const char *fname, Format fmt, FormatOptions fmt_options)
{
	switch (fmt)
	{
	case DEM_FMT_ASCII: return load_file<DEM_FMT_ASCII>(fname, fmt_options);
	case DEM_FMT_VTK: return load_file<DEM_FMT_VTK>(fname, fmt_options);
	case DEM_FMT_XYZ: return load_file<DEM_FMT_XYZ>(fname, fmt_options);
	}
	throw std::invalid_argument("Unsupported format for " + string(fname));
}


double
TopoCube::SetPartMass(const double dx, const double rho)
{
	const double mass = dx*dx*dx*rho;

	m_origin(3) = m_center(3) = mass;
	return mass;
}

void TopoCube::setEulerParameters(const EulerParameters &ep)
{
	// TODO: do not print a warning if non-zero
	//m_ep = EulerParameters(ep);
	//m_ep.ComputeRot();
	printf("WARNING: trying to applying a rotation to a TopoCube. Ignoring\n");
}

void TopoCube::getBoundingBox(Point &output_min, Point &output_max)
{
	getBoundingBoxOfCube(output_min, output_max, m_origin, m_vx, m_vy, m_vz );
}

void TopoCube::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_origin += poff;
}

void
TopoCube::FillBorder(PointVect& points, const double dx, const int face_num, const bool fill_edges,
	const int layers, const int starting_layer)
{
	const int abs_layers = layers < 0 ? -layers : layers;
	const double layer_dx = layers < 0 ? -dx : dx;
	const double layer_depth = (abs_layers-1)*layer_dx;
	const double layer_width = abs_layers*layer_dx;

	Point   rorigin;
	Vector  x_dir = m_vx/m_vx.norm();
	Vector  y_dir = m_vy/m_vy.norm();
	Vector  v, o_shift, v_shift;

	for (int l = starting_layer ; l < abs_layers + starting_layer; ++l) {
		const double layer_offset = layer_dx*l;

		switch(face_num){
		case 0:
			rorigin = m_origin;
			v = m_vx;
			o_shift = y_dir;
			v_shift = x_dir;
			break;
		case 1:
			rorigin = m_origin + m_vx;
			v = m_vy;
			o_shift = -x_dir;
			v_shift =  y_dir;
			break;
		case 2:
			rorigin = m_origin + m_vx + m_vy;
			v = -m_vx;
			o_shift = -y_dir;
			v_shift = -x_dir;
			break;
		case 3:
			rorigin = m_origin + m_vy;
			v = -m_vy;
			o_shift =  x_dir;
			v_shift = -y_dir;
			break;
		}
		rorigin += layer_offset*o_shift;
		if (fill_edges) {
			rorigin += layer_width*v_shift;
			v -= 2*layer_width*v_shift;
		} else if (starting_layer > 0) {
			rorigin += layer_dx*v_shift;
			v -= 2*layer_dx*v_shift;
		}

		const int n = (int) (v.norm()/dx);
		// const double delta = v.norm()/((double) n);
		int nstart = 0;
		int nend = n;
		if (!fill_edges) {
			nstart++;
			nend--;
		}

		for (int i = nstart; i <= nend; i++) {
			const double x = rorigin(0) + (double) i/((double) n)*v(0);
			const double y = rorigin(1) + (double) i/((double) n)*v(1);
			float z = m_H;
			while (DemDist(x, y, z, dx) > layer_depth) {
				Point p(x, y, z, m_center(3));
				points.push_back(p);
				z -= dx;
			}
		}
	}
}


void
TopoCube::FillDem(PointVect& points, const int layers, const double dx)
{
	const int abs_layers = layers < 0 ? -layers : layers;
	const double layer_dx = layers < 0 ? -dx : dx;

	int nx = (int) (m_vx.norm()/dx);
	int ny = (int) (m_vy.norm()/dx);
	/*
	double deltax = m_vx.norm()/((double) nx);
	double deltay = m_vy.norm()/((double) ny);
	*/

	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			const double x = (double) i/((double) nx)*m_vx(0) + (double) j/((double) ny)*m_vy(0);
			const double y = (double) i/((double) nx)*m_vx(1) + (double) j/((double) ny)*m_vy(1);
			const double z = DemInterpolInternal(x, y);

			const double g_x = m_origin(0) + x;
			const double g_y = m_origin(0) + y;
			const double g_z = m_origin(0) + z;
			for (int l = 0; l < abs_layers; ++l) {
				Point p(g_x, g_y, g_z + l*layer_dx, m_center(3));
				points.push_back(p);
			}
		}
	}
}

// x, y are coordinates in the global reference system
double
TopoCube::DemInterpol(const double x, const double y) const  // x and y ranging in [0, ncols/ewres]x[0, nrows/nsres]
{
	return DemInterpolInternal(x - m_origin(0), y - m_origin(1));
}

// x, y should already be mapped to the DEM domain, and thus in the range
// [0, (ncols-1)*ewres] and [0, (nrows-1)*nsres] respectively
double
TopoCube::DemInterpolInternal(const double x, const double y) const
{
	const double xb  = x/m_ewres; // map to [0, ncols - 1]
	const double yb  = y/m_nsres; // map to [0, nrows - 1]
	// find the vertices of the square this points belongs to,
	// and ensure we are within the domain covered by the DEM
	// (outer points will be squashed to the edge values)
	const int    i   = clamp(floor(xb), 0, m_ncols - 1);
	const int    j   = clamp(floor(yb), 0, m_nrows - 1);
	const int    ip1 = clamp(i + 1,     0, m_ncols - 1);
	const int    jp1 = clamp(j + 1,     0, m_nrows - 1);
	const double pa  = xb - (double) i;
	const double pb  = yb - (double) j;
	const double ma  = 1 - pa;
	const double mb  = 1 - pb;
	const double z00 = ma*mb*m_dem[i   + j   * m_ncols];
	const double z10 = pa*mb*m_dem[ip1 + j   * m_ncols];
	const double z01 = ma*pb*m_dem[i   + jp1 * m_ncols];
	const double z11 = pa*pb*m_dem[ip1 + jp1 * m_ncols];

	return z00 + z10 + z01 + z11;
}

// x, y, z in global coordinate space
double
TopoCube::DemDist(const double x, const double y, const double z, double dx) const
{
	return DemDistInternal(x - m_origin(0), y - m_origin(1), z - m_origin(2), dx);
}

// x, y, z are relative to the DEM origin
double
TopoCube::DemDistInternal(const double x, const double y, const double z, double dx) const
{
	// We compute the distance of the particle to the DEM as the distance
	// to the tangent plane passing through the projection of the particle
	// on the DEM.

	if (dx > 0.5*min(m_nsres, m_ewres))
		dx = 0.5*min(m_nsres, m_ewres);

	// These three points are used in both the FAST and symmetrized version
	// of the DEM computation
	const double z0 = DemInterpolInternal(x, y);
	const double zpx = DemInterpolInternal(x + dx, y);
	const double zpy = DemInterpolInternal(x, y + dx);

#if FASTDEM
	// 'Classic', 'fast' computation: find the plane through three points,
	// where the three points are z0 (projection) and two other points
	// obtained by sampling the DEM at distance dx along both the x and y axes.
	// Note that this disregards any DEM symmetry information.

	// A(x, y, z0) B(x + h, y, z1) C(x, y + h, z2)
	// AB(h, 0, z1 - z0) AC(0, h, z2 - z0)
	// AB^AC = ( -h*(z1 - z0), -h*(z2 - z0), h*h)
#if 0
	const double a = dy*(z0 - zpx);
	const double b = dx*(z0 - zpy);
	const double c = dy*dx;
#else
	// dx = dy allows us to simplify to:
	const double a = (z0 - zpx);
	const double b = (z0 - zpy);
	const double c = dx;
#endif
#else
	// We compute the slope of the plane in each direction
	// by taking the difference between two symmetrically
	// placed points
	const double zmx = DemInterpolInternal(x - dx, y);
	const double zmy = DemInterpolInternal(x, y - dx);

	// The slope in the zx plane is A/∆ = (zmx - zpx)/(2*dx),
	// the slope in the zy plane is B/∆ = (zmy - zpy)/(2*dy),
	// the plane A*x + B*y + z + D must pass through z0,
	// so D = -(A*x + B*y + z0).
#if 0
	const double a = 2*dy*(zmx - zpx);
	const double b = 2*dx*(zmy - zpy);
	const double c = 4*dx*dy;
#else
	// dx = dy allows us to simplify to:
	const double a = (zmx - zpx);
	const double b = (zmy - zpy);
	const double c = 2*dx;
#endif
#endif
	const double d = -(a*x + b*y + c*z0);
	const double l = sqrt(a*a + b*b + c*c);

	// Getting distance along the normal
	double r = (a*x + b*y + c*z + d)/l;
	return r;
}


int
TopoCube::Fill(PointVect& points, const double H, const double dx, const bool faces_filled, const bool fill)
{
	int nparts = 0;

	const int nx = (int) (m_vx.norm()/dx);
	const int ny = (int) (m_vy.norm()/dx);
	/*
	const double deltax = m_vx.norm()/((double) nx);
	const double deltay = m_vy.norm()/((double) ny);
	*/

	int startx = 0;
	int starty = 0;
	int endx = nx;
	int endy = ny;
	if (faces_filled) {
		startx ++;
		starty++;
		endx--;
		endy--;
	}

	const double filling_offset = std::isfinite(m_filling_offset) ? m_filling_offset : dx;

	for (int i = startx; i <= endx; i++) {
		for (int j = starty; j <= endy; j++) {
			double x = i/((float) nx)*m_vx(0) + (float) j/((float) ny)*m_vy(0);
			double y = i/((float) nx)*m_vx(1) + (float) j/((float) ny)*m_vy(1);
			double z = H;
			while (DemDistInternal(x, y, z, dx) > filling_offset) {
				Point p(m_origin(0) + x, m_origin(1) + y, m_origin(2) + z, m_center(3));
				nparts ++;
				if (fill)
					points.push_back(p);
				z -= dx;
			}
		}
	}

	return nparts;
}

void
TopoCube::FillIn(PointVect& points, const double dx, const int layers)
{
	FillBorder(points, dx, 0, true , layers, 1);
	FillBorder(points, dx, 1, false, layers, 1);
	FillBorder(points, dx, 2, true , layers, 1);
	FillBorder(points, dx, 3, false, layers, 1);
	FillDem(points, layers, dx);
}


bool
TopoCube::IsInside(const Point& pt, const double dx) const
{
	// TODO should check that pt falls within the DEM area
	return DemDist(pt(0), pt(1), pt(3), dx) < dx;
}
