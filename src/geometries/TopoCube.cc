/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#include <cstring> // memcpy

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdexcept>

#include "TopoCube.h"
#include "Point.h"
#include "Vector.h"
#include "Rect.h"

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
 */

void TopoCube::SetCubeDem(const float *dem,
		double sizex, double sizey, double H,
		int ncols, int nrows, double voff)
{
	m_origin = Point(0, 0, 0, 0);
	m_vx = Vector(sizex, 0.0, 0.0);
	m_vy = Vector(0.0, sizey,0.0);
	SetCubeHeight(H);

	m_ncols = ncols;
	m_nrows = nrows;
	m_ewres = sizex/(ncols-1);
	m_nsres = sizey/(nrows-1);

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

/* Create a TopoCube from a (GRASS) ASCII grid file */
template<>
TopoCube *TopoCube::load_file<TopoCube::DEM_FMT_ASCII>(const char* fname)
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
		ncols, nrows, -zmin);
	ret->SetGeoLocation(north, south, east, west);

	delete [] dem;

	return ret;
}

TopoCube *TopoCube::load_ascii_grid(const char *fname)
{ return load_file<DEM_FMT_ASCII>(fname); }

/* Create a TopoCube from a VTK Structure Points datafile */
template<>
TopoCube *TopoCube::load_file<TopoCube::DEM_FMT_VTK>(const char* fname)
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

	TopoCube *ret = new TopoCube();

	ret->SetCubeDem(dem, ewres*ncols, nsres*nrows, zmax - zmin,
		ncols, nrows, -zmin);
	ret->SetGeoLocation(south + nsres*nrows, south, west + ewres*ncols, west);

	delete [] dem;

	return ret;
}

TopoCube *TopoCube::load_vtk_file(const char* fname)
{ return load_file<DEM_FMT_VTK>(fname); }


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
 */
template<>
TopoCube *TopoCube::load_file<TopoCube::DEM_FMT_XYZ>(const char* fname)
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

	for (int i = 1; i <= 2; i++) {
		fdem >> s;
		if (s.find("cols:") != string::npos) fdem >> ncols;
		else if (s.find("rows:") != string::npos) fdem >> nrows;
	}

	double zmin = NAN, zmax = NAN;
	float *dem = new float[ncols*nrows];
	// resolution in x and y directions
	double xres = 0, yres = 0;
	double x ,y, z;
	for (int col = 0; col < ncols; ++col) {
		for (int row = 0; row < nrows; ++row) {
			fdem >> x >> y >> z;
			if (row == 0 && col == 0) {
				xres = x;
				yres = y;
			} else if (row == 1 && col == 0)
				yres = y - yres;
			else if (row == 0 && col == 1)
				xres = x - xres;
			zmax = max(z, zmax);
			zmin = min(z, zmin);
			dem[row*ncols+col] = z;
		}
	}
	fdem.close();

	// support degenerate case of single-row or single-column files
	if (xres == 0 && yres)
		xres = yres;
	else if (yres == 0 && xres)
		yres = xres;

	south = 0;
	west = 0;
	north = nrows*yres;
	east = ncols*xres;

	TopoCube *ret = new TopoCube();

	ret->SetCubeDem(dem, east-west, north-south, zmax-zmin,
		ncols, nrows, -zmin);
	ret->SetGeoLocation(north, south, east, west);

	delete [] dem;

	return ret;
}

TopoCube *TopoCube::load_xyz_file(const char* fname)
{ return load_file<DEM_FMT_XYZ>(fname); }

TopoCube *TopoCube::load_file(const char *fname, Format fmt)
{
	switch (fmt)
	{
	case DEM_FMT_ASCII: return load_file<DEM_FMT_ASCII>(fname);
	case DEM_FMT_VTK: return load_file<DEM_FMT_VTK>(fname);
	case DEM_FMT_XYZ: return load_file<DEM_FMT_XYZ>(fname);
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
TopoCube::FillBorder(PointVect& points, const double dx, const int face_num, const bool fill_edges)
{
	Point   rorigin;
	Vector  v;

	switch(face_num){
		case 0:
			rorigin = m_origin;
			v = m_vx;
			break;
		case 1:
			rorigin = m_origin + m_vx;
			v = m_vy;
			break;
		case 2:
			rorigin = m_origin + m_vx + m_vy;
			v = -m_vx;
			break;
		case 3:
			rorigin = m_origin + m_vy;
			v = -m_vy;
			break;
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
		while (DemDist(x, y, z, dx) > 0) {
			Point p(x, y, z, m_center(3));
			points.push_back(p);
			z -= dx;
		}
	}
}


void
TopoCube::FillDem(PointVect& points, double dx)
{
	int nx = (int) (m_vx.norm()/dx);
	int ny = (int) (m_vy.norm()/dx);
	/*
	double deltax = m_vx.norm()/((double) nx);
	double deltay = m_vy.norm()/((double) ny);
	*/

	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			const double x = m_origin(0) + (double) i/((double) nx)*m_vx(0) + (double) j/((double) ny)*m_vy(0);
			const double y = m_origin(1) + (double) i/((double) nx)*m_vx(1) + (double) j/((double) ny)*m_vy(1);
			const double z = DemInterpol(x, y);
			Point p(x, y, z, m_center(3));
			points.push_back(p);
			}
		}
}


double
TopoCube::DemInterpol(const double x, const double y) const  // x and y ranging in [0, ncols/ewres]x[0, nrows/nsres]
{
	const double xb = x/m_ewres;
	const double yb = y/m_nsres;
	int i = floor(xb);
	int ip1 = i < m_ncols - 1 ? i + 1 : i;
	int j = floor(yb);
	int jp1 = j < m_nrows - 1 ? j + 1 : j;
	const double a = xb - (float) i;
	const double b = yb - (float) j;
	double z = (1 - a)*(1 - b)*m_dem[i + j*m_ncols];
	z +=  a*(1 - b)*m_dem[ ip1 + j*m_ncols];
	z +=  (1 - a)*b*m_dem[i + jp1*m_ncols];
	z += a*b*m_dem[ip1 + jp1*m_ncols];
	return z;
}



double
TopoCube::DemDist(const double x, const double y, const double z, double dx) const
{
	if (dx > 0.5*min(m_nsres, m_ewres))
		dx = 0.5*min(m_nsres, m_ewres);
	const double z0 = DemInterpol(x, y);
	const double z1 = DemInterpol(x + dx, y);
	const double z2 = DemInterpol(x, y + dx);
	// A(x, y, z0) B(x + h, y, z1) C(x, y + h, z2)
	// AB(h, 0, z1 - z0) AC(0, h, z2 - z0)
	// AB^AC = ( -h*(z1 - z0), -h*(z2 - z0), h*h)
	const double a = dx*(z0 - z1);
	const double b = dx*(z0 - z2);
	const double c = dx*dx;
	const double d = - a*x - b*y - c*z0;
	const double l = sqrt(a*a + b*b + c*c);

	// Getting distance along the normal
	double r = fabs(a*x + b*y + c*z + d)/l;
	if (z <= z0)
		r = 0;
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

	for (int i = startx; i <= endx; i++) {
		for (int j = starty; j <= endy; j++) {
			float x = m_origin(0) + (float) i/((float) nx)*m_vx(0) + (float) j/((float) ny)*m_vy(0);
			float y = m_origin(1) + (float) i/((float) nx)*m_vx(1) + (float) j/((float) ny)*m_vy(1);
			float z = H;
			while (DemDist(x, y, z, dx) > dx) {
				Point p(x, y, z, m_center(3));
				nparts ++;
				if (fill)
					points.push_back(p);
				z -= dx;
			}
		}
	}

	return nparts;
}


bool
TopoCube::IsInside(const Point& pt, const double dx) const
{
	// TODO should check that pt falls within the DEM area
	return DemDist(pt(0), pt(1), pt(3), dx) < dx;
}
