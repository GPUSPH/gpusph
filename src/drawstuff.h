/*************************************************************************
 *                                                                       *
 * Open Dynamics Engine, Copyright (C) 2001-2003 Russell L. Smith.       *
 * All rights reserved.  Email: russ@q12.org   Web: www.q12.org          *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of EITHER:                                  *
 *   (1) The GNU Lesser General Public License as published by the Free  *
 *       Software Foundation; either version 2.1 of the License, or (at  *
 *       your option) any later version. The text of the GNU Lesser      *
 *       General Public License is included with this library in the     *
 *       file LICENSE.TXT.                                               *
 *   (2) The BSD-style license that is included with this library in     *
 *       the file LICENSE-BSD.TXT.                                       *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files    *
 * LICENSE.TXT and LICENSE-BSD.TXT for more details.                     *
 *                                                                       *
 *************************************************************************/

/** @defgroup drawstuff DrawStuff

DrawStuff is a library for rendering simple 3D objects in a virtual
environment, for the purposes of demonstrating the features of ODE.
It is provided for demonstration purposes and is not intended for
production use.

@section Notes

In the virtual world, the z axis is "up" and z=0 is the floor.

The user is able to click+drag in the main window to move the camera:
  * left button - pan and tilt.
  * right button - forward and sideways.
  * left + right button (or middle button) - sideways and up.
*/


#ifndef __DRAWSTUFF_H__
#define __DRAWSTUFF_H__

/* texture numbers */
  enum DS_TEXTURE_NUMBER
  {
    DS_NONE = 0,       /* uses the current color instead of a texture */
    DS_WOOD,
    DS_CHECKERED,
    DS_GROUND,
    DS_SKY,
  };

/* draw modes */

#define DS_POLYFILL  0
#define DS_WIREFRAME 1



/**
 * @brief Sets the viewpoint
 * @ingroup drawstuff
 * @param xyz camera position.
 * @param hpr contains heading, pitch and roll numbers in degrees. heading=0
 * points along the x axis, pitch=0 is looking towards the horizon, and
 * roll 0 is "unrotated".
 */
void dsSetViewpoint (float xyz[3], float hpr[3]);


/**
 * @brief Gets the viewpoint
 * @ingroup drawstuff
 * @param xyz position
 * @param hpr heading,pitch,roll.
 */
void dsGetViewpoint (float xyz[3], float hpr[3]);


/**
 * @brief Set the color with which geometry is drawn.
 * @ingroup drawstuff
 * @param red Red component from 0 to 1
 * @param green Green component from 0 to 1
 * @param blue Blue component from 0 to 1
 */
void dsSetColor (float red, float green, float blue);

/**
 * @brief Set the color and transparency with which geometry is drawn.
 * @ingroup drawstuff
 * @param alpha Note that alpha transparency is a misnomer: it is alpha opacity.
 * 1.0 means fully opaque, and 0.0 means fully transparent.
 */
void dsSetColorAlpha (float red, float green, float blue, float alpha);

/**
 * @brief Draw a box.
 * @ingroup drawstuff
 * @param pos is the x,y,z of the center of the object.
 * @param R is a 3x3 rotation matrix for the object, stored by row like this:
 *        [ R11 R12 R13 0 ]
 *        [ R21 R22 R23 0 ]
 *        [ R31 R32 R33 0 ]
 * @param sides[] is an array of x,y,z side lengths.
 */
void dsDrawBox (const float pos[3], const float R[12], const float sides[3]);

/**
 * @brief Draw a sphere.
 * @ingroup drawstuff
 * @param pos Position of center.
 * @param R orientation.
 * @param radius
 */
void dsDrawSphere (const float pos[3], const float R[12], float radius);

/**
 * @brief Draw a triangle.
 * @ingroup drawstuff
 * @param pos Position of center
 * @param R orientation
 * @param v0 first vertex
 * @param v1 second
 * @param v2 third vertex
 * @param solid set to 0 for wireframe
 */
void dsDrawTriangle (const float pos[3], const float R[12],
		     const float *v0, const float *v1, const float *v2, int solid);

/**
 * @brief Draw a z-aligned cylinder
 * @ingroup drawstuff
 */
void dsDrawCylinder (const float pos[3], const float R[12],
		     float length, float radius);

/**
 * @brief Draw a z-aligned capsule
 * @ingroup drawstuff
 */
void dsDrawCapsule (const float pos[3], const float R[12],
		    float length, float radius);

/**
 * @brief Draw a line.
 * @ingroup drawstuff
 */
void dsDrawLine (const float pos1[3], const float pos2[3]);

/**
 * @brief Draw a convex shape.
 * @ingroup drawstuff
 */
void dsDrawConvex(const float pos[3], const float R[12],
		  float *_planes,
		  unsigned int _planecount,
		  float *_points,
		  unsigned int _pointcount,
		  unsigned int *_polygons);

 /* these drawing functions are identical to the ones above, except they take
 * double arrays for `pos' and `R'.
 */
void dsDrawBox (const double pos[3], const double R[12],
		 const double sides[3]);
void dsDrawSphere (const double pos[3], const double R[12],
		    const float radius);
void dsDrawTriangle (const double pos[3], const double R[12],
		      const double *v0, const double *v1, const double *v2, int solid);
void dsDrawCylinder (const double pos[3], const double R[12],
		      float length, float radius);
void dsDrawCapsule (const double pos[3], const double R[12],
		     float length, float radius);
void dsDrawLine (const double pos1[3], const double pos2[3]);
void dsDrawConvex(const double pos[3], const double R[12],
		  double *_planes,
		  unsigned int _planecount,
		  double *_points,
		  unsigned int _pointcount,
		  unsigned int *_polygons);

/**
 * @brief Set the quality with which curved objects are rendered.
 * @ingroup drawstuff
 * Higher numbers are higher quality, but slower to draw.
 * This must be set before the first objects are drawn to be effective.
 * Default sphere quality is 1, default capsule quality is 3.
 */
void dsSetSphereQuality (int n);		/* default = 1 */
void dsSetCapsuleQuality (int n);		/* default = 3 */

/**
 * @brief Set Drawmode 0=Polygon Fill,1=Wireframe).
 * Use the DS_POLYFILL and DS_WIREFRAME macros.
 * @ingroup drawstuff
 */
void dsSetDrawMode(int mode);

#endif

