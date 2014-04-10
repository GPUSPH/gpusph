#!/usr/bin/env python

"""
This script converts an ASCII GRID Digital Elevation Model into a VTK Structured Points
dataset (legacy format), simply by replacing the ASCII GRID header with the appropriate VTK
metadata. 

The VTK dataset is two-dimensional (flat), so a Warp By Scalar filter should be applied in
ParaView to convert the Height field into actual height information.
"""

import sys
import string
import os

def usage():
    print "%s filename -- convert filename (ASCII GRID) into a VTK dataset" % os.path.basename(sys.argv[0])

if len(sys.argv) < 2:
    usage()
    sys.exit(0)

grid_fname = sys.argv[1]
vts_fname = os.path.splitext(sys.argv[1])[0] + '.vtk'

grid = open(grid_fname, 'r')
vts = None


md = {}

for line in grid:
    if string.find(line, ': ') >= 0:
        key, val = string.split(line, ': ', 2)
        md[key] = int(val) if string.find(val, '.') < 0 else float(val)
    else:
        if vts is None:
            # Open vts file and write the header
            vts = open(vts_fname, 'w')
            vts.write("# vtk DataFile Version 2.0\n")
            vts.write("Converted from %s\n" % grid_fname)
            vts.write("ASCII\nDATASET STRUCTURED_POINTS\n")
            vts.write("DIMENSIONS {cols} {rows} 1\n".format(**md))
            vts.write("ORIGIN {west} {south} 0\n".format(**md))
            resolution = min(
                (md['east']-md['west'])/md['cols'],
                (md['north'] - md['south'])/md['rows'])
            vts.write("SPACING {0} {0} {0}\n".format(resolution))
            vts.write("POINT_DATA {0}\n".format(md['rows']*md['cols']))
            vts.write("SCALARS Height float 1\nLOOKUP_TABLE default\n")
        vts.write(line)

