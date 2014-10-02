#!/usr/bin/python

# Sample script on how to script ParaView to extract salient data from a GPUSPH simulation

# Usage: scripts/paraview-bbox-exampe.py pathname

# The script will read the VTU datafiles at the given pathname and extract the evolution
# of the fluid bounding box over time

import os, sys

from paraview.simple import *

Connect()

path = sys.argv[1]

# Open the index file
try:
    vtp = OpenDataFile(os.path.join(path, 'VTUinp.pvd'))
except RuntimeError:
    vtp = OpenDataFile(os.path.join(path, 'data', 'VTUinp.pvd'))


# get all available timesteps
timestep = vtp.TimestepValues

# threshold on particle type 0 (fluid)
fluid = Threshold(vtp)
fluid.Scalars = 'Part type'
fluid.ThresholdRange = [0, 0]

# if the data file stores Part flag,
# threshold on particle flag 0 to avoid FIXED particles
# (and potentially other future flags such as inlet/outlet particles
if fluid.PointData.GetArray('Part flag'):
    noflags = Threshold(fluid)
    noflags.Scalars = 'Part flag'
    noflags.ThresholdRange = [0, 1]
else:
    noflags = fluid

print "#time,xmin,xmax,ymin,ymax,zmin,zmax"

# iterate over available timestep, gather bounds, print time and bounds
for time in timestep:
    noflags.UpdatePipeline(time)
    bounds = noflags.GetDataInformation().GetBounds()[:]
    print "%f,%f,%f,%f,%f,%f,%f" % ((time,) + bounds )

