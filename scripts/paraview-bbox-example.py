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

fluid.UpdatePipeline(timestep[0])

array_names = [a[0] for a in fluid.PointData.items()]

if 'Fluid number' in array_names:
    fluidnum_range = fluid.PointData.GetArray('Fluid number').GetRange()
    nfluids = int(fluidnum_range[1]) + 1
    fluids = []
    for f in range(0, nfluids):
        afluid = Threshold(fluid)
        afluid.Scalars = 'Fluid number'
        afluid.ThresholdRange = [f, f]
        fluids.append(afluid)
else:
    nfluids = 1
    fluids = [fluid]

header = "#time,xmin,xmax,ymin,ymax,zmin,zmax,rhomin,rhomax"
fmt = "%f,%f,%f,%f,%f,%f,%f,%f,%f"
endfmt = "**,%f,%f,%f,%f,%f,%f,%f,%f"
if nfluids > 1:
    for f in range(0, nfluids):
        header += ",xmin{0},xmax{0},ymin{0},ymax{0},zmin{0},zmax{0},rhomin{0},rhomax{0}".format(f)
        fmt += ",%f,%f,%f,%f,%f,%f,%f,%f"
        endfmt += ",%f,%f,%f,%f,%f,%f,%f,%f"

print header

first = None
last = None

# iterate over available timestep, gather bounds, print time and bounds
for time in timestep:
    fluid.UpdatePipeline(time)
    bounds =    fluid.GetDataInformation().GetBounds()[:] + \
                fluid.PointData.GetArray('Density').GetRange()
    if nfluids > 1:
        for f in range(0, nfluids):
            afluid = fluids[f]
            afluid.UpdatePipeline(time)
            bounds += afluid.GetDataInformation().GetBounds()[:] + \
                     afluid.PointData.GetArray('Density').GetRange()
    if not first:
        first = bounds
    last = bounds
    print fmt % ((time,) + bounds)

print endfmt % tuple(pair[1] - pair[0] for pair in zip(first, last))
