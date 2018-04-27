#!/usr/bin/python

# Tabulate the range of gamma (semi-analytical boundary model correction)
# for each set of particle type.

# Usage: scripts/check_gamma.py pathname

import os, sys

from paraview.simple import *

Connect()

path = sys.argv[1]

# Open the index file
try:
    vtp = OpenDataFile(os.path.join(path, 'VTUinp.pvd'))
except RuntimeError:
    vtp = OpenDataFile(os.path.join(path, 'data', 'VTUinp.pvd'))

# We find separate bounds for each type, so let's threshold on each of them
def part_type(vtp, idx):
    t = Threshold(vtp)
    t.Scalars = 'Part type'
    t.ThresholdRange = [idx, idx]
    return t

types = [ part_type(vtp, i) for i in range(3) ]

header = "\t".join(["time"] + ["min%d\tmax%d" % (i, i) for i in range(3) ])
fmt = "\t".join(["%g"] + ["%g\t%g" for i in range(3) ])

# get all available timesteps
timestep = vtp.TimestepValues

print header

# iterate over available timestep, gather bounds, print time and bounds
for time in timestep:
    for t in types: t.UpdatePipeline(time)
    bounds = []
    map(bounds.extend, [t.PointData.GetArray('Gamma').GetRange() for t in types])
    print fmt % ((time,) + tuple(bounds))
