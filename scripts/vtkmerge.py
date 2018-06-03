#!/usr/bin/python

# Merge two Paraview datasets.
# All common arrays will be stored in a single copy, but if differences are found, then
# a copy of the second array will be added prefixed by 'New'
# The two datasets must have the same number of points, with the same IDs

# Usage: scripts/vtkdiff.py old new output

import os, sys, math
import inspect

from paraview.simple import *

def match(t1, t2):
    if all(math.isnan(x) for x in t1) and all(math.isnan(x) for x in t2):
        return True
    return t1 == t2

def scriptfilter(self, inputs, output, request):
    #print self
    #print inputs
    #print output
    #print request

    # We expect two inputs
    if len(inputs) != 2:
        raise ValueError("need 2 inputs, got {}".format(len(inputs)))

    # old = inputs[0]
    # new = inputs[1]

    old = self.GetInputDataObject(0, 0)
    new = self.GetInputDataObject(0, 1)
    out = self.GetOutputDataObject(0)

    # Check that they have the same number of points
    oldpts = old.GetNumberOfPoints()
    newpts = new.GetNumberOfPoints()

    if oldpts != newpts:
        raise ValueError("number of points differs: {} vs {}".format(oldpts, newpts))

    olddata = old.GetPointData()
    newdata = new.GetPointData()

    # Check that they have the same array names
    oldnames = [olddata.GetArrayName(i) for i in range(0, olddata.GetNumberOfArrays())]
    newnames = [newdata.GetArrayName(i) for i in range(0, newdata.GetNumberOfArrays())]

    if oldnames != newnames:
        raise ValueError("mismatching arrays: {} vs {}".format(oldnames, newnames))

    # Check that the particle IDs match

    oldids = olddata.GetArray("Part id")
    newids = newdata.GetArray("Part id")

    for pt in range(0, oldpts):
        oid = oldids.GetValue(pt)
        nid = newids.GetValue(pt)
        if oid != nid:
            raise ValueError("ids don't match @ {} : {} vs {}".format(pt, oid, nid))

    # Function to do the matching
    def match(t1, t2):
        if len(t1) != len(t2):
            return False
        for i in range(len(t1)):
            v1 = t1[i]
            v2 = t2[i]
            nan1 = math.isnan(v1)
            nan2 = math.isnan(v2)
            if nan1 != nan2:
                return False
            # Now they are either both NaN (which is fine), or both not NaN,
            # in which case we need to check that they are the same
            if not nan1 and v1 != v2:
                return False
        # No components mismatched, so the items match
        return True

    # Loop over each array and, if any has data that differ, copy it
    tocopy = []
    for array in oldnames:
        oldvals = olddata.GetArray(array)
        newvals = newdata.GetArray(array)

        print "Checking {}".format(array)

        for pt in range(0, oldpts):
            oldval = oldvals.GetTuple(pt)
            newval = newvals.GetTuple(pt)
            if not match(oldval, newval):
                print "Mismatch found @ {}: {} vs {}".format(pt, oldval, newval)
                tocopy.append(array)
                break

    if not tocopy:
        return

    print "Need to copy: {}".format(tocopy)
    for name in tocopy:
        src = newdata.GetArray(name)
        src.SetName("New " + name)
        out.GetPointData().AddArray(src)

script = inspect.getsource(scriptfilter)

script += "scriptfilter(self, inputs, output, request)"


Connect()

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]

# Open the index file
datafile1 = OpenDataFile(file1)
datafile2 = OpenDataFile(file2)

datafile1.UpdatePipeline()
datafile2.UpdatePipeline()

datafile1.UpdatePipeline()
datafile2.UpdatePipeline()

flt = ProgrammableFilter(
        CopyArrays = True,
        Input = [datafile1, datafile2],
        Script = script)

flt.UpdatePipeline()

SaveData(file3, flt)
print "{} saved".format(file3)
