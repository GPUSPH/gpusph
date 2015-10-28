#!/usr/bin/python

# Compare two Paraview data files

# Usage: scripts/vtkdiff.py file1 file2

import os, sys, math

from paraview.simple import *

def tuplediff(t1, t2):
        return tuple(a - b for a, b in zip(t1, t2))

def tuplereldiff(t1, t2):
        return tuple((a - b)/(a+b) for a, b in zip(t1, t2))

def checkdiff(desc, i, id1, t1, t2):
    if t1 != t2:
        print "{} differ @ {} id {}".format(desc, i, id1)
        print "\t\t{}".format(t1)
        print "\t\t{}".format(t2)
        print "\tdiff\t{}".format(tuplediff(t1, t2))
        print "\terr\t{}".format(tuplereldiff(t1, t2))
        exit(1)

Connect()

file1 = sys.argv[1]
file2 = sys.argv[2]

# Open the index file
datafile1 = OpenDataFile(file1)
datafile2 = OpenDataFile(file2)

# check that they have the array names
arrays1 = [a[0] for a in datafile1.PointData.items()]
arrays2 = [a[0] for a in datafile2.PointData.items()]

if arrays1 != arrays2:
    raise ValueError("mismatching arrays: {} vs {}".format(arrays1, arrays2))

# get the number of points
data1 = servermanager.Fetch(datafile1)
pts1 = data1.GetNumberOfPoints()

data2 = servermanager.Fetch(datafile2)
pts2 = data2.GetNumberOfPoints()

if pts1 != pts2:
    raise ValueError("number of points differs: {} vs {}".format(pts1, pts2))

# fetch the point data
pointdata1 = data1.GetPointData()
pointdata2 = data2.GetPointData()

# compare particle data
for pt in range(0, pts1):
    id1 = pointdata1.GetArray('Part id').GetValue(pt)
    id2 = pointdata2.GetArray('Part id').GetValue(pt)
    if id1 != id2:
        raise ValueError("Part id differ @ {} : {} vs {}".format(pt, id1, id2))

    pos1 = data1.GetPoint(pt)
    pos2 = data2.GetPoint(pt)

    checkdiff("Position", pt, id1, pos1, pos2)

    for ar in arrays1:
        val1 = pointdata1.GetArray(ar).GetTuple(pt)
        val2 = pointdata2.GetArray(ar).GetTuple(pt)
        checkdiff(ar, pt, id1, val1, val2)
