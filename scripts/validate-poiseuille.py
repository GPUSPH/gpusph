#!/usr/bin/python

# Compute the error on a Poiseuille flow simulation

import argparse, math, subprocess, os

from paraview.simple import *

parser = argparse.ArgumentParser(description="Poiseuille flow validation")
parser.add_argument('--ppH', type=int, default=32, help='particles per height')
parser.add_argument('--rho', type=float, default=1.0, help='fluid density')
parser.add_argument('--kinvisc', type=float, default=0.1, help='kinematic viscosity')
parser.add_argument('--driving-force', type=float, default=0.05, help='driving force magnitude')
parser.add_argument('--gnuplot', type=bool, default=False, help='output data for gnuplot')

args = parser.parse_args()

lz=1.0

ref_ppH=args.ppH
rho=args.rho
kinvisc=args.kinvisc
driving_force=args.driving_force

gnuplot = args.gnuplot

ppH_range = (ref_ppH/2, ref_ppH, ref_ppH*2)

def compute_poiseuille_vel(depth):
    if (abs(depth) > lz/2):
        return 0
    a = driving_force/(2*kinvisc)
    b = (lz/2)*(lz/2)
    return a*(b - depth*depth)

if gnuplot:
    pfx='# '
else:
    pfx=''

max_vel=compute_poiseuille_vel(0)
Re=lz*max_vel/kinvisc

print "%s=== Poiseuille validation ===" % pfx
print "%skinematic visc\t: %g" % (pfx, kinvisc)
print "%sdriving force\t: %g" % (pfx, driving_force)
print "%smaximum vel\t: %g" % (pfx, max_vel)
print "%sReynolds num\t: %g" % (pfx, Re)
print "%swith ppH\t: %d %d %d" % ((pfx,) + ppH_range)

subprocess.check_call(['make', 'Poiseuille'])

errors = []

for ppH in ppH_range:

    test_name="Poiseuille-%d-%g-%g-%g" % (ppH, rho, kinvisc, driving_force)

    data_dir="tests/%s" % test_name

    print "\n%sTest %s ...\n" % (pfx, test_name)

    if not os.path.exists(data_dir):
        with open(test_name + '.log', 'w') as logfile:
            subprocess.check_call(['./GPUSPH', '--ppH', str(ppH), '--dir', data_dir], stdout=logfile)

    print "%s ... done" % pfx

    vtp = OpenDataFile(data_dir + "/data/VTUinp.pvd")

    # Filter by particle type
    t = Threshold(vtp)
    t.Scalars = 'Part type'
    t.ThresholdRange = [0, 0]

    # Go to last step
    t.UpdatePipeline(vtp.TimestepValues[-1])

    fdata = servermanager.Fetch(t)
    pts = fdata.GetNumberOfPoints()

    print "%s (%d points)" % (pfx, pts)

    veldata = fdata.GetPointData().GetArray('Velocity')

    linf = 0
    l1 = 0
    l2 = 0

    if gnuplot:
        print "z\tvel\ttheory\terror\terrpct"

    for pt in range(0, pts):
        pos = fdata.GetPoint(pt)
        vel = veldata.GetTuple(pt)

        z = pos[2]
        vel_theory = ( compute_poiseuille_vel(z), 0, 0 )

        err = tuple(abs(a - b) for a, b in zip(vel, vel_theory))
        err_x = err[0]

        errpct = err_x*100/(vel_theory[0])

        linf = max(linf, err_x)
        l1 += err_x
        l2 += err_x*err_x

        if gnuplot:
            print "%g\t%g\t%g\t%g\t%g" % (z, vel[0], vel_theory[0], err[0], errpct)

    l1 = l1/pts
    l2 = math.sqrt(l2/pts)
    print "%s[%d] linf\tl1\tl2" % (pfx, ppH)
    print "%s%.2e\t%.2e\t%.2e" % (pfx, linf, l1, l2)

    errors.append([ppH, linf, l1, l2])

print "%sres\tlinf\tl1\tl2" % (pfx,)
for data in errors:
    print "%s%d\t%.2e\t%.2e\t%.2e" % (pfx, data[0], data[1], data[2], data[3])
