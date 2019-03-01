#!/usr/bin/python
#
# Show metadata information about hotfiles
#
# Usage: scripts/hotinfo.py file1 [file]...

import sys
from struct import unpack, calcsize
from operator import sub

# version, buffer_count, particle_count, body_count, numOpenBoundaries, reserved1, iterations, t, dt, reserved2 
header_enc = '@IIIII48xLdf12x'

# name len, name, element_size, num_buffers
buffer_enc = '@I64sII'

for arg in sys.argv[1:]:
    print("File: {}".format(arg))
    with open(arg, "rb") as f1:
        sz = calcsize(header_enc)
        h1 = unpack(header_enc, f1.read(sz))
        print("Version {}, {} buffers, {} particles, {} bodies, {} open boundaries, {} iterations, time {}/{}".format(*h1))
        nbufs = h1[1]
        nparts = h1[2]
        nbodies = h1[3]
        for i in range(nbufs):
            sz = calcsize(buffer_enc)
            buf = f1.read(sz)
            if (len(buf) == 0):
                print("End of file reached with pending buffers (ephemeral buffers were not stored)")
                break
            h1 = unpack(buffer_enc, buf)
            bufname = h1[1][:h1[0]]
            elsize = h1[2]
            bufcount = h1[3]
            print("Buffer: {} ({}), element size {}, count {}".format(bufname, h1[0], elsize, bufcount))
            sz = elsize*nparts
            buf1 = f1.read(sz) # skip

        if nbodies > 0:
            raise NotImplementedError("no support for moving bodies yet")
