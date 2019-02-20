#!/usr/bin/python
#
# Compare two HotFile
#
# Usage: scripts/hotdiff.py file1 fil2

import sys
from struct import unpack, calcsize
from operator import sub

# version, buffer_count, particle_count, body_count, numOpenBoundaries, reserved1, iterations, t, dt, reserved2 
header_enc = '@IIIII48xLdf12x'

# name len, name, element_size, num_buffers
buffer_enc = '@I64sII'

with open(sys.argv[1], "rb") as f1:
    with open(sys.argv[2], "rb") as f2:
        sz = calcsize(header_enc)
        h1 = unpack(header_enc, f1.read(sz))
        h2 = unpack(header_enc, f2.read(sz))
        print("Version {}, {} buffers, {} particles, {} bodies, {} open boundaries, {} iterations, time {}/{}".format(*h1))
        if h1 != h2:
            raise ValueError(tuple(map(sub, h1, h2)))
        nbufs = h1[1]
        nparts = h1[2]
        nbodies = h1[3]
        for i in range(nbufs):
            sz = calcsize(buffer_enc)
            h1 = unpack(buffer_enc, f1.read(sz))
            h2 = unpack(buffer_enc, f2.read(sz))
            bufname = h1[1][:h1[0]]
            elsize = h1[2]
            bufcount = h1[3]
            print("Buffer: {} ({}), element size {}, count {}".format(bufname, h1[0], elsize, bufcount))
            if h1 != h2:
                raise ValueError(tuple(map(sub, h1, h2)))
            if bufcount > 1:
                raise NotImplementedError("no support for multi-component buffers yet")
            sz = elsize*nparts
            buf1 = f1.read(sz)
            buf2 = f2.read(sz)
            for offset in range(sz):
                if buf1[offset] != buf2[offset]:
                    print("\tFirst difference at particle index {}".format(offset/elsize))
                    break

        if nbodies > 0:
            raise NotImplementedError("no support for moving bodies yet")
