#!/bin/sh

# Test functions and methods defined in common.sh

. scripts/common.sh

func() {
	echo "$1 $sfx $ref"
}

for_each_problem func
