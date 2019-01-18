#!/bin/sh

# Run a specific problem for maxiter iterations and check against the reference
# Syntax: check-problem problem [reference_suffix] [maxiter]
# default reference_suffix is 'reference'
# default maxiter is 1000

abort() {
	echo "$@" >&2
	exit 1
}

sfx=check
problem="$1"
ref="$2"
maxiter="$3"

[ -z "$ref" ] && ref=reference
[ -z "$maxiter" ] && maxiter=1000

echo "Testing ${problem} ..."
outdir="tests/${problem}_${sfx}"
refdir="tests/${problem}_${ref}"

[ -d "$refdir" ] || abort "Reference directory $refdir for problem $problem not found"

rm -rf "$outdir"

make $problem && ./GPUSPH --dir "$outdir" --maxiter $maxiter || abort "Failed!"
diff -q "${outdir}/data" "${refdir}/data" || abort "$problem differs!"
