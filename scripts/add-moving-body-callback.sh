#!/bin/sh

abort() {
	printf "%s\n" "$1" >&2
	exit 1
}

must_exist() {
	fname="$1"
	[ -e "$fname" ] || abort "Source file '${fname}' does not exist"
}

problem_name="$1"

[ -z "$problem_name" ] && abort "Please specfiy a problem name"

dst_h="src/problems/${problem_name}.h"
dst_cc="src/problems/${problem_name}.cc"
dst_cu="src/problems/${problem_name}.cu"

must_exist "${dst_h}"

grep -q 'moving_body_dynamics_callback' "$dst_h" && abort "$dst_h already defines moving_body_dynamics_callback"
grep -q 'moving_bodies_callback' "$dst_h" && abort "$dst_h already defines the old-style moving_bodies_callback"

body_dst="$dst_cu"

[ -e "$dst_cc" ] && body_dst="$dst_cc"

signature="$(sed -n -e '/moving_body_dynamics_callback/,/);/s/\t//p' src/ProblemCore.h | tr -d ';')"

workfile="$(mktemp "${problem_name}.XXX")"

sed -e "/^};/i\
\\
\\
\tvoid\\
$(printf '%s' "${signature}" | sed 's/$/\\/') override;
" "${dst_h}" > "${workfile}"

mv "${workfile}" "${dst_h}"

echo "Moving body callback declaration added to ${dst_h}"

{
printf '\nvoid\n%s::' "$problem_name"
printf '%s' "$signature" | sed 's/\t//'
printf '\n{\n\t/* set kdata, adata, dx and dr here */\n}\n'
} >>  "${body_dst}"

echo "Moving body callback template added to ${body_dst}"
