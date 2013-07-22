#!/bin/sh
export NAME=$(pwd | cut -d "/" -f 6)
export TYPE=$(pwd | cut -d "/" -f 7)
export EXT=png
#export EXT=jpg
#for fps in "15" "25"
for fps in "25"
do
	mencoder "mf://*.$EXT" -mf fps=$fps -ovc lavc -lavcopts vcodec=mpeg4:vpass=1:vbitrate=2160000 -o ../../video/video_${NAME}_${TYPE}_${fps}fps.avi
done

