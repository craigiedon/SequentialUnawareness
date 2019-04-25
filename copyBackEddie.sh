#!/bin/bash
set -euo pipefail

eddie_loc="s0929508@eddie3.ecdf.ed.ac.uk"

if [ "$#" -ne 1 ]
then
  echo "Usage: copyBackEddie.sh <folder-destination>"
  exit 1
fi

mkdir $1
scp -r "$eddie_loc:~/logs" "$1/resultLogs"
scp -r "$eddie_loc:~/OutputLogs" "$1/outputLogs"
