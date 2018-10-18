#!/bin/sh
#$ -cwd
#$ -l h_rt=20:00:00
#$ -l h_vmem=8G

# Load modules command?
. /etc/profile.d/modules.sh

# Use java module
module load java

# Run the program
# $1 - JAR name
# $2 - MPD File Name
# $3 - Start MDP File Name
# $4 - Config File Name
java -Xmx4G -jar -Djava.library.path=$1/lib $1/lib/$1.jar $2 $3 $4 $SGE_TASK_ID
