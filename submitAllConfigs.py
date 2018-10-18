#!/usr/bin/python
import subprocess
import sys
from os import listdir
from os.path import splitext, basename

if len(sys.argv) != 3:
    print("Incorrect Arguments: sumbitAllConfigs.py <build_name> <num_experiments>")
    sys.exit(0)

build_name = sys.argv[1]
num_experiments = int(sys.argv[2])

# Environment setup

for config_file in listdir("./configs"):
    for mdp_file in listdir("./mdps"):
        for start_mdp_file in listdir("./startMDPs"):
            experiment_name = splitext(mdp_file)[0] + splitext(start_mdp_file)[0] + splitext(config_file)[0]
            subprocess.call("mkdir -p OutputLogs/" + experiment_name, shell=True)
            subprocess.call("qsub -t 1-{0} -N {1} -o OutputLogs/{1}/ -e OutputLogs/{1}/ runJar.sh {2} {3} {4} {5}"
                            .format(num_experiments, experiment_name, build_name, mdp_file, start_mdp_file, config_file),
                            shell=True
                            )
