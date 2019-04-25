#!/bin/bash
set -euo pipefail

tar_path="build/distributions/SequentialPlanning.tar"
echo $tar_path

tar_name=$(basename $tar_path)
echo $tar_name

raw_name="${tar_name%.*}"
echo $raw_name

eddie_loc="s0929508@eddie3.ecdf.ed.ac.uk"
config_folder="eddieConfigs"
mdp_folder="eddieMDPs"
start_mdp_folder="eddieStartMDPs"

echo "Removing old files from eddie"
ssh $eddie_loc "rm -rf configs ; rm -rf mdps ; rm -rf startMDPs ; rm -rf OutputLogs ; rm -rf logs ; rm submitAllConfigs.py ; rm runJar.sh ; ls"

echo "Copying tar and scripts"
scp -r $tar_path submitAllConfigs.py runJar.sh "$eddie_loc:./"

echo "Copying configs"
scp -r $config_folder "$eddie_loc:./configs"

echo "Copying mdps"
scp -r $mdp_folder "$eddie_loc:./mdps"

echo "Copying start mdp"
scp -r $start_mdp_folder "$eddie_loc:./startMDPs"

echo "Unzipping and Submitting Jobs"
ssh $eddie_loc "tar -xvf $tar_name ; rm $tar_name ; python submitAllConfigs.py $raw_name 20"
