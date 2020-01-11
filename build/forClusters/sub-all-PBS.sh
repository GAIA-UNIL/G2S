#!/bin/bash 
prefix="sub-"
algo="${0/${prefix}}"
algo=${algo%".sh"}
OMP_SETTINGS="OMP_PLACES=cores; OMP_PROC_BIND=spread";
#command="source ./eo_set_env.sh > /dev/null; $OMP_SETTINGS $algo $@"
command="$algo $@"

#account_id=$(echo $@ | grep -oP "\\-account\s+\K\w+")
wall_time=$(echo $@ | grep -oP "\\-walltime\s*\K(\w|\/|\.|\:)*" )

#numberCore=$(echo $@ | grep -oP "\\-j\s+\K\w+" || echo "1" )
numberCore=$(echo $@ | grep -oP "\\-j\s+\K([0-9]|\ |\.)*[^\ \-]" |tr ' ' '*' |bc || echo "1" )
logFile=$(echo $@ | grep -oP "\\-r\s*\K(\w|\/|\.)*" )
echo "$logFile"
touch $logFile

jobId=${logFile%".log"}
jobId=${jobId#"logs/"}

echo "\"$command\""

export G2S_COMMAND_2_RUN=$command

qsub -N "$jobId" -d . -l nodes=1:ppn=2 -l walltime=$wall_time -I -x ./runAny.sh -v G2S_COMMAND_2_RUN

exit