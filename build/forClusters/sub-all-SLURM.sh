#!/bin/bash 
prefix="sub-"
algo="${0/${prefix}}"
algo=${algo%".sh"}
OMP_SETTINGS="OMP_PLACES=cores; OMP_PROC_BIND=spread";
#command="source ./eo_set_env.sh > /dev/null; $OMP_SETTINGS $algo $@"
command="source ./eo_set_env.sh > /dev/null; $algo $@"

account_id=$(echo $@ | grep -oP "\\-account\s+\K\w+")
#numberCore=$(echo $@ | grep -oP "\\-j\s+\K\w+" || echo "1" )
numberCore=$(echo $@ | grep -oP "\\-j\s+\K([0-9]|\ |\.)*[^\ \-]" |tr ' ' '*' |bc || echo "1" )
logFile=$(echo $@ | grep -oP "\\-r\s*\K(\w|\/|\.)*" )
echo "$logFile"
touch $logFile

jobId=${logFile%".log"}
jobId=${jobId#"logs/"}
#echo $account_id
#echo $numberCore
# echo $command
# eval $command

if [[ $account_id ]]; then
	echo "sbatch -W --account $account_id --job-name \"G2S_$jobId\" --chdir . --nodes 1 --ntasks 1 --cpus-per-task \"$numberCore\" --wrap \"$command\""
	sbatch -W --account $account_id --job-name "$jobId" --chdir . --nodes 1 --ntasks 1 --cpus-per-task "$numberCore"  --wrap "$command"
#	echo "finish"
fi
