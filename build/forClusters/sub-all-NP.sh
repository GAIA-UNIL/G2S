#!/bin/bash 
prefix="sub-"
algo="${0/${prefix}}"
algo=${algo%".sh"}
command="$algo $@"

logFile=$(echo $@ | grep -oP "\\-r\s*\K(\w|\/|\.)*" )
echo "$logFile"
touch $logFile

jobId=${logFile%".log"}
jobId=${jobId#"logs/"}

np_client -b -cmd "$command";

