#!/bin/bash

# gat script dir
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"


if [[ "$1" == "SLURM" || "$1" == "slurm" ]]; then
	submissionScriptName="sub-all-SLURM.sh"
fi

if [[ "$1" == "PBS" || "$1" == "pbd"  || "$1" == "qsub" ]]; then
    submissionScriptName="sub-all-PBS.sh"
    echo -e '#!/bin/bash \npwd\n$G2S_COMMAND_2_RUN\nexit\n' >"$DIR/../c++-build/runAny.sh"
    echo -e '#!/bin/bash \npwd\n$G2S_COMMAND_2_RUN\nexit\n' >"$DIR/../intel-build/runAny.sh"
    chmod +x "$DIR/../c++-build/runAny.sh" "$DIR/../intel-build/runAny.sh"
fi

if [[ "$1" == "NP" || "$1" == "np"  || "$1" == "nodeParallel" ]]; then
    submissionScriptName="sub-all-NP.sh"
fi

if [ -n "$submissionScriptName" ]; then
	if [ -f $DIR/../algosName.config.bk ]; then
		cp $DIR/../algosName.config.bk $DIR/../algosName.config
	fi
	mv $DIR/../algosName.config $DIR/../algosName.config.bk
	cat $DIR/../algosName.config.bk | sed 's/.\//.\/sub-/g' | sed 's/	/.sh	/2' >> $DIR/../algosName.config

	cat $DIR/../algosName.config.bk | cut -f2 | while read line 
	do
		ln -fs $DIR/$submissionScriptName "$DIR/../c++-build/sub-${line:2}.sh"
		ln -fs $DIR/$submissionScriptName "$DIR/../intel-build/sub-${line:2}.sh"
	done

	ln -fs $(which bash) "$DIR/../c++-build/bash"
	ln -fs $(which bash) "$DIR/../intel-build/bash"

	echo -e '/bin/bash\n$G2S_COMMAND_2_RUN\nexit\n' > $DIR/../c++-build/runAnny.sh
	echo -e '/bin/bash\n$G2S_COMMAND_2_RUN\nexit\n' > $DIR/../intel-build/runAnny.sh

fi

