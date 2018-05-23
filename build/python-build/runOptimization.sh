#PBS -l nodes=11:skl:ppn=2
if [ ! -z "$PBS_O_WORKDIR" ]
then
	cd $PBS_O_WORKDIR
fi

REPETION=12;
TO_SKIP=4

if [ -z "$PBS_NODEFILE" ]
then
	echo "localhost" > test.txt 
	PBS_NODEFILE="test.txt"
fi

echo -n > python_host.txt
cat $PBS_NODEFILE | uniq > MPI_host.txt
for (( i = 0; i < $REPETION; i++ )); do
	if [[ i -lt $TO_SKIP ]]; then
		sed "/`hostname`/d" MPI_host.txt >> python_host.txt
	else
		cat MPI_host.txt >> python_host.txt
	fi
done

echo "MPI_host.txt"
cat MPI_host.txt
echo 
echo "python_host.txt"
cat python_host.txt

mpirun -machinefile MPI_host.txt -genv I_MPI_PIN off -wdir ~/githubProject/G2S/build/intel-build/ ./server -To 60 &

python3 kernelGAO.py 10000 python_host.txt

wait $!


