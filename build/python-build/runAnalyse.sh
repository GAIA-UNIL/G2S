#PBS -l nodes=4:skl:ppn=2
if [ ! -z "$PBS_O_WORKDIR" ]
then
	cd $PBS_O_WORKDIR
fi
REPETION=6;

if [ -z "$PBS_NODEFILE" ]
then
	echo "localhost" > test.txt 
	PBS_NODEFILE="test.txt"
fi

echo -n > python_host.txt
cat $PBS_NODEFILE | uniq > MPI_host.txt
for (( i = 0; i < $REPETION; i++ )); do
	cat MPI_host.txt >> python_host.txt
done

echo "MPI_host.txt"
cat MPI_host.txt
echo 
echo "python_host.txt"
cat python_host.txt

export OMP_PLACES=cores
export OMP_PROC_BIND=close

mpirun-openmpi-mp -machinefile $PBS_NODEFILE -wdir ~/githubProject/G2S/build/intel-build/ -N 1 -x OMP_PLACES -x OMP_PROC_BIND ./server &
PID_MPIRUN=$!
python3 kernelOptimization.py python_host.txt &
PID_PYTHON_3=$!

pkill -P $PID_MPIRUN




