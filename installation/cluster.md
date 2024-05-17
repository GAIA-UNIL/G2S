# Server autostart

When running the server on a workstation, sometimes an automatic reboot may be needed (e.g., update, power cut, etc). To make matters worse, we may need to login to start again the server. In addition to machine reboots, sometimes servers can crash.

On Linux machines, it is possible to run the software in the background and launch it automatically at start-up. Here is the installation for Ubuntu, but it's very simple to adapt it for other systems.

Many solutions exist with cron, etc., but my favorite one is with systemd

1. First follow the standard Linux installation.
2. Test if everything works correctly.
3. Create a new service, let's call it G2S.service, to do so create a new file `/etc/systemd/system/G2S.service`
4. Paste the following code and it's done:
```
	[Unit]
	Description=G2S service
	After=network-online.target
	StartLimitIntervalSec=0

	[Service]
	Type=simple
	Restart=always
	RestartSec=1
	User=tesla-k20c
	# update {pathToTheServer} with the path where "./server" is located,
	# intel-build or c++-build, regarding your choice 
	WorkingDirectory={pathToTheServer}
	# if you use the Intel version, you probably want to load the libraries
	# like : /bin/bash -c "source /opt/Intel/parallel_studio_xe_2019/bin/psxevars.sh intel64; ./server -kod -age 864000"
	ExecStart=/bin/bash -c "./server -kod -age 864000"
	# -age 864000	==> keep the file for 10 days
	# -kod		==> don't remove old files in case of reboot or crash

	[Install]
	WantedBy=multi-user.target
```
5. Start the service with `systemctl start G2S`
6. Once everything works flawlessly, set it to start automatically at boot with `systemctl enable G2S`
7. Enjoy!

# Cluster

In this section, I will give you a brief overview of how to set up G2S on a cluster. In this mode, the computation is not distributed through the cluster, but runs on a single cluster node.

The installation on a cluster can be a quite difficult task. Please don't hesitate to ask for help from the team that manages the cluster.

The installation really depends on what the cluster team allows you to do or not.

## Installation of libraries

In the best-case scenario, fftw3 is installed in a non-standard path (you may not find two clusters that have the stuff at the same place. Why? To make your life harder, of course! 🤪). Potentially you can have the Intel compiler installed with all the libraries. Please refer to your cluster documentation for this.

Probably ZMQ is not already installed, and don't expect that they will install it for you. Also, jsoncpp is most probably missing too. In order to make life a bit easier for you, I prepared a small script that downloads and compiles all of these libraries and even modifies the Makefile.

1. Log in to a computation node (or submit a task).
2. Execute the script in `build/forClusters/compileExternalLibs.sh` (I assume that the front and the computational nodes are from the same generation, in other cases you can run into some unsupported instruction issues).
3. Wait until it's finished, take this time to pray 🙏 to the god of the clusters.

## Compilation of G2S

The previous script updates the Makefile automatically, so refer to the standard Linux installation. ⚠️ Be careful to compile in a computational node and not on the front node; otherwise, you can have lots of painful issues later.

## Execute the G2S server

Take a deep breath and don't give up; it's almost finished!

Now the challenging part starts. It is highly dependent on how lucky you are and how much energy you want to spend to make it work. But keep this in mind, spending some time now can make you win a lot later.

### Simple, ugly, and inefficient solution

You can use this solution if you don't have a queuing system. You really don't? Are you sure you are on a cluster?

1. You simply start the G2S server on a node.
2. Once it's running, connect to the node using an ssh tunnel: `ssh -L 8128:theNodeName:8128 clusterAddress`, this will redirect your localhost to the computation node.
3. Then run any G2S computation you like on your own computer.
4. Once you have all your computation finished, shut down the server with the '-shutdown' argument.
⚠️ Although this solution looks simple, in reality, it's dangerous and relatively hard to use on a busy cluster. In fact, when you want to run the server, you will probably finish in the waiting queue. If you are paying for the computations, you would be charged even when it isn't computing any simulation because the node is busy (waiting for jobs). And if you are not charged, you still use resources that other people may need. Furthermore, with this solution, it's extremely difficult to distribute many jobs across all the clusters.
Using nodeParallel could be a (smoother) solution too, in particular for multiple node submission.

### The way to go

The goal here is to create a task for each simulation and to add it to the queue. Again each cluster is different, some use PBS (qsub), some use SLURM(sbatch, srun),... (ok, it's not the place to debate about this). We need to convert each call to an algorithm that doesn't starts the computation, but instead submits it in the queue.

⚠️ currently computation interruption does not work on some clusters (Ctrl+C as well as '-kill'). Therefore, you need to manage job interruptions with the submission queue system. I will try to fix this issue as soon as I can, however it's not on my priority list now.

1. If the cluster uses PBS or SLURM, check the related file in the forClusters directory. If it uses another system, check one and adapt for your situation. I advise asking the cluster team to help you with this, and they are paid to do that. (please contact me only if they couldn't find out how to make it work)
2. if needed, adapt configureCluster.sh
3. execute configureCluster.sh
4. Open a tunnel to the front node ssh -L 8128:clusterAddress:8128 clusterAddress
5. start the server ./server -kod -age 864000 (adjust the 864000 regarding how long the file needs to be stored on the cluster (here 10 days))
6. Submit the jobs
	once all the jobs are submitted you can close the connection (it is possible that you run into trouble if you run many thousands of tasks at the same time, in that case keep the connection open)

### even better

If you are lucky you may be able to keep the server running all the time on the front node. Then ....

You can try to run server in demons -d, if it's authorized. So you can always cut the connection even if you submit one million tasks at the same time.

Then... if you’re even luckier you can open a port on the front node (here 8128). Don't hesitate to ask the support if they can do this favor for you. If you can, 1) invest in a good bottle of wine for the support department, 2) If you are that lucky you should play lottery!

So, once the port is open and the server are executed as a deamon, you can close all connections, and simply put the address of the front node in each call to g2s '-sa',clusterAddress

## How to efficiently use it

Once the server runs on the front node use '‑submitOnly','‑statusOnly' and '‑waitAndDownload' to manage your computations. The G2S server is coming with its own submission queue and dependency manager. Because the dependencies are unknown, each job runs sequentially by default. To over pass this limitation we can simply specify dependency as :'‑after',0, so each job can start in parallel.

A standard job submission should look like id=g2s(... usualParameter,'‑statusOnly','‑after',0).
and to get the final result sim=g2s('‑waitAndDownload',id)

## Using nodeParallel
NodeParallel (NP) allow to forward command from one node to another, in our context from the login node to some worker nodes (it require a shared file system). Therefore we can use NP to forward computation command to computation nodes without making a new task in the submission queue. This is particularly interesting when we have lots of short jobs (it allows removing the initialization for each task), or when we have significantly more jobs than the number of tasks that we are allowed to submit to the queue.

To use it simple download NP from its github page, follow the instruction to install and add it to the PATH (at least each time you want to use it :) ). Then adapt the compilation for use on cluster using ./configureCluster.sh NP available in the forClusters folder. This configuration is not compatible with a queue submission one, so if you plan to use both strategies in parallel you need to duplicate the G2S installation. Run the NP and G2S servers, both on the login node.

1. You can run NP in demon mode ./server -d
2. When running the G2S server, add the -maxCJ n flag, with n between the number and few(<4x) times the number of jobs you plan to run in parallel.
3. Start (and you can even finish) to submit G2S tasks. (Refer to the previous sections to communicate with the login node, ssh tunneling, open ports on the login node ...). Don't forget the '-after',0 to allow parallel computations.
4. Finally, run workers by submitting tasks to the normal submission queue(or directly to the computation nodes ) using the np_client -sa $HOSTNAME -sac -w in the folder that contain the G2S server. With $HOSTNAME the name or IP of the login node.

