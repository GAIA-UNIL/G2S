if(~exist(strcat('g2s.',mexext), 'file'))
   CompileG2S; 
end

serverAddress='localhost';

% config
numberOfVariable=10;
precision=10;
phase=(1:numberOfVariable)'./numberOfVariable;
waveLength=(1:numberOfVariable)'./numberOfVariable/precision;
N=10000;
source=sin((1:N)'*waveLength'+phase');
destination=single(nan(N,numberOfVariable));
dataType=zeros(1,numberOfVariable);

conDestination=source;
conDestination(rand(size(conDestination))<0.99)=nan;

%plot(source)



%% simple echo

data=g2s('-sa',serverAddress,'-a','echo','-ti',source,'-dt',dataType);
plot(data);


%% simple unconditional simulation with QS

[data,t]=g2s('-sa',serverAddress,'-a','qs','-ti',source(:,1),'-di',destination(:,1),'-dt',dataType(:,1),'-k',1.5,'-n',50,'-s',100);
plot(data);
disp(t)

%% simulation with random value at random position
data=g2s('-sa',serverAddress,'-a','qs','-ti',source(:,1),'-di',conDestination(:,3),'-dt',dataType(:,1),'-k',1.5,'-n',50,'-s',100);
plot(data);

%% specifing a kernel

% The kernel need to be define for each variable 
kernel=zeros(401,1);
kernel(201,1)=1;
kernel=exp(-0.03*bwdist(kernel));

data=g2s('-sa',serverAddress,'-a','qs','-ti',source(:,1),'-di',destination(:,1),'-dt',dataType(:,1),'-k',1.5,'-n',10,'-s',100,'-ki',kernel);
plot(data);

%% Multivariate

data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',dataType,'-k',1.5,'-n',50,'-s',100);
plot(data);

%% Multi-threaded, if supported
nbThreads=4;
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',dataType,'-k',1.5,'-n',50,'-s',100,'-j',nbThreads);
plot(data);

%% Multivariate with missing data
incompleteSource=source;
incompleteSource(rand(size(incompleteSource))>0.90)=nan;
data=g2s('-sa',serverAddress,'-a','qs','-ti',incompleteSource,'-di',destination,'-dt',dataType,'-k',1.5,'-n',50,'-s',100);
plot(data);

%% full simulation
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',dataType,'-k',1.5,'-n',50,'-s',100,'-j',4,'-fs');
plot(data);

%% full simulation
path=reshape(randperm(numel(destination)),(size(destination)));
path(rand(size(path))<0.01)=-inf;
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',dataType,'-k',1,'-n',50,'-s',100,'-sp',path,'-j',4);
plot(data);

%% full simulation with missing data
data=g2s('-sa',serverAddress,'-a','qs','-ti',incompleteSource,'-di',destination,'-dt',dataType,'-k',1.5,'-n',50,'-s',100,'-j',2,'-fs');
plot(data);

%% DS mode

data=g2s('-sa',serverAddress,'-a','ds-l','-ti',source,'-di',destination,'-dt',dataType,'-th',0.05,'-f',0.3,'-n',50,'-s',100);
plot(data);

