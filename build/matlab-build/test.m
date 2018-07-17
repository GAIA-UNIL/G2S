if(~exist(strcat('g2s.',mexext), 'file'))
   CompileG2S; 
end

% config
source=single(imread('source.png'))/255.;
destination=single(nan.*ones(200));
set(0,'DefaultFigureWindowStyle','docked')
serverAddress='localhost';

% conditional
pourcantage=0.25; %% 0.25%
conDestination=destination;
sizeDest=length(conDestination(:));
position1=randperm(sizeDest,sizeDest*pourcantage/100);
position2=randperm(sizeDest,sizeDest*pourcantage/100);
conDestination(position1)=source(position2);

%% simple echo

data=g2s('-sa',serverAddress,'-a','echo','-ti',single(source),'-dt',zeros(1,1));
imshow(data);

%% simple unconditional simulation with QS

data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data);

%% simple unconditional simulation with QS with GPU if integrated GPU avaible
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-W_GPU');
imshow(data);

%% simulation with random value at random position
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',single(conDestination),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data);

%% simple non conditional simulation with multi TI of differente size

data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source(:,1:150)),single(rot90(source,1)),single(rot90(source(:,1:175),2)),single(rot90(source(:,1:150),3)),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data);

%% simulation with a fixed path, row path
path=zeros(200);
path(:)=(1:200*200);
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-sp',path);
imshow(data);

%% simulation with a fixed path, partial random path
path=zeros(200);
path(:)=randperm(200*200);
path(randperm(200*200,200*200/2))=-inf;
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-sp',path);
imshow(data);

%% specifing a kernel

% The kernel need to be define for each variable 
kernel=zeros(101,101);
kernel(51,51)=1;
kernel=exp(-0.1*bwdist(kernel));

data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-ki',kernel);
imshow(data);

%% multi variete

source3=cat(3,source,source,source);
destination3=cat(3,destination,destination,destination);
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source3),'-di',destination3,'-dt',zeros(1,3),'-k',1.5,'-n',50,'-s',100);
imshow(data);

%% Multi-threaded, if suported
nbThreads=4;
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-j',nbThreads);
imshow(data);

%% DS mode
data=g2s('-sa',serverAddress,'-a','ds-l','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-th',0.05,'-f',0.3,'-n',50,'-s',100);
imshow(data);

%% Categorical Mode
% creation of the image
sourceCat=repmat(eye(2),25,25);
sourceCat=cat(1,cat(2,sourceCat*1,sourceCat*2),cat(2,sourceCat*3,sourceCat*4));
imagesc(sourceCat)
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(sourceCat),'-di',single(nan.*ones(100)),'-dt',ones(1,1),'-k',1,'-n',50,'-s',100,'-j',1);
imagesc(data)
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(sourceCat),rot90(single(sourceCat),1),rot90(single(sourceCat),2),rot90(single(sourceCat),3),'-di',single(nan.*ones(100)),'-dt',ones(1,1),'-k',1,'-n',50,'-s',100,'-j',1);
imagesc(data)

%% G2S interface options
% async submission
id=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-submitOnly');
disp(id);
pause(5)
% progression check
progression=g2s('-sa',serverAddress,'-statusOnly',id);
disp(progression)
% Download data
data=g2s('-sa',serverAddress,'-waitAndDownload',id);  % '-kill' to interrupt a job
imagesc(data)

%% silent mode
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-silent');
imagesc(data)

%% without timeout
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-noTO');
imagesc(data)
%% shutdown the server
g2s('-sa',serverAddress,'-shutdown');





