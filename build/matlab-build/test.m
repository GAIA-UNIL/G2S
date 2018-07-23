if(~exist(strcat('g2s.',mexext), 'file'))
   CompileG2S; 
end

% config
source=single(imread('../TrainingImages/source.png'))/255.;
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

%Categorical
sourceCat=imread('../TrainingImages/gobi_dune_dimitri.png');
%denoise
%sourceCat=imagesc(colfilt(sourceCat, [5 5], 'sliding', @mode));
sourceCat=sourceCat(1:end/2,1:end/2);
sourceCat=single(sourceCat);

%% simple echo

data=g2s('-sa',serverAddress,'-a','echo','-ti',single(source),'-dt',zeros(1,1));
imshow(data);


%% simple unconditional simulation with QS

[data,t]=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data);
disp(t)

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

%% Multivariate

source3=cat(3,source,source,source);
destination3=cat(3,destination,destination,destination);
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source3),'-di',destination3,'-dt',zeros(1,3),'-k',1.5,'-n',50,'-s',100);
imshow(data);

%% Multi-threaded, if supported
nbThreads=4;
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-j',nbThreads);
imshow(data);

%% DS mode
data=g2s('-sa',serverAddress,'-a','ds-l','-ti',single(source),'-di',destination,'-dt',zeros(1,1),'-th',0.05,'-f',0.3,'-n',50,'-s',100);
imshow(data);

%% Categorical Mode
% creation of the image
imagesc(sourceCat)
data=g2s('-sa',serverAddress,'-a','qs','-ti',sourceCat,'-di',destination,'-dt',ones(1,1),'-k',1,'-n',50,'-s',100,'-j');
imagesc(data)
data=g2s('-sa',serverAddress,'-a','qs','-ti',sourceCat,rot90(sourceCat,1),rot90(sourceCat,2),rot90(sourceCat,3),'-di',destination,'-dt',ones(1,1),'-k',1,'-n',50,'-s',100,'-j');
imagesc(data)

%% categorical and continous
threshold=graythresh(source);
combinedSource=cat(3,source,(source<threshold));
% the relative importance can be seted with a kernel
data=g2s('-sa',serverAddress,'-a','qs','-ti',combinedSource,'-di',nan([size(destination),size(combinedSource,3)]),'-dt',[0,1],'-k',1,'-n',50,'-s',100,'-j');
imshow(data(:,:,1))

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
imshow(data)

%% silent mode
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-silent');
imshow(data)

%% without timeout
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-noTO');
imshow(data)
%% shutdown the server
g2s('-sa',serverAddress,'-shutdown');





