if(~exist(strcat('g2s.',mexext), 'file'))
   CompileG2S; 
end

source=imread('source.png');
set(0,'DefaultFigureWindowStyle','docked')

serverAddress='localhost';

%% simple echo

data=g2s('-sa',serverAddress,'-a','echo','-ti',single(source),'-dt',zeros(1,1));
imshow(data/255.);

%% simple non conditional simulation with QS

data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',single(nan.*ones(200)),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data/255.);

%with GPU is integrated GPU avaible
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',single(nan.*ones(200)),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-W_GPU');
imshow(data/255.);

%% simple conditional simulation with QS

pourcantage=0.25; %% 0.25%
dest=nan.*ones(200);
sizeDest=length(dest(:));

position=randperm(sizeDest,sizeDest*pourcantage/100);

dest(position)=source(position);

% simuulation of the source
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',single(dest),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data/255.);

% simulation with random value at random position
dest=dest*nan;
position2=randperm(sizeDest,sizeDest*pourcantage/100);
dest(position2)=source(position);

% simuulation of the source
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',single(dest),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data/255.);

%% simple non conditional simulation with multi TI of differente size

data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source(:,1:150)),single(rot90(source,1)),single(rot90(source(:,1:175),2)),single(rot90(source(:,1:150),3)),'-di',single(nan.*ones(200)),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data/255.);

%% simulation with a fixed path

% row path
path=zeros(200);
path(:)=(1:200*200);
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',single(nan.*ones(200)),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-sp',path);
imshow(data/255.);

% partial random path
path=zeros(200);
path(:)=randperm(200*200);
path(randperm(200*200,200*200/2))=-inf;
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',single(nan.*ones(200)),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-sp',path);
imshow(data/255.);

%% specifing a kernel

% ? kernel need to be define for each variable 

kernel=zeros(101,101);
kernel(51,51)=1;
kernel=exp(-0.1*bwdist(kernel));

data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',single(nan.*ones(200)),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-ki',kernel);
imshow(data/255.);

%% multi variete

source3=cat(3,source,source,source); %% need to fine better example
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source3),'-di',single(nan.*ones(200,200,3)),'-dt',zeros(1,3),'-k',2,'-n',50,'-s',100);
imshow(data/255.);


%% multithreaded if suported
nbThreads=4;
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source),'-di',single(nan.*ones(200)),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-j',nbThreads);
imshow(data/255.);


%% ds mode

data=g2s('-sa',serverAddress,'-a','ds-l','-ti',single(source),'-di',single(nan.*ones(200)),'-dt',zeros(1,1),'-th',10,'-f',0.3,'-n',50,'-s',100);
imshow(data/255.);


%% Categorical Mode
% creation of the image
sourceCat=repmat(eye(2),25,25);
sourceCat=cat(1,cat(2,sourceCat*1,sourceCat*2),cat(2,sourceCat*3,sourceCat*4));
imagesc(sourceCat)
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(sourceCat),'-di',single(nan.*ones(100)),'-dt',ones(1,1),'-k',1,'-n',50,'-s',100,'-j',1);
imagesc(data)
data=g2s('-sa',serverAddress,'-a','qs','-ti',single(sourceCat),rot90(single(sourceCat),1),rot90(single(sourceCat),2),rot90(single(sourceCat),3),'-di',single(nan.*ones(100)),'-dt',ones(1,1),'-k',1,'-n',50,'-s',100,'-j',1);
imagesc(data)

