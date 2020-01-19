if(~exist(strcat('g2s.',mexext), 'file'))
   CompileG2S; 
end

% config
source=single(imread('../TrainingImages/source.png'))/255.;
destination=single(nan.*ones(200));
set(0,'DefaultFigureWindowStyle','docked')
serverAddress='tesla-k20c.gaia.unil.ch';

% conditional
pourcantage=0.25; %% 0.25%
conDestination=destination;
sizeDest=length(conDestination(:));
position1=randperm(sizeDest,sizeDest*pourcantage/100);
position2=randperm(sizeDest,sizeDest*pourcantage/100);
conDestination(position1)=source(position2);

%Categorical
sourceCat=imread('../TrainingImages/gobi_dune.png');
%denoise
%sourceCat=imagesc(colfilt(sourceCat, [5 5], 'sliding', @mode));
sourceCat=sourceCat(1:end/2,1:end/2);
sourceCat=single(sourceCat);

%% simple echo

data=g2s('-sa',serverAddress,'-a','echo','-ti',source,'-dt',zeros(1,1));
imshow(data);


%% simple unconditional simulation with QS
destination=source;
destination(50,75)=nan;

destination(rand(size(destination))<0.98)=nan;
%%

[data,t]=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1,'-n',50,'-s',100);
imshow(data);
disp(t)

%% simple unconditional simulation with QS with GPU if integrated GPU avaible
[data,t]=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1,'-n',50,'-s',100,'-W_GPU');
imshow(data);
disp(t)

%% simple unconditional simulation with QS with CUDA-GPU if avaible
[data,t]=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-W_CUDA',0);
imshow(data);
disp(t)

%% simulation with random value at random position
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',single(conDestination),'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data);

%% simple non conditional simulation with multi TI of differente size

data=g2s('-sa',serverAddress,'-a','qs','-ti',single(source(:,1:150)),single(rot90(source,1)),single(rot90(source(:,1:175),2)),single(rot90(source(:,1:150),3)),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data);

%% simulation with a fixed path, row path
path=zeros(200);
path(:)=(1:200*200);
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-sp',path);
imshow(data);

%% simulation with a fixed path, partial random path
path=zeros(200);
path(:)=randperm(200*200);
path(randperm(200*200,200*200/2))=-inf;
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-sp',path);
imshow(data);

%% specifing a kernel

% The kernel need to be define for each variable 
kernel=zeros(101,101);
kernel(51,51)=1;
kernel=exp(-0.1*bwdist(kernel));

data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-ki',kernel);
imshow(data);

%% Multivariate
source3=cat(3,source,source,source);
destination3=cat(3,destination,destination,destination);
[data,t]=g2s('-sa',serverAddress,'-a','qs','-ti',source3,'-di',destination3,'-dt',zeros(1,3),'-k',1.5,'-n',50,'-s',100);
t
imshow(data);

%% Multi-threaded, if supported
nbThreads=4;
[data2,t]=g2s('-sa',serverAddress,'-a','qs','-ti',data,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-j',nbThreads);
t
imshow(data);

%% With missing data
incompleteSource=source;
incompleteSource(repmat(rand(size(incompleteSource,1),size(incompleteSource,2))>0.90,1,1,1))=nan; % remove randomly 10% of the data
data=g2s('-sa',serverAddress,'-a','qs','-ti',incompleteSource,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100);
imshow(data);

%% Multivariate with missing data
incompleteSource3=cat(3,source,source,source);
destination3=cat(3,destination,destination,destination);
%incompleteSource3(repmat(rand(size(incompleteSource3,1),size(incompleteSource3,2))>0.90,1,1,size(incompleteSource3,3)))=nan; % remove randomly 10% of the data
incompleteSource3(rand(size(incompleteSource3))>0.90)=nan; % remove randomly 10% of the data
data=g2s('-sa',serverAddress,'-a','qs','-ti',incompleteSource3,'-di',destination3,'-dt',zeros(1,3),'-k',1.5,'-n',50,'-s',100);
imshow(data);

%% full simulation
data=g2s('-sa',serverAddress,'-a','qs','-ti',source3,'-di',destination3,'-dt',zeros(1,3),'-k',1.5,'-n',50,'-s',100,'-j',4,'-fs');
imshow(data);

%% full simulation
path=reshape(randperm(200*200*3),200,200,3);
path(rand(size(path))<0.01)=-inf;
data=g2s('-sa',serverAddress,'-a','qs','-ti',source3,'-di',destination3,'-dt',zeros(1,3),'-k',1.5,'-n',50,'-s',100,'-sp',path,'-j',4);
imshow(data);

%% full simulation with missing data
data=g2s('-sa',serverAddress,'-a','qs','-ti',incompleteSource3,'-di',destination3,'-dt',zeros(1,3),'-k',1.5,'-n',50,'-s',100,'-j',2,'-fs');
imshow(data);

%% DS mode

data=g2s('-sa',serverAddress,'-a','ds-l','-ti',source,'-di',destination,'-dt',zeros(1,1),'-th',0.05,'-f',0.3,'-n',50,'-s',100);
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

%% test stack job
%stak 10 jobs

% ids={uint32(0)};
% for i=1:10
%     ids{end}
%     id=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100+i,'-submitOnly','-j',6,'-after',ids{end});
%     ids{i}=id;
% end

% %%
% ids={uint32(0)};
% for i=1:10
%     ids{end}
%     id=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100+i,'-submitOnly','-j',6);
%     ids{i}=id;
% end


% %%
% [bigTi,t]=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',nan(512),'-dt',zeros(1,1),'-k',1,'-n',50,'-s',100,'-j',6);
% imshow(bigTi);
% disp(t)
% %%
% [data,t]=g2s('-sa',serverAddress,'-a','qs','-ti',bigTi(1:end/2,1:end/2),bigTi(1:end/2,end/2+1:end),bigTi(end/2+1:end,1:end/2),bigTi(end/2+1:end,end/2+1:end),...
%     '-di',destination,'-dt',zeros(1,1),'-k',4,'-n',50,'-s',100);
% imshow(data);
% disp(t)
% %%
% [data,t]=g2s('-sa',serverAddress,'-a','qs','-ti',bigTi(1:end/2,1:end/2),bigTi(1:end/2,end/2+1:end),bigTi(end/2+1:end,1:end/2),bigTi(end/2+1:end,end/2+1:end),...
%     '-di',destination,'-dt',zeros(1,1),'-k',4,'-n',50,'-s',100,'-far');
% imshow(data);
% disp(t)


