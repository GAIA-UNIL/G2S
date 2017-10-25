if(~exist(strcat('g2s.',mexext), 'file'))
   CompileG2S; 
end

set(0,'DefaultFigureWindowStyle','docked')

N=32
source=zeros(N+1);
source(:)=mod(1:length(source(:)),2)==0;
sourceRand=source(1:N,1:N);
sourceStruct=source(1:N,1:N);

sourceRand(sourceRand==1)=randi(5,length(sourceRand(sourceRand==1)),1);
sourceStruct=sourceStruct.*kron([1,2;3,4],sourceStruct(1:N/2,1:N/2));

figure();
subplot(4,2,1);
imshow(sourceRand/5)
colormap(colorcube)
subplot(4,2,5);
imshow(sourceStruct/5)
colormap(colorcube)



%%  simulation with QS

dataRS_qs=(g2s('-sa',serverAddress,'-a','qs','-ti',single(sourceRand),'-di',single(nan.*ones(200)),'-k',1.5,'-n',50,'-s',100,'-j',20));
subplot(4,2,2);
imshow(dataRS_qs/5)
colormap(colorcube)

dataSS_qs=(g2s('-sa',serverAddress,'-a','qs','-ti',single(sourceStruct),'-di',single(nan.*ones(200)),'-k',1.5,'-n',50,'-s',100,'-j',20));
subplot(4,2,6);
imshow(dataSS_qs/5)
colormap(colorcube)


%%  simulation with DS

ths=0.01:.15:1.01;

for index=1:length(ths);
    dataRS_ds=(g2s('-sa',serverAddress,'-a','ds-l','-ti',single(sourceRand),'-di',single(nan.*ones(200)),'-th',ths(index),'-f',0.3,'-n',50,'-s',100,'-j',20));
    subplot(4,length(ths),1*length(ths)+index);
    imshow(dataRS_ds/5)
    colormap(colorcube)
end
    

%%
for index=1:length(ths);
    dataSS_ds=(g2s('-sa',serverAddress,'-a','ds-l','-ti',single(sourceStruct),'-di',single(nan.*ones(200)),'-th',ths(index),'-f',1,'-n',50,'-s',100,'-j',20));
    subplot(4,length(ths),3*length(ths)+index);
    imshow(dataSS_ds/5)
    colormap(colorcube)
end
