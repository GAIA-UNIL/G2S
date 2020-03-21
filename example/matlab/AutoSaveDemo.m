%load data
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');
% empty grid 
conditioning=nan(200,200); 
% fill the grid with 50 random points
conditioning(randperm(numel(conditioning),50))=ti(randperm(numel(ti),50));
%% normal call
param={'-a','qs','-ti',ti,'-di',conditioning,'-dt',[0],'-k',1.2,'-n',50,'-j',1,'-s',10};
% QS call using G2S
[simulation,idx1,~]=g2s(param{:});

%% with autoSave



id=0;
for i=1:5
    if(i>1)g2s('-kill',id);end
    id=g2s(param{:},'-submitOnly','-as',1,num2str(id));
    pause(2);
    
end

% Download data
[simulation2,idx2,~]=g2s('-waitAndDownload',id);  % '-kill' to interrupt a job

%%


%Display results 
suptitle('AutoSave');
subplot(2,3,1);
imshow(simulation);
title('Sim1');
subplot(2,3,2);
imshow(simulation2);
title('Sim2');
subplot(2,3,3);
imshow(simulation-simulation2);
title('diff');
subplot(2,3,4);
imagesc(idx1);
title('index1');
subplot(2,3,5);
imagesc(idx2);
title('index2');
subplot(2,3,6);
imagesc(idx1-idx2);
title('diff');
sum(idx1(:)-idx2(:))