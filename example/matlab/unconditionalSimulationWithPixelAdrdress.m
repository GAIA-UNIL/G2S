%load data
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');
tis={ti, rot90(ti,2),rot90(ti,3)};

% QS call using G2S
[simulation,position,t]=g2s('-a','qs','-ti',tis{:},'-di',nan(200,200),'-dt',[0],'-k',1.2,'-n',50,'-j',0.5);

%% Display results 
sgtitle('Unconditional simulation');
subplot(3,2,1);
imshow(ti);
title('Training image');
subplot(3,2,2);
imshow(simulation);
title('Simulation');

subplot(3,2,3);
imageIdx=mod(position,length(tis));
imagesc(imageIdx);
title('image id');
subplot(3,2,4);
Lindex=idivide(position,length(tis));
imagesc(Lindex);
title('linear index');

tisSizes=nan(length(tis),2);
for i=1:length(tis)
    tisSizes(i,:)=size(tis{i});
end

[x,y]=ind2sub(tisSizes(imageIdx(:)+1,:),Lindex);
subplot(3,2,5);
imagesc(x);
title('x index');
subplot(3,2,6);
imagesc(y);
title('y index');
