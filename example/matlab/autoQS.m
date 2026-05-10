% load example training image ('stone')
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');

% simple calibration kernel
kernel=ones(15,15);

% AutoQS call using G2S
result=g2s('-a','autoQS',...
           '-ti',ti,...
           '-ki',kernel,...
           '-dt',[0],...
           '-maxk',2,...
           '-maxn',80,...
           '-density',0.0312,0.0625,0.125,0.25,...
           '-maxIter',5000,...
           '-minIter',200,...
           '-mpow',2,...
           '-j',0.5);

result2d=squeeze(result);
while ndims(result2d)>2
    result2d=result2d(:,:,1);
end

% Display results
sgtitle('AutoQS calibration');
subplot(1,2,1);
imshow(ti);
title('Training image');
subplot(1,2,2);
imagesc(result2d);
axis image;
title('Calibration result');
colorbar;
