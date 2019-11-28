%load data
tiWithGap=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');
tiWithGap(60:140,60:140)=nan;

% QS call using G2S
simulation=g2s('-a','qs','-ti',tiWithGap,'-di',tiWithGap,'-dt',[0],'-k',1.2,'-n',25,'-j',0.5);

%Display results 
suptitle('Gap filling');
subplot(1,2,1);
imshow(tiWithGap);
title('Training image');
subplot(1,2,2);
imshow(simulation);
title('Simulation');
