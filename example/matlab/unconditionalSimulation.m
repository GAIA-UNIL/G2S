%load data
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');

% QS call using G2S
simulation=g2s('-a','qs','-ti',ti,'-di',nan(200,200),'-dt',[0],'-k',1.2,'-n',50,'-j',0.5);

%Display results 
sgtitle('Unconditional simulation');
subplot(1,2,1);
imshow(ti);
title('Training image');
subplot(1,2,2);
imshow(simulation);
title('Simulation');
