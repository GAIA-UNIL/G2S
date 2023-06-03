%load data
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');
% empty grid 
conditioning=nan(200,200); 
% fill the grid with 50 random points
conditioning(randperm(numel(conditioning),50))=ti(randperm(numel(ti),50));

% QS call using G2S
simulation=g2s('-a','qs','-ti',ti,'-di',conditioning,'-dt',[0],'-k',1.2,'-n',50,'-j',0.5);

%Display results 
sgtitle('Conditional simulation');
subplot(1,3,1);
imshow(ti);
title('Training image');
subplot(1,3,2);
imshow(conditioning);
title('Conditioning');
subplot(1,3,3);
imshow(simulation);
title('Simulation');
