%load data
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/ti_3_variables.tiff');

% QS call using G2S (with dt set to two continuous and one categorical
% variable)
simulation=g2s('-a','qs','-ti',ti,'-di',nan(400,400,3),'-dt',[0,0,1],'-k',1.2,'-n',30,'-j',0.5);

%Display results 
sgtitle('Unconditional simulation');
subplot(2,3,1);
imshow(ti(:,:,1));
title('Training image dim 1');
subplot(2,3,2);
imshow(ti(:,:,2));
title('Training image dim 2');
subplot(2,3,3);
imshow(ti(:,:,3));
title('Training image dim 3');
subplot(2,3,4);
imshow(simulation(:,:,1));
title('Simulation dim 1');
subplot(2,3,5);
imshow(simulation(:,:,2));
title('Simulation dim 2');
subplot(2,3,6);
imshow(simulation(:,:,3));
title('Simulation dim 3');
colormap winter;
