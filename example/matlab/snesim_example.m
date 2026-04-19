% load example training image ('strebelle')
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff');

% SNESIM call using G2S
simulation=g2s('-a','snesim',...
               '-ti',ti,...
               '-di',nan(1000,1000),...
               '-dt',[1],...
               '-j',0.5,...
               '-mg',4,...
               '-tpl',3);

% Display results
sgtitle('SNESIM unconditional simulation');
subplot(1,2,1);
imshow(ti,[]);
title('Training image');
subplot(1,2,2);
imshow(simulation,[]);
title('Simulation');
