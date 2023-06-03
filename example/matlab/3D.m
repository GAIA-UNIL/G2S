%% 3D multivariate

clear;home;clf
%download TI (first variable, continuous)
ti1=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/fold_continuous.tiff');
%create second variable (categorical) by thresholding the first
ti2=ti1>0.5;
ti=cat(4,ti1,ti2);
ti=double(ti(1:80,1:50,101:120,:));

%display TI
subplot(2,2,1)
[X,Y,Z] = meshgrid(1:size(ti,2),1:size(ti,1),1:size(ti,3));
slice(X,Y,Z,ti(:,:,:,1),[1 size(ti,2)],[1 size(ti,1)],[1 size(ti,3)])
shading flat
camproj('perspective')
axis equal
title('TI variable 1');

subplot(2,2,2)
[X,Y,Z] = meshgrid(1:size(ti,2),1:size(ti,1),1:size(ti,3));
slice(X,Y,Z,ti(:,:,:,2),[1 size(ti,2)],[1 size(ti,1)],[1 size(ti,3)])
shading flat
camproj('perspective')
axis equal
title('TI variable 2');

% QS call using G2S
tic
simulation=g2s('-a','qs','-ti',ti,'-di',nan(80,50,10,2),'-dt',[0,1],'-k',1.2,'-n',{20 10},'-j',0.5);
toc

%display simulation
subplot(2,2,3)
[X,Y,Z] = meshgrid(1:size(simulation,2),1:size(simulation,1),1:size(simulation,3));
slice(X,Y,Z,simulation(:,:,:,1),[1 size(simulation,2)],[1 size(simulation,1)],[1 size(simulation,3)])
shading flat
camproj('perspective')
axis equal
title('Simulation variable 1');

subplot(2,2,4)
[X,Y,Z] = meshgrid(1:size(simulation,2),1:size(simulation,1),1:size(simulation,3));
slice(X,Y,Z,simulation(:,:,:,2),[1 size(simulation,2)],[1 size(simulation,1)],[1 size(simulation,3)])
shading flat
camproj('perspective')
axis equal
title('Simulation variable 2');
sgtitle('3D Simulation with two variables');
