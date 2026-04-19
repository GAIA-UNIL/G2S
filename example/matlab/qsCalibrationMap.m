%This code requires the G2S server to be running
% load example training image ('stone')
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');

maxK=100;
maxN=100;
size=1000;

y=linspace(sqrt(2),sqrt(maxN),size).^2;
x=exp(linspace(0,log(maxK),size));
[xv,yv]=meshgrid(x,y);

simulation=g2s('-a','qs',...
               '-ti',ti,...
               '-di',nan(size,size),...
               '-dt',[0],...
               '-kvi',xv,...
               '-ni',yv,...
               '-j');

imagesc(simulation);
axis image;
colormap parula;

x_labels=arrayfun(@(v) sprintf('%.1f',v),x,'UniformOutput',false);
y_labels=arrayfun(@(v) sprintf('%.0f',v),y,'UniformOutput',false);
spacingLabelX=max(1,floor(size/20));
spacingLabelY=max(1,floor(size/20));

xticks(round(linspace(1,size,length(x))));
yticks(round(linspace(1,size,length(y))));
xticklabels(x_labels);
yticklabels(y_labels);
xticks(xticks(1:spacingLabelX:end));
yticks(yticks(1:spacingLabelY:end));
xticklabels(x_labels(1:spacingLabelX:end));
yticklabels(y_labels(1:spacingLabelY:end));

xlabel('k');
ylabel('n');
title('QS calibration map');
colorbar;
