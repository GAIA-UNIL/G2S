%This code requires the G2S server to be running
% load example training image ('stone')
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');

maxK=100;
maxN=100;
gridSize=1000;

y=linspace(sqrt(2),sqrt(maxN),gridSize).^2;
x=exp(linspace(0,log(maxK),gridSize));
[xv,yv]=meshgrid(x,y);

simulation=g2s('-a','qs',...
               '-ti',ti,...
               '-di',nan(gridSize,gridSize),...
               '-dt',[0],...
               '-kvi',xv,...
               '-ni',yv,...
               '-j', '-legacy_output');

imagesc(simulation);
axis image;
colormap parula;

x_labels=arrayfun(@(v) sprintf('%.1f',v),x,'UniformOutput',false);
y_labels=arrayfun(@(v) sprintf('%.0f',v),y,'UniformOutput',false);
spacingLabelX=max(1,floor(gridSize/20));
spacingLabelY=max(1,floor(gridSize/20));

xTickPositions=round(linspace(1,gridSize,length(x)));
yTickPositions=round(linspace(1,gridSize,length(y)));
xTickIndex=1:spacingLabelX:numel(xTickPositions);
yTickIndex=1:spacingLabelY:numel(yTickPositions);
xticks(xTickPositions(xTickIndex));
yticks(yTickPositions(yTickIndex));
xticklabels(x_labels(xTickIndex));
yticklabels(y_labels(yTickIndex));

xlabel('k');
ylabel('n');
title('QS calibration map');
colorbar;
