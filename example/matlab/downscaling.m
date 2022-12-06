%load example training image 'Dunes gobi'
ti_fine=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/dunes_gobi.tiff');

%create an artificial coarse resolution image on the same grid 
ti_coarse = imresize(imresize(ti_fine,0.2,'nearest'),5,'nearest');

%display the full training image at both resolutions
figure(1);
tiledlayout(2,1);
nexttile;
image(ti_fine);
title('Fine TI');
nexttile;
image(ti_coarse);
title('Coarse TI');

ti_size = 500;
di_size = 200;

%crop half of the image, to be used as ti
ti = cat(3,ti_fine(1:ti_size,1:ti_size),ti_coarse(1:ti_size,1:ti_size));
%crop upper right corner to be used as di
di_coarse = ti_coarse(1:di_size,(end-di_size+1):end);
di_fine  = nan(di_size,di_size);
di = cat(3,di_fine,di_coarse);


% QS call using G2S
[simulation,index,time]=g2s('-a','qs',...
                '-ti',ti,...
                '-di',di,...
                '-dt',[0],... #Zero for continuous variables
                '-k',1.2,...
                '-n',20,...
                '-j',0.5);

figure(2);
tiledlayout(2,2)
nexttile;
image(di_coarse);
title('Coarse DI');
nexttile;
image(simulation(:,:,1));
title('Simulation');
nexttile;
image(index(:,:,1),'CDataMapping','scaled');
title('Index');
nexttile;
image(ti_fine(1:di_size,(end-di_size+1):end));
title('True image');


