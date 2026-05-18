% This code requires the G2S server to be running.
% Load example categorical training image.
url = 'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff';
try
    ti = imread(url);
catch
    localTi = websave(fullfile(tempdir, 'strebelle.tiff'), url);
    ti = imread(localTi);
end
ti = single(ti);
di = single(nan(size(ti)));

[simulation, index] = g2s('-a', 'DirectSampling', ...
                          '-ti', ti, ...
                          '-di', di, ...
                          '-dt', [1], ...
                          '-th', 0.12, ...
                          '-f', 0.4, ...
                          '-n', 48, ...
                          '-j', 1.00001, ...
                          '-s', 321);

figure;
sgtitle('Native DS categorical unconditional simulation');
subplot(1, 2, 1);
imagesc(ti);
axis image off;
title('Strebelle TI');
subplot(1, 2, 2);
imagesc(simulation);
axis image off;
title('Native DS');
colormap(parula);
