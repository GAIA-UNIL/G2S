% This code requires the G2S server to be running.
% Load example continuous training image.
url = 'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff';
try
    ti = imread(url);
catch
    localTi = websave(fullfile(tempdir, 'stone.tiff'), url);
    ti = imread(localTi);
end
ti = single(ti);
di = single(nan(size(ti)));

[simulation, index] = g2s('-a', 'ds', ...
                          '-ti', ti, ...
                          '-di', di, ...
                          '-dt', [0], ...
                          '-th', 0.08, ...
                          '-f', 0.35, ...
                          '-n', 40, ...
                          '-j', 1.00001, ...
                          '-s', 123);

figure;
sgtitle('Native DS continuous unconditional simulation');
subplot(1, 2, 1);
imshow(ti, []);
title('Stone TI');
subplot(1, 2, 2);
imshow(simulation, []);
title('Native DS');
