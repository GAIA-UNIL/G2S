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

[xx, yy] = meshgrid(single(1:size(di, 2)), single(1:size(di, 1)));
cy = single(size(di, 1) + 1) / 2;
cx = single(size(di, 2) + 1) / 2;
radius = sqrt((yy - cy).^2 + (xx - cx).^2);
radius = radius ./ max(radius(:));

rotationMap = single(radius * pi / 3);
rotationTolerance = single(ones(size(di)) * (5 * pi / 180));
scaleMap = single(0.9 + 0.25 * radius);
scaleTolerance = single(ones(size(di)) * 0.05);

[schemaResult, ~] = g2s('-a', 'DS', ...
                          '-ti', ti, ...
                          '-di', di, ...
                          '-dt', [0], ...
                          '-th', 0.08, ...
                          '-f', 0.35, ...
                          '-n', 40, ...
                          '-j', 1.00001, ...
                          '-s', 456, ...
                          '-rmi', rotationMap, ...
                          '-rti', rotationTolerance, ...
                          '-smi', scaleMap, ...
                          '-sti', scaleTolerance);
simulation = schemaResult.simulation;
index = schemaResult.indexmap;

figure;
sgtitle('Native DS transform unconditional simulation');
subplot(1, 4, 1);
imshow(ti, []);
title('Stone TI');
subplot(1, 4, 2);
imagesc(rotationMap);
axis image off;
title('Rotation center');
subplot(1, 4, 3);
imagesc(scaleMap);
axis image off;
title('Scale center');
subplot(1, 4, 4);
imshow(simulation, []);
title('Native DS');
