% This code requires the G2S server to be running.
% Build a two-variable TI from the categorical Strebelle image.
url = 'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff';
try
    categorical = imread(url);
catch
    localTi = websave(fullfile(tempdir, 'strebelle.tiff'), url);
    categorical = imread(localTi);
end
categorical = single(categorical);
continuous = (categorical + circshift(categorical, 1, 1) + circshift(categorical, -1, 1) + ...
              circshift(categorical, 1, 2) + circshift(categorical, -1, 2)) / 5;
continuous = (continuous - min(continuous(:))) / (max(continuous(:)) - min(continuous(:)));

ti = cat(3, categorical, single(continuous));
di = single(nan(size(ti)));

[schemaResult, ~] = g2s('-a', 'ds', ...
                          '-ti', ti, ...
                          '-di', di, ...
                          '-dt', [1, 0], ...
                          '-th', 0.15, ...
                          '-f', 0.4, ...
                          '-n', 48, ...
                          '-j', 1.00001, ...
                          '-cnorm', 2.0, ...
                          '-fs', ...
                          '-s', 789);
simulation = schemaResult.simulation;
index = schemaResult.indexmap;

figure;
sgtitle('Native DS full mixed unconditional simulation');
subplot(1, 4, 1);
imagesc(ti(:, :, 1));
axis image off;
title('Categorical TI');
subplot(1, 4, 2);
imagesc(ti(:, :, 2));
axis image off;
title('Derived continuous TI');
subplot(1, 4, 3);
imagesc(simulation(:, :, 1));
axis image off;
title('DS category');
subplot(1, 4, 4);
imagesc(simulation(:, :, 2));
axis image off;
title('DS continuous');
colormap(parula);
