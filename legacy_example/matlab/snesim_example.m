% This code requires the G2S server to be running.
% load example training image ('strebelle')
url = 'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff';
try
    ti = imread(url);
catch
    % Older MATLAB releases may not support direct HTTP reads with imread.
    localTi = websave(fullfile(tempdir, 'strebelle.tiff'), url);
    ti = imread(localTi);
end
ti = single(ti);

% SNESIM call using G2S
% 5 grid levels total: 4 -> 3 -> 2 -> 1 -> 0 (because -mg is the max level)
[simulation, elapsed] = g2s('-a', 'snesim', ...
                            '-ti', single(ti), ...
                            '-di', single(nan(1000, 1000)), ...
                            '-dt', [1], ...
                            '-j', 1.001, ...
                            '-mg', 4, ...
                            '-tpl', 3, '-legacy_output');

fprintf('SNESIM duration: %.3f s\n', elapsed);

% Display results
figure;
sgtitle('SNESIM unconditional simulation');
subplot(1, 2, 1);
imagesc(ti);
title('Training image (Strebelle)');
axis image off;
subplot(1, 2, 2);
imagesc(simulation);
title('Simulation');
axis image off;
colormap(parula);
