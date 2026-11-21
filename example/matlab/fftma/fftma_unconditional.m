% Unconditional simulation with FFT-MA and with the spectral method, and a
% periodic field (-csim) that tiles without seams.
model = g2sCov.matern('range', [50 20], 'angle', 60, 'nu', 1.5);
args = model.args();
destination = nan(256, 384);

[fftma, ~] = g2s('-a', 'fftma', '-di', destination, '-dt', [0], args{:}, '-nr', 2, '-s', 1, '-j', 0.5);
[spectral, ~] = g2s('-a', 'spectral', '-di', destination, '-dt', [0], args{:}, '-nr', 2, '-s', 1, '-j', 0.5);
[periodic, ~] = g2s('-a', 'fftma', '-di', destination, '-dt', [0], args{:}, '-csim', '-s', 1, '-j', 0.5);
fprintf('extended grid used by fftma: %s\n', fftma.extended_grid);

figure;
subplot(1, 3, 1); imagesc(fftma.simulation(:, :, 1)); axis image off; title('fftma');
subplot(1, 3, 2); imagesc(spectral.simulation(:, :, 1)); axis image off; title('spectral');
subplot(1, 3, 3); imagesc(repmat(periodic.simulation, 1, 2)); axis image off; title('fftma -csim, tiled twice');
