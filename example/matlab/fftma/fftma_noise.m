% FFT-MA with a given white noise: the noise of two realisations is returned
% (-on) and combined, as in gradual deformation, to make a continuous family
% of realisations that all have the model covariance.
model = g2sCov.spherical('range', 40);
args = model.args();
destination = nan(150, 150);

[result, ~] = g2s('-a', 'fftma', '-di', destination, '-dt', [0], args{:}, '-nr', 2, '-on', '-s', 4);
noise = result.noise;  % white noise on the extended grid, one layer per realisation

figure;
t = linspace(0, pi/2, 5);
for k = 1:5
    w = cos(t(k))*noise(:, :, 1) + sin(t(k))*noise(:, :, 2);  % still N(0, 1) white noise
    [field, ~] = g2s('-a', 'fftma', '-di', destination, '-dt', [0], args{:}, '-ni', w);
    subplot(1, 5, k);
    imagesc(field.simulation, [-2.5 2.5]); axis image off;
    title(sprintf('t = %.2f', t(k)));
end

% t = 0 gives back the first realisation
[first, ~] = g2s('-a', 'fftma', '-di', destination, '-dt', [0], args{:}, '-ni', noise(:, :, 1));
fprintf('identical to realisation 0: %d\n', isequal(first.simulation, result.simulation(:, :, 1)));

% the whole noise (both realisations, one per layer) can be given back at once
[both, ~] = g2s('-a', 'fftma', '-di', destination, '-dt', [0], args{:}, '-ni', noise);
fprintf('both realisations identical: %d\n', isequal(both.simulation, result.simulation));
