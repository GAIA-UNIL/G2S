% FFT-MA conditioned to hard data by kriging of the residuals, with the
% direct (all data) and the local (moving neighbourhood) kriging methods.
rng(2);
model = 0.1*g2sCov.nugget() + 0.9*g2sCov.exponential('range', 60);
args = model.args();
destination = nan(200, 200);

[ref, ~] = g2s('-a', 'fftma', '-di', destination, '-dt', [0], args{:}, '-s', 9);
reference = ref.simulation;
data = destination;
cells = randperm(numel(data), 400);
data(cells) = reference(cells);

[direct, ~] = g2s('-a', 'fftma', '-di', data, '-dt', [0], args{:}, '-nr', 20, '-s', 1, '-j', 0.5);
[local, ~] = g2s('-a', 'fftma', '-di', data, '-dt', [0], args{:}, '-nr', 20, '-s', 1, ...
                 '-km', 'local', '-n', 40, '-j', 0.5);
fprintf('kriging method chosen by auto: %s\n', direct.kriging_method);

[r, c] = ind2sub(size(data), cells);
images = {reference, direct.simulation(:, :, 1), std(direct.simulation, 0, 3)};
titles = {'reference', 'conditional fftma', 'std of 20 realisations'};
figure;
for k = 1:3
    subplot(1, 3, k);
    imagesc(images{k}); axis image off; hold on;
    plot(c, r, 'k.', 'MarkerSize', 3);
    title(titles{k});
end
