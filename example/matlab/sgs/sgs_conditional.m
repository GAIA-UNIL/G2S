% Sequential Gaussian simulation conditioned to hard data, with simple and
% ordinary kriging, compared with the kriging estimate.
rng(0);
model = g2sCov.exponential('range', 40);
args = model.args();

% Hard data: 150 values taken from a reference field (normal scores, mean 0)
[ref, ~] = g2s('-a', 'fftma', '-di', nan(150, 150), '-dt', [0], args{:}, '-s', 3);
reference = ref.simulation;
data = nan(150, 150);
cells = randperm(numel(data), 150);
data(cells) = reference(cells);

[sk, ~] = g2s('-a', 'sgs', '-di', data, '-dt', [0], args{:}, '-kt', 'sk', '-m', 0, ...
              '-n', 30, '-nr', 50, '-s', 1, '-j', 0.5);
[ok, ~] = g2s('-a', 'sgs', '-di', data, '-dt', [0], args{:}, '-kt', 'ok', ...
              '-n', 30, '-nr', 50, '-s', 1, '-j', 0.5);
[krig, ~] = g2s('-a', 'kriging', '-di', data, '-dt', [0], args{:}, '-kt', 'sk');

realisations = reshape(sk.simulation, [], 50);
fprintf('hard data honoured: %d\n', max(max(abs(realisations(cells, :) - data(cells)')))  < 1e-5);

[r, c] = ind2sub(size(data), cells);
images = {reference, sk.simulation(:, :, 1), mean(sk.simulation, 3), krig.estimate};
titles = {'reference', 'SGS (SK), one realisation', 'mean of 50 SGS (SK)', 'simple kriging'};
figure;
for k = 1:4
    subplot(1, 4, k);
    imagesc(images{k}, [-2.5 2.5]); axis image off; hold on;
    plot(c, r, 'k.', 'MarkerSize', 4);
    title(titles{k});
end
