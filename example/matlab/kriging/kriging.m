% Kriging of the empty cells of a grid from scattered data: estimate and
% kriging variance, with the direct (unique neighbourhood) and the local
% (moving neighbourhood) strategies.
rng(1);
model = 0.05*g2sCov.nugget() + 0.95*g2sCov.spherical('range', [80 30], 'angle', -20);
args = model.args();

[ref, ~] = g2s('-a', 'fftma', '-di', nan(200, 250), '-dt', [0], args{:}, '-m', 10, '-s', 5);
reference = ref.simulation;
data = nan(size(reference));
cells = randperm(numel(data), 300);
data(cells) = reference(cells);

[direct, ~] = g2s('-a', 'kriging', '-di', data, '-dt', [0], args{:}, '-method', 'direct', '-var');
[local, ~] = g2s('-a', 'kriging', '-di', data, '-dt', [0], args{:}, '-method', 'local', '-n', 32, '-os', 8, '-var');
fprintf('largest difference between direct and local: %g\n', max(abs(direct.estimate(:) - local.estimate(:))));

[r, c] = ind2sub(size(data), cells);
images = {reference, direct.estimate, direct.variance};
titles = {'reference', 'ordinary kriging (direct)', 'kriging variance'};
figure;
for k = 1:3
    subplot(1, 3, k);
    imagesc(images{k}); axis image; colorbar; hold on;
    plot(c, r, 'k.', 'MarkerSize', 4);
    title(titles{k});
end
