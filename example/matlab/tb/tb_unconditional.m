% Turning bands: a nested model with a nugget, a spherical and a Gaussian
% structure (each structure is simulated on its own lines).
model = 0.1*g2sCov.nugget() + 0.6*g2sCov.spherical('range', [80 30], 'angle', 30) + 0.3*g2sCov.gaussian('range', 15);
args = model.args();

[result, ~] = g2s('-a', 'tb', ...
                  '-di', nan(200, 300), ...
                  '-dt', [0], ...
                  args{:}, ...
                  '-nb', 1000, ...  % lines per structure
                  '-nr', 3, ...
                  '-m', 5, ...      % mean
                  '-s', 7, ...
                  '-j', 0.5);
simulation = result.simulation;     % 200 x 300 x 3

figure;
for r = 1:3
    subplot(1, 3, r);
    imagesc(simulation(:, :, r)); axis image off;
    title(sprintf('realisation %d', r - 1));
end
