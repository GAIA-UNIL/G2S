% Sequential Gaussian simulation: four unconditional realisations in one run.
model = 0.1*g2sCov.nugget() + 0.9*g2sCov.spherical('range', [60 20], 'angle', 30);
args = model.args();   % {'-cov', '0.1*nugget + 0.9*spherical(...)'}

[result, ~] = g2s('-a', 'sgs', ...
                  '-di', nan(200, 300), ...   % the grid: every cell is simulated
                  '-dt', [0], ...
                  args{:}, ...
                  '-n', 30, ...               % neighbours per node
                  '-nr', 4, ...               % realisations on the same path
                  '-s', 100, ...
                  '-j', 0.5);
simulation = result.simulation;   % 200 x 300 x 4
neighbours = result.indexmap;     % neighbours used at each node

figure;
for r = 1:4
    subplot(1, 4, r);
    imagesc(simulation(:, :, r));
    axis image off;
    title(sprintf('realisation %d', r - 1));
end
