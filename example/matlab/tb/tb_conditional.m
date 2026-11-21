% Turning bands in 3D, conditioned to vertical wells by kriging of the residuals.
rng(3);
nz = 40; ny = 100; nx = 100;
% 'range', [x y z]: component 0 is the last array index (x), component 2 the first (z)
model = g2sCov.exponential('range', [60 30 8], 'angle', [20 0 0]);
args = model.args();

data = nan(nz, ny, nx);
for well = 1:12  % 12 vertical wells
    data(:, randi(ny), randi(nx)) = randn(nz, 1);
end

[result, ~] = g2s('-a', 'tb', '-di', data, '-dt', [0], args{:}, ...
                  '-nr', 2, '-km', 'local', '-n', 40, '-s', 1, '-j', 0.5);
simulation = result.simulation;  % 40 x 100 x 100 x 2

figure;
subplot(1, 2, 1); imagesc(squeeze(simulation(nz/2, :, :, 1))); axis image; title('horizontal slice');
subplot(1, 2, 2); imagesc(squeeze(simulation(:, ny/2, :, 1))); title('vertical section');
