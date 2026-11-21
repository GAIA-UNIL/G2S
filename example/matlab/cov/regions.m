% Several covariance models, one per region, selected for each cell by an
% index map (-ii): region 0 is isotropic and smooth, region 1 anisotropic.
ny = 200; nx = 200;
[columns, rows] = meshgrid(0:nx-1, 0:ny-1);
index = double(rows > 80 + 30*sin(columns/25));  % 0 above the boundary, 1 below

smooth = g2sCov.gaussian('range', 30);
layered = 0.1*g2sCov.nugget() + 0.9*g2sCov.exponential('range', [60 6], 'angle', 10);
regions = g2sCov.regions({smooth, layered}, index);
args = regions.args();  % -cov model0 model1 -ii index

[result, ~] = g2s('-a', 'sgs', ...
                  '-di', nan(ny, nx), ...
                  '-dt', [0], ...
                  args{:}, ...
                  '-n', 40, ...
                  '-s', 7, ...
                  '-j', 0.5);

figure;
subplot(1, 2, 1); imagesc(index); axis image; title('model index (-ii)');
subplot(1, 2, 2); imagesc(result.simulation); axis image; title('sgs');
