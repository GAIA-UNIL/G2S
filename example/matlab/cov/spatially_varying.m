% A covariance model whose parameters vary in space: the range grows from
% left to right and the anisotropy direction turns from top to bottom.
% Each node uses the model with the parameter values at that node (local
% stationarity).
ny = 200; nx = 300;
[columns, rows] = meshgrid(0:nx-1, 0:ny-1);
rangeMap = 10 + 40*columns/(nx - 1);   % practical range of the major axis, 10 to 50 cells
angleMap = -60 + 120*rows/(ny - 1);    % degrees, -60 at the top to +60 at the bottom

L = g2sCov.var('L', rangeMap);
model = 0.05*g2sCov.nugget() + 0.95*g2sCov.exponential('range', [L, L/4], 'angle', angleMap);
disp(char(model))  % 0.05*nugget + 0.95*exponential(range=[L, L/4], angle=_a1)

args = model.args();  % -cov text and -cvar L rangeMap -cvar _a1 angleMap
[result, ~] = g2s('-a', 'sgs', ...
                  '-di', nan(ny, nx), ...
                  '-dt', [0], ...
                  args{:}, ...
                  '-n', 40, ...
                  '-s', 42, ...
                  '-j', 0.5);
simulation = result.simulation;

% The covariance at one node, as covEval resolves it (pos = [column, row], 0-based)
disp(model.eval([0 0; 10 0], 'pos', [nx - 1, floor(ny/2)]))

figure;
subplot(1, 3, 1); imagesc(rangeMap); axis image; colorbar; title('range (cells)');
subplot(1, 3, 2); imagesc(angleMap); axis image; colorbar; title('angle (degrees)');
subplot(1, 3, 3); imagesc(simulation); axis image; colorbar; title('sgs');
