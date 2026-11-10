ti = single(imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff'));

result = g2s( ...
    '-a', 'autoQS', ...
    '-ti', ti, ...
    '-dt', [0], ...
    '-n', 40, ...
    '-k', 1.2, ...
    '-j', 0.5);

disp(['job ', num2str(result.job_id)]);
if isfield(result, 'mean_error')
    disp(size(result.mean_error));
elseif isfield(result, 'simulation')
    disp(size(result.simulation));
end
if isfield(result, 'deviation_error')
    disp(size(result.deviation_error));
end
if isfield(result, 'sample_count')
    disp(size(result.sample_count));
end
