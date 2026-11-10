function legacy = g2sSchemaToLegacy(result)
if ~isstruct(result)
    legacy = result;
    return;
end

legacy = {};
if isfield(result, 'simulation')
    legacy{end + 1} = result.simulation;
end

artifactNames = {};
if isfield(result, 'artifacts') && isstruct(result.artifacts)
    artifactNames = fieldnames(result.artifacts);
    artifactNames = setdiff(artifactNames, {'log', 'warning', 'error', 'progress', 'meta', 'simulation'}, 'stable');
end
for i = 1:numel(artifactNames)
    name = artifactNames{i};
    if isfield(result, name)
        legacy{end + 1} = result.(name);
    end
end

if isfield(result, 'time')
    legacy{end + 1} = result.time;
end

reserved = [{'simulation', 'time', 'job_id', 'status', 'progress', 'artifacts', 'error', 'warnings'}, artifactNames(:)'];
metaFields = setdiff(fieldnames(result), reserved, 'stable');
if ~isempty(metaFields)
    meta = struct();
    for i = 1:numel(metaFields)
        meta.(metaFields{i}) = result.(metaFields{i});
    end
    legacy{end + 1} = meta;
end

if isfield(result, 'progress')
    legacy{end + 1} = result.progress;
end
if isfield(result, 'job_id')
    legacy{end + 1} = result.job_id;
end

if numel(legacy) == 1
    legacy = legacy{1};
end
