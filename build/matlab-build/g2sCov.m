classdef g2sCov
%G2SCOV Covariance models of the Gaussian geostatistics algorithms of G2S.
%   Builds the -cov text argument (and its -cvar bindings) of the Gaussian
%   algorithms (sgs, kriging, covEval, ...) and parses such a text back. It
%   never re-implements the covariance functions: model.eval(h) and
%   model.variogram(h) run the covEval algorithm on a G2S server, so every
%   model has a single implementation, in C++
%   (include/geostat/covarianceModel.hpp).
%
%   model = 0.1*g2sCov.nugget() + 0.9*g2sCov.spherical('range', [40 20], 'angle', thetaMap);
%   args  = model.args();
%   sim   = g2s('-a', 'sgs', '-di', di, args{:}, '-n', 50);
%   char(model)   % '0.1*nugget + 0.9*spherical(range=[40, 20], angle=_a1)', the canonical
%                 % text, identical to the C++ one (thetaMap got the automatic name _a1)
%
%   nug = g2sCov.var('nug', 0.1);
%   L   = g2sCov.var('L', rangeMap);              % matrix on the -di grid
%   model = nug*g2sCov.nugget() + (1 - nug)*g2sCov.spherical('range', [L, L/2], 'angle', 30);
%   % char(model): 'nug*nugget + (1 - nug)*spherical(range=[L, L/2], angle=30)'
%   % model.args(): {'-cov', <text>, '-cvar', 'nug', '0.1', '-cvar', 'L', rangeMap}
%
%   Structures (static methods): nugget(), spherical, exponential, gaussian,
%   cubic, matern ('nu'), stable ('alpha'), map(table). Combine them with +
%   and scale them with * (the factor is the partial sill; without a factor
%   the partial sill is 1). Structures cannot be subtracted.
%
%   RANGE ('range') is the practical range, in grid cells: spherical and
%   cubic reach 0 there; exponential, gaussian, stable and matern have
%   dropped to exp(-3) = 4.98% of their sill. 'scale' gives the raw scale
%   parameter instead (not both): exponential range = 3*scale, gaussian
%   range = sqrt(3)*scale, stable range = 3^(1/alpha)*scale, matern range =
%   u95(nu)*scale (u95(0.5) = 3), spherical and cubic range = scale. The
%   matern is the plain (h/scale)^nu K_nu(h/scale) form.
%
%   AXES. Lags, 'range', [r0 r1 r2] and 'angle' are given per axis in the
%   server's axis order: component 0 is the fastest-varying axis, which is
%   the LAST index of the MATLAB array (the columns of a 2D matrix, i.e. the
%   second index), component 1 the index before it (the rows, first index of
%   a 2D matrix), component 2 the first index of a 3D array. So for a
%   ny x nx matrix, 'range', [rx ry] is rx along A(i,:) and ry along A(:,j),
%   and a lag [dx dy] is a shift of dx columns and dy rows.
%   2D: 'angle', a (degrees) turns the direction of range(1) from axis 0
%   towards axis 1; with axis 0 to the right and axis 1 upwards a > 0 is
%   counter-clockwise, so on an image shown with row 1 at the top
%   (imagesc/imshow) it appears clockwise. 3D: 'angle', [a b c] ('angle', a
%   means [a 0 0]): the direction of range(1) is
%   (cos a cos b, sin a cos b, sin b): a is its azimuth in the (0,1) plane,
%   b its elevation towards axis 2, and c turns the two other principal axes
%   about it (axis 1 towards axis 2).
%
%   PARAMETERS. A parameter (partial sill, range, scale, angle, nu, alpha) is
%     - a number;
%     - a matrix: a spatially varying parameter, one value per cell of the
%       simulation grid (same size as -di); it becomes a variable with an
%       automatic name (_a1, _a2, ...);
%     - a named variable g2sCov.var('L', value), reusable in several places;
%     - an expression of those: + - * / (also .* ./), unary minus, min(a,b),
%       max(a,b), sqrt, exp, log, abs.
%   'range', 'scale' and 'angle' take one value per axis: a numeric vector
%   of at most 4 values ([40 20]), [L, L/2], or a cell {L, 20, map}. A
%   numeric array that is not such a short vector is ONE spatially varying
%   value; wrap a small map in g2sCov.var (or a cell) to avoid ambiguity.
%
%   PRECISION. Numbers are written with the shortest text that reads back to
%   the same double, in the model text and in the -cvar values, which are
%   passed as text (the MEX interface would round a number to single
%   precision and 6 decimals). Negative constants are passed as one
%   'name=value' token, because the interface reads a text starting with '-'
%   as a flag.
%
%   Methods: char, string, disp, args, eval, variogram, variables.
%   Static: nugget, spherical, exponential, gaussian, cubic, matern, stable,
%   map, var, parse, regions, formatNumber, isReservedWord,
%   isValidVariableName.
%   See also g2s, testG2sCov.

	properties (SetAccess = private)
		Kind = 'model'   % 'model', 'expr' (parameter expression), 'vector' ([a, b]) or 'set' (regions)
		Terms = {}       % model: cell of term structs
		Node = []        % expr: expression tree
		Items = {}       % vector: cell of expression trees
		Models = {}      % set: cell of models
		Index = []       % set: model index map (-ii)
	end

	methods
		function obj = g2sCov(kind, payload)
			% Internal constructor: use the static methods (g2sCov.spherical, g2sCov.var, g2sCov.parse, ...).
			if nargin == 0
				return
			end
			obj.Kind = kind;
			switch kind
				case 'model'
					obj.Terms = payload;
				case 'expr'
					obj.Node = payload;
				case 'vector'
					obj.Items = payload;
				case 'set'
					obj.Models = payload{1};
					obj.Index = payload{2};
				otherwise
					error('g2sCov:invalid', 'unknown g2sCov kind %s', kind);
			end
		end

		% ------------------------------------------------------------ algebra
		function r = plus(a, b)
			if isModel(a) || isModel(b)
				if isModel(a) && isModel(b)
					r = g2sCov('model', [a.Terms, b.Terms]);
				elseif isModel(a) && isZero(b)
					r = a;
				elseif isModel(b) && isZero(a)
					r = b;
				else
					fail('a number or parameter cannot be added to a covariance model; a constant term is a nugget: c*g2sCov.nugget()');
				end
				return
			end
			r = binaryExpr('add', a, b);
		end

		function r = minus(a, b)
			if isModel(a) || isModel(b)
				fail('a structure cannot be subtracted; parenthesise a difference in a partial sill: (a - b)*g2sCov.spherical(...)');
			end
			r = binaryExpr('sub', a, b);
		end

		function r = mtimes(a, b)
			if isModel(a) && isModel(b)
				fail('two covariance models cannot be multiplied (only a model and a partial sill)');
			elseif isModel(a)
				r = scaleModel(a, b, false);
			elseif isModel(b)
				r = scaleModel(b, a, false);
			else
				r = binaryExpr('mul', a, b);
			end
		end

		function r = times(a, b)
			r = mtimes(a, b);
		end

		function r = mrdivide(a, b)
			if isModel(b)
				fail('a covariance model cannot divide (only a partial sill can)');
			elseif isModel(a)
				r = scaleModel(a, b, true);
			else
				r = binaryExpr('div', a, b);
			end
		end

		function r = rdivide(a, b)
			r = mrdivide(a, b);
		end

		function r = uminus(a)
			if isModel(a)
				fail('the partial sill must be >= 0; a structure cannot be negated');
			end
			r = g2sCov('expr', mkOp('neg', toExpr(a, 'a parameter')));
		end

		function r = uplus(a)
			r = a;
		end

		function r = horzcat(varargin)
			% [L, L/2, 20]: a list of per-axis values for 'range', 'scale' or 'angle'.
			items = {};
			for k = 1:numel(varargin)
				x = varargin{k};
				if isa(x, 'g2sCov') && strcmp(x.Kind, 'vector')
					items = [items, x.Items]; %#ok<AGROW>
				elseif isa(x, 'g2sCov') && ~strcmp(x.Kind, 'expr')
					fail('covariance models cannot be concatenated; use a cell array {m1, m2} (e.g. for g2sCov.regions)');
				elseif isnumeric(x) && isempty(x)
					continue
				else
					items{end + 1} = toExpr(x, 'a vector element'); %#ok<AGROW>
				end
			end
			r = g2sCov('vector', items);
		end

		function r = vertcat(varargin)
			r = horzcat(varargin{:});
		end

		function r = min(a, b)
			r = binaryExpr('min', a, b);
		end

		function r = max(a, b)
			r = binaryExpr('max', a, b);
		end

		function r = sqrt(x)
			r = g2sCov('expr', mkOp('sqrt', toExpr(x, 'sqrt()')));
		end

		function r = exp(x)
			r = g2sCov('expr', mkOp('exp', toExpr(x, 'exp()')));
		end

		function r = log(x)
			r = g2sCov('expr', mkOp('log', toExpr(x, 'log()')));
		end

		function r = abs(x)
			r = g2sCov('expr', mkOp('abs', toExpr(x, 'abs()')));
		end

		% ------------------------------------------------------------ text
		function s = char(obj)
			% Canonical text (models: identical to what the C++ prints).
			switch obj.Kind
				case 'model'
					ctx = compileModels({obj});
					s = ctx.texts{1};
				case 'expr'
					s = printNode(obj.Node, emptyContext());
				case 'vector'
					parts = cellfun(@(n) printNode(n, emptyContext()), obj.Items, 'UniformOutput', false);
					s = ['[' strjoin(parts, ', ') ']'];
				case 'set'
					ctx = compileModels(obj.Models);
					s = strjoin(ctx.texts, sprintf('\n'));
			end
		end

		function s = string(obj)
			s = string(char(obj));
		end

		function disp(obj)
			fprintf('  g2sCov %s: %s\n', obj.Kind, char(obj));
		end

		function [names, values] = variables(obj)
			% Names and values of the model variables, in order of first appearance.
			ctx = compileModels(modelsOf(obj));
			names = ctx.bindNames;
			values = ctx.bindValues;
		end

		function a = args(obj)
			% Arguments for g2s(...): {'-cov', text, '-cvar', name, value, ...} (plus '-ii', index for regions).
			% Constants are passed as round-trip text (negative ones as one 'name=value' token), arrays as arrays.
			ctx = compileModels(modelsOf(obj));
			validateContext(ctx);
			a = [{'-cov'}, coverTexts(ctx.texts), cvarArgs(ctx)];
			if strcmp(obj.Kind, 'set')
				a = [a, {'-ii', single(obj.Index)}];
			end
		end

		function c = eval(obj, h, varargin)
			%EVAL Covariance C(h), computed by the covEval algorithm on a running G2S server.
			%   c = model.eval(h)                       stationary model
			%   c = model.eval(h, 'pos', [12 30])       spatially varying model, resolved at a node
			%   c = model.eval(h, '-sa', 'host', '-p', 8128, '-TO', 5000)   g2s options are passed through
			%   This needs the g2s MEX file and a G2S server ('g2s server', or ./g2s_server in
			%   build/c++-build) reachable with the usual -sa / -p options.
			%   h: lags in grid cells. A scalar or a vector is a list of 1D lags; an N x ndim
			%      matrix (N >= 2) is one lag per row, column 1 being the shift along the LAST
			%      array index (the columns of a matrix), column 2 along the index before it
			%      (the rows). For a single multi-dimensional lag give 'ndim', e.g.
			%      model.eval([10 0], 'ndim', 2).
			%   'pos': node where the parameters are resolved, integer 0-based cell coordinates
			%      in the same component order (column index first): 1 x ndim for every lag or
			%      N x ndim. 'verbose', true shows the job progress.
			%   Returns an N x 1 vector.
			c = covEvalCall(obj, h, false, varargin);
		end

		function g = variogram(obj, h, varargin)
			%VARIOGRAM gamma(h) = C(0) - C(h), computed by covEval on a G2S server (see eval).
			g = covEvalCall(obj, h, true, varargin);
		end
	end

	methods (Static)
		function m = nugget()
			% Nugget: the partial sill at lag exactly 0, 0 elsewhere.
			m = g2sCov('model', {mkTerm(mkConst(1), 'nugget', {}, false, {}, [], [])});
		end

		function m = spherical(varargin)
			% Spherical: rho = 1 - 1.5 t + 0.5 t^3 for t < 1, 0 beyond (t = lag/range); scale = range.
			% g2sCov.spherical('range', r, 'angle', a) or 'scale', s; see the class help for axes and angles.
			m = buildStructure('spherical', varargin);
		end

		function m = exponential(varargin)
			% Exponential: rho = exp(-3 t) = exp(-h/scale); practical range = 3*scale.
			m = buildStructure('exponential', varargin);
		end

		function m = gaussian(varargin)
			% Gaussian: rho = exp(-3 t^2) = exp(-(h/scale)^2); practical range = sqrt(3)*scale.
			m = buildStructure('gaussian', varargin);
		end

		function m = cubic(varargin)
			% Cubic (GSLIB): rho = 1 - 7t^2 + 35/4 t^3 - 7/2 t^5 + 3/4 t^7 for t < 1, 0 beyond; scale = range.
			m = buildStructure('cubic', varargin);
		end

		function m = matern(varargin)
			% Matern, 'nu' in (0, 100]: rho = 2^(1-nu)/Gamma(nu) (h/scale)^nu K_nu(h/scale);
			% practical range = u95(nu)*scale (u95(0.5) = 3, u95(1.5) = 4.7490, u95(2.5) = 5.9245).
			m = buildStructure('matern', varargin);
		end

		function m = stable(varargin)
			% Stable, 'alpha' in (0, 2]: rho = exp(-3 t^alpha) = exp(-(h/scale)^alpha); range = 3^(1/alpha)*scale.
			m = buildStructure('stable', varargin);
		end

		function m = map(table)
			% Covariance given as a map M: C(h) = sill*M(centre + round(h)), 0 outside M.
			% table: a matrix (same size as the grid; centre = floor(size/2) per axis, 0-based,
			% must be > 0), a g2sCov.var(name, M), or a name (char) to bind later.
			if ischar(table) || (isstring(table) && isscalar(table))
				node = mkVar(char(table), [], 0);
			elseif isa(table, 'g2sCov') && strcmp(table.Kind, 'expr') && strcmp(table.Node.op, 'var')
				node = table.Node;
			elseif isnumeric(table) && ~isscalar(table) && ~isempty(table)
				node = mkVar('', double(table), newId());
			else
				fail('map takes one table name: map(name), a g2sCov.var or a matrix');
			end
			if isnumeric(node.data) && isscalar(node.data)
				fail(sprintf('map(%s) needs a covariance map image, not a number', node.name));
			end
			m = g2sCov('model', {mkTerm(mkConst(1), 'map', {}, false, {}, [], node)});
		end

		function v = var(name, value)
			% A named variable bound with -cvar: g2sCov.var('L', 40), g2sCov.var('theta', angleMap).
			% value: a number (constant), a matrix on the simulation grid (spatially varying, or the
			% table of map), a char (name of data already on the server) or [] (bind later).
			if nargin < 2
				value = [];
			end
			if isstring(name)
				name = char(name);
			end
			checkVariableName(name);
			v = g2sCov('expr', mkVar(name, checkValue(name, value), 0));
		end

		function m = parse(text, varargin)
			% Parses a model text: g2sCov.parse('nug*nugget + (1-nug)*exponential(range=L)', 'nug', 0.1, 'L', map).
			% Errors mirror the C++ messages (with the column).
			if isstring(text)
				text = char(text);
			end
			if ~ischar(text)
				fail('a covariance model text must be a char vector or a string');
			end
			if isempty(text)
				text = '';
			end
			context = sprintf('covariance model "%s": ', text);
			try
				m = parseModel(text);
			catch err
				if strcmp(err.identifier, 'g2sCov:invalid') && ~startsWith(err.message, context)
					error('g2sCov:invalid', '%s', [context err.message]);
				end
				rethrow(err);
			end
			if mod(numel(varargin), 2) ~= 0
				fail('g2sCov.parse(text, name, value, ...) expects name/value pairs');
			end
			for k = 1:2:numel(varargin)
				m = bindVariable(m, char(varargin{k}), varargin{k + 1});
			end
		end

		function s = regions(models, index)
			% Several models for discrete regions: index (same size as -di) selects models{index+1}.
			% s = g2sCov.regions({m0, m1}, index); args = s.args() gives '-cov', t0, t1, '-cvar', ..., '-ii', index.
			if ~iscell(models) || isempty(models)
				fail('g2sCov.regions takes a non-empty cell array of models');
			end
			for k = 1:numel(models)
				if ~isModel(models{k})
					fail('g2sCov.regions takes a cell array of covariance models');
				end
			end
			if ~isnumeric(index) || isempty(index) || any(~isfinite(index(:))) || any(index(:) ~= floor(index(:))) || any(index(:) < 0)
				fail('the model index must hold model indices (0, 1, ...)');
			end
			if max(index(:)) >= numel(models)
				fail(sprintf('model index %d but only %d model(s) given (the index is 0-based)', max(index(:)), numel(models)));
			end
			s = g2sCov('set', {models(:).', double(index)});
		end

		function s = formatNumber(v)
			% Shortest text of a double, identical to the C++ formatNumber.
			s = fmt(v);
		end

		function tf = isReservedWord(name)
			% Structure names, parameter keywords, function names and 'h' (any case).
			tf = reserved(char(name));
		end

		function tf = isValidVariableName(name)
			% [A-Za-z_][A-Za-z0-9_]* and not a reserved word.
			tf = validName(char(name));
		end
	end
end

% ===================================================================== local functions

function fail(message)
error('g2sCov:invalid', '%s', message);
end

function tf = isModel(x)
tf = isa(x, 'g2sCov') && strcmp(x.Kind, 'model');
end

function tf = isZero(x)
tf = isnumeric(x) && isscalar(x) && x == 0;
end

function models = modelsOf(obj)
switch obj.Kind
	case 'model'
		models = {obj};
	case 'set'
		models = obj.Models;
	otherwise
		fail('args/eval/variables apply to a covariance model, not to a parameter expression');
end
end

% --------------------------------------------------------------------- names and numbers

function tf = reserved(name)
lower_ = lower(name);
tf = any(strcmp(lower_, {'nugget', 'spherical', 'exponential', 'gaussian', 'cubic', 'matern', 'stable', 'map', ...
	'range', 'scale', 'angle', 'nu', 'alpha', 'h', 'min', 'max', 'sqrt', 'exp', 'log', 'abs'}));
end

function tf = validName(name)
tf = ischar(name) && ~isempty(regexp(name, '^[A-Za-z_][A-Za-z0-9_]*$', 'once')) && ~reserved(name);
end

function checkVariableName(name)
if ~ischar(name)
	fail('a variable name must be a char vector or a string');
end
if validName(name)
	return
end
if reserved(name)
	fail(sprintf('''%s'' is a reserved word (structure, keyword or function name) and cannot be a variable', name));
end
fail(sprintf('''%s'' is not a valid variable name (letters, digits and ''_'', not starting with a digit)', name));
end

function value = checkValue(name, value)
if isempty(value) && isnumeric(value)
	value = [];
	return
end
if ischar(value) || (isstring(value) && isscalar(value))
	value = char(value);
	return
end
if isa(value, 'g2sCov') || ~(isnumeric(value) || islogical(value)) || ~isreal(value)
	fail(sprintf('variable ''%s'': the value must be a number or a numeric array', name));
end
value = double(value);
if isscalar(value) && ~isfinite(value)
	fail(sprintf('variable ''%s'': the value is not finite', name));
end
end

function s = fmt(v)
v = double(v);
if v == 0
	s = '0';
elseif isnan(v)
	s = 'nan';
elseif isinf(v)
	if v > 0, s = 'inf'; else, s = '-inf'; end
elseif abs(v) < 1e15 && v == floor(v)
	s = sprintf('%.0f', v);
else
	for p = 1:17
		s = sprintf(sprintf('%%.%dg', p), v);
		if str2double(s) == v
			break
		end
	end
end
end

function id = newId()
persistent counter
if isempty(counter)
	counter = 0;
end
counter = counter + 1;
id = counter;
end

% --------------------------------------------------------------------- expression trees

function n = mkNode(op)
n = struct('op', op, 'value', 0, 'name', '', 'id', 0, 'data', [], 'left', [], 'right', []);
end

function n = mkConst(v)
n = mkNode('const');
n.value = double(v);
end

function n = mkVar(name, data, id)
n = mkNode('var');
n.name = name;
n.data = data;
n.id = id;
end

function tf = isUnaryOp(op)
tf = any(strcmp(op, {'neg', 'sqrt', 'exp', 'log', 'abs'}));
end

function n = mkOp(op, left, right)
% Adds an operation, folding it into a constant when its operands are constant.
if nargin < 3
	right = [];
end
if isUnaryOp(op)
	if strcmp(left.op, 'const')
		n = foldOp(op, left.value, []);
		return
	end
elseif strcmp(left.op, 'const') && strcmp(right.op, 'const')
	n = foldOp(op, left.value, right.value);
	return
end
n = mkNode(op);
n.left = left;
n.right = right;
end

function n = foldOp(op, a, b)
if isempty(b)
	value = applyOp(op, a, 0);
else
	value = applyOp(op, a, b);
end
if ~isfinite(value)
	fail(sprintf('constant sub-expression %s is not finite', opText(op, a, b)));
end
n = mkConst(value);
end

function v = applyOp(op, a, b)
% IEEE arithmetic like the C++ (never complex).
switch op
	case 'neg', v = -a;
	case 'add', v = a + b;
	case 'sub', v = a - b;
	case 'mul', v = a * b;
	case 'div', v = a / b;
	case 'min'
		if b < a, v = b; else, v = a; end
	case 'max'
		if a < b, v = b; else, v = a; end
	case 'sqrt'
		if a < 0, v = NaN; else, v = sqrt(a); end
	case 'exp', v = exp(a);
	case 'log'
		if a < 0, v = NaN; else, v = log(a); end
	case 'abs', v = abs(a);
end
end

function s = opText(op, a, b)
fa = fmt(a);
fb = '';
if ~isempty(b), fb = fmt(b); end
switch op
	case 'neg', s = ['-' fa];
	case 'add', s = [fa ' + ' fb];
	case 'sub', s = [fa ' - ' fb];
	case 'mul', s = [fa '*' fb];
	case 'div', s = [fa '/' fb];
	case {'min', 'max'}, s = [op '(' fa ', ' fb ')'];
	otherwise, s = [op '(' fa ')'];
end
end

function n = toExpr(x, what)
% Number -> constant; matrix -> automatic spatially varying variable; g2sCov expression -> its tree.
if isa(x, 'g2sCov')
	switch x.Kind
		case 'expr'
			n = x.Node;
		case 'vector'
			fail(sprintf('%s takes a single value, got a vector', what));
		otherwise
			fail(sprintf('%s cannot be a covariance model', what));
	end
elseif (isnumeric(x) || islogical(x)) && isreal(x) && ~isempty(x)
	if islogical(x)
		fail(sprintf('%s must be numeric, not logical', what));
	end
	if isscalar(x)
		if ~isfinite(x)
			fail(sprintf('a parameter is not finite (%s)', fmt(x)));
		end
		n = mkConst(double(x));
	else
		n = mkVar('', double(x), newId());
	end
elseif ischar(x) || isstring(x)
	fail(sprintf('%s: got the text ''%s''; use a number, a matrix, g2sCov.var(name, value) or g2sCov.parse(text)', what, char(x)));
else
	fail(sprintf('%s must be a number, a matrix, a g2sCov.var or an expression, not %s', what, class(x)));
end
end

function items = toVector(x, what, maxCount)
% Per-axis values: cell, [a, b] (g2sCov vector), short numeric vector, or one value.
if iscell(x)
	if isempty(x)
		fail(['empty vector for ' what]);
	end
	items = cellfun(@(e) toExpr(e, what), x(:).', 'UniformOutput', false);
elseif isa(x, 'g2sCov') && strcmp(x.Kind, 'vector')
	if isempty(x.Items)
		fail(['empty vector for ' what]);
	end
	items = x.Items;
elseif isnumeric(x) && isempty(x)
	fail(['empty vector for ' what]);
elseif isnumeric(x) && isvector(x) && numel(x) <= maxCount
	items = arrayfun(@(e) toExpr(e, what), x(:).', 'UniformOutput', false);
else
	items = {toExpr(x, what)};
end
end

function r = binaryExpr(op, a, b)
r = g2sCov('expr', mkOp(op, toExpr(a, 'a parameter'), toExpr(b, 'a parameter')));
end

function p = precedence(n)
switch n.op
	case {'add', 'sub'}, p = 1;
	case {'mul', 'div'}, p = 2;
	case 'neg', p = 3;
	case 'const'
		if n.value < 0, p = 3; else, p = 4; end
	otherwise, p = 4;
end
end

function ctx = emptyContext()
ctx = struct('autoIds', [], 'autoNames', {{}});
end

function s = varName(n, ctx)
if ~isempty(n.name)
	s = n.name;
	return
end
k = find(ctx.autoIds == n.id, 1);
if isempty(k)
	s = '?';
else
	s = ctx.autoNames{k};
end
end

function s = printNode(n, ctx)
switch n.op
	case 'const'
		s = fmt(n.value);
	case 'var'
		s = varName(n, ctx);
	case 'neg'
		inner = printNode(n.left, ctx);
		if precedence(n.left) < 3
			inner = ['(' inner ')'];
		end
		s = ['-' inner];
	case {'add', 'sub', 'mul', 'div'}
		own = precedence(n);
		left = printNode(n.left, ctx);
		right = printNode(n.right, ctx);
		if precedence(n.left) < own
			left = ['(' left ')'];
		end
		if precedence(n.right) <= own
			right = ['(' right ')'];
		end
		symbols = struct('add', ' + ', 'sub', ' - ', 'mul', '*', 'div', '/');
		s = [left symbols.(n.op) right];
	case {'min', 'max'}
		s = [n.op '(' printNode(n.left, ctx) ', ' printNode(n.right, ctx) ')'];
	otherwise
		s = [n.op '(' printNode(n.left, ctx) ')'];
end
end

function leaves = collectVars(n, leaves)
if isempty(n)
	return
end
if strcmp(n.op, 'var')
	leaves{end + 1} = n;
	return
end
leaves = collectVars(n.left, leaves);
leaves = collectVars(n.right, leaves);
end

function v = evalNode(n, names, values)
% Value of an expression when every variable it uses has a scalar value; [] otherwise.
switch n.op
	case 'const'
		v = n.value;
	case 'var'
		k = find(strcmp(names, n.name), 1);
		if ~isempty(n.name) && ~isempty(k)
			v = values(k);
		elseif isnumeric(n.data) && isscalar(n.data)
			v = n.data;
		else
			v = [];
		end
	otherwise
		a = evalNode(n.left, names, values);
		if isempty(a)
			v = [];
			return
		end
		b = 0;
		if ~isempty(n.right)
			b = evalNode(n.right, names, values);
			if isempty(b)
				v = [];
				return
			end
		end
		v = applyOp(n.op, a, b);
end
end

function n = setVarData(n, name, value)
if isempty(n)
	return
end
if strcmp(n.op, 'var')
	if strcmp(n.name, name)
		n.data = value;
	end
	return
end
n.left = setVarData(n.left, name, value);
n.right = setVarData(n.right, name, value);
end

% --------------------------------------------------------------------- structures and models

function t = mkTerm(sill, type, range, useScale, angle, shape, table)
t = struct();
t.sill = sill;
t.type = type;
t.range = range;
t.useScale = useScale;
t.angle = angle;
t.shape = shape;
t.table = table;
end

function problem = checkValues(type, sill, range, angle, shape)
% Mirror of ModelSpec::checkValues (C++); unknown values are [] and skipped.
problem = '';
if ~isempty(sill)
	if ~isfinite(sill)
		problem = sprintf('the partial sill is not finite (%s)', fmt(sill)); return
	end
	if sill < 0
		problem = ['the partial sill must be >= 0, got ' fmt(sill)]; return
	end
end
for k = 1:numel(range)
	if ~isfinite(range(k)) || ~(range(k) > 0)
		problem = ['range/scale must be > 0, got ' fmt(range(k))]; return
	end
end
for k = 1:numel(angle)
	if ~isfinite(angle(k))
		problem = sprintf('angle is not finite (%s)', fmt(angle(k))); return
	end
end
if ~isempty(shape)
	if strcmp(type, 'matern') && ~(shape > 0 && shape <= 100)
		problem = ['nu must be in (0, 100], got ' fmt(shape)]; return
	end
	if strcmp(type, 'stable') && ~(shape > 0 && shape <= 2)
		problem = ['alpha must be in (0, 2], got ' fmt(shape)]; return
	end
end
end

function values = constantsOf(nodes)
values = [];
for k = 1:numel(nodes)
	if strcmp(nodes{k}.op, 'const')
		values(end + 1) = nodes{k}.value; %#ok<AGROW>
	end
end
end

function m = buildStructure(type, options)
if mod(numel(options), 2) ~= 0
	fail(sprintf('g2sCov.%s expects name/value pairs, e.g. g2sCov.%s(''range'', 10)', type, type));
end
allowed = {'range', 'scale', 'angle'};
shapeKey = '';
if strcmp(type, 'matern'), shapeKey = 'nu'; end
if strcmp(type, 'stable'), shapeKey = 'alpha'; end
if ~isempty(shapeKey), allowed{end + 1} = shapeKey; end
given = struct();
for k = 1:2:numel(options)
	key = options{k};
	if ~(ischar(key) || (isstring(key) && isscalar(key)))
		fail(sprintf('g2sCov.%s expects name/value pairs', type));
	end
	key = lower(char(key));
	if ~any(strcmp(key, allowed))
		fail(sprintf('unknown keyword ''%s'' for %s (allowed: %s)', char(options{k}), type, strjoin(allowed, ', ')));
	end
	if isfield(given, key) || (any(strcmp(key, {'range', 'scale'})) && (isfield(given, 'range') || isfield(given, 'scale')))
		if any(strcmp(key, {'range', 'scale'}))
			fail(['range/scale given twice in ' type]);
		end
		fail(sprintf('''%s'' given twice in %s', key, type));
	end
	given.(key) = options{k + 1};
end
if ~isfield(given, 'range') && ~isfield(given, 'scale')
	fail([type ' needs range= or scale=']);
end
useScale = isfield(given, 'scale');
if useScale, keyword = 'scale'; else, keyword = 'range'; end
range = toVector(given.(keyword), [keyword ' of ' type], 4);
if numel(range) > 4
	fail(sprintf('%s has more than 4 values', keyword));
end
angle = {};
if isfield(given, 'angle')
	angle = toVector(given.angle, ['angle of ' type], 3);
	if numel(angle) ~= 1 && numel(angle) ~= 3
		fail('angle takes one value (2D) or three values (3D)');
	end
end
shape = [];
shapeValue = [];
if ~isempty(shapeKey)
	if ~isfield(given, shapeKey)
		fail(sprintf('%s needs %s=', type, shapeKey));
	end
	value = given.(shapeKey);
	% like range: a short numeric vector is a list of values (an error here), a larger array a map
	if iscell(value) || (isa(value, 'g2sCov') && strcmp(value.Kind, 'vector')) || ...
			(isnumeric(value) && isvector(value) && numel(value) > 1 && numel(value) <= 4)
		fail([shapeKey ' takes a single value']);
	end
	shape = toExpr(value, shapeKey);
	if strcmp(shape.op, 'const')
		shapeValue = shape.value;
	end
end
problem = checkValues(type, [], constantsOf(range), constantsOf(angle), shapeValue);
if ~isempty(problem)
	fail([type ': ' problem]);
end
m = g2sCov('model', {mkTerm(mkConst(1), type, range, useScale, angle, shape, [])});
end

function r = scaleModel(model, factor, divide)
if isa(factor, 'g2sCov') && ~strcmp(factor.Kind, 'expr')
	fail('two covariance models cannot be multiplied (only a model and a partial sill)');
end
f = toExpr(factor, 'the partial sill');
terms = model.Terms;
for k = 1:numel(terms)
	sill = terms{k}.sill;
	if divide
		newSill = mkOp('div', sill, f);
	elseif strcmp(sill.op, 'const') && sill.value == 1
		newSill = f;
	elseif strcmp(f.op, 'const') && f.value == 1
		newSill = sill;
	else
		newSill = mkOp('mul', f, sill);
	end
	if strcmp(newSill.op, 'const')
		problem = checkValues(terms{k}.type, newSill.value, [], [], []);
		if ~isempty(problem)
			fail([terms{k}.type ': ' problem]);
		end
	end
	terms{k}.sill = newSill;
end
r = g2sCov('model', terms);
end

function m = bindVariable(m, name, value)
value = checkValue(name, value);
terms = m.Terms;
found = false;
for k = 1:numel(terms)
	t = terms{k};
	if ~isempty(t.table) && strcmp(t.table.name, name)
		if isnumeric(value) && isscalar(value)
			fail(sprintf('map(%s) needs a covariance map image, not a number', name));
		end
		t.table.data = value;
		found = true;
	end
	leaves = collectAll(t);
	for j = 1:numel(leaves)
		if strcmp(leaves{j}.name, name)
			found = true;
		end
	end
	t.sill = setVarData(t.sill, name, value);
	for j = 1:numel(t.range), t.range{j} = setVarData(t.range{j}, name, value); end
	for j = 1:numel(t.angle), t.angle{j} = setVarData(t.angle{j}, name, value); end
	t.shape = setVarData(t.shape, name, value);
	terms{k} = t;
end
if ~found
	fail(sprintf('variable ''%s'' is bound but the model does not use it', name));
end
m = g2sCov('model', terms);
end

function leaves = collectAll(t)
% Variable leaves of a term, in canonical order (sill, range, angle, shape).
leaves = collectVars(t.sill, {});
for j = 1:numel(t.range), leaves = collectVars(t.range{j}, leaves); end
for j = 1:numel(t.angle), leaves = collectVars(t.angle{j}, leaves); end
leaves = collectVars(t.shape, leaves);
end

% --------------------------------------------------------------------- compilation (names, texts, args)

function ctx = compileModels(models)
named = {};
namedValues = {};
namedIsMap = [];
autoIds = [];
autoValues = {};
autoIsMap = [];
order = zeros(0, 2); % [kind (1 named, 2 auto), index]
for m = 1:numel(models)
	terms = models{m}.Terms;
	for k = 1:numel(terms)
		t = terms{k};
		leaves = collectAll(t);
		isMapLeaf = false(1, numel(leaves));
		if ~isempty(t.table)
			leaves{end + 1} = t.table; %#ok<AGROW>
			isMapLeaf(end + 1) = true; %#ok<AGROW>
		end
		for j = 1:numel(leaves)
			leaf = leaves{j};
			isMap = isMapLeaf(j);
			if isempty(leaf.name)
				a = find(autoIds == leaf.id, 1);
				if isempty(a)
					autoIds(end + 1) = leaf.id; %#ok<AGROW>
					autoValues{end + 1} = leaf.data; %#ok<AGROW>
					autoIsMap(end + 1) = isMap; %#ok<AGROW>
					order(end + 1, :) = [2, numel(autoIds)]; %#ok<AGROW>
				elseif autoIsMap(a) ~= isMap
					fail('the same array is used both as a map table and as a parameter');
				end
				continue
			end
			a = find(strcmp(named, leaf.name), 1);
			if isempty(a)
				named{end + 1} = leaf.name; %#ok<AGROW>
				namedValues{end + 1} = leaf.data; %#ok<AGROW>
				namedIsMap(end + 1) = isMap; %#ok<AGROW>
				order(end + 1, :) = [1, numel(named)]; %#ok<AGROW>
			else
				if namedIsMap(a) ~= isMap
					fail(sprintf('''%s'' is used both as a map table and as a parameter variable', leaf.name));
				end
				if isempty(namedValues{a})
					namedValues{a} = leaf.data; % an unbound occurrence takes the bound value
				elseif ~isempty(leaf.data) && ~isequal(namedValues{a}, leaf.data)
					fail(sprintf('variable ''%s'' is defined twice with different values', leaf.name));
				end
			end
		end
	end
end
% automatic names avoid every user name
autoNames = cell(1, numel(autoIds));
counter = 1;
for a = 1:numel(autoIds)
	while any(strcmp(named, sprintf('_a%d', counter)))
		counter = counter + 1;
	end
	autoNames{a} = sprintf('_a%d', counter);
	counter = counter + 1;
end
ctx = struct('autoIds', autoIds, 'autoNames', {autoNames});
ctx.bindNames = cell(1, size(order, 1));
ctx.bindValues = cell(1, size(order, 1));
for k = 1:size(order, 1)
	if order(k, 1) == 1
		ctx.bindNames{k} = named{order(k, 2)};
		ctx.bindValues{k} = namedValues{order(k, 2)};
	else
		ctx.bindNames{k} = autoNames{order(k, 2)};
		ctx.bindValues{k} = autoValues{order(k, 2)};
	end
end
ctx.models = models;
ctx.texts = cell(1, numel(models));
for m = 1:numel(models)
	ctx.texts{m} = modelText(models{m}, ctx);
end
end

function s = vectorText(nodes, ctx)
if numel(nodes) == 1
	s = printNode(nodes{1}, ctx);
	return
end
parts = cellfun(@(n) printNode(n, ctx), nodes, 'UniformOutput', false);
s = ['[' strjoin(parts, ', ') ']'];
end

function text = modelText(model, ctx)
parts = cell(1, numel(model.Terms));
for k = 1:numel(model.Terms)
	t = model.Terms{k};
	sill = printNode(t.sill, ctx);
	if any(strcmp(t.sill.op, {'add', 'sub'}))
		sill = ['(' sill ')'];
	end
	switch t.type
		case 'nugget'
			parts{k} = [sill '*nugget'];
		case 'map'
			parts{k} = [sill '*map(' varName(t.table, ctx) ')'];
		otherwise
			if t.useScale, key = 'scale'; else, key = 'range'; end
			s = [sill '*' t.type '(' key '=' vectorText(t.range, ctx)];
			if ~isempty(t.angle)
				s = [s ', angle=' vectorText(t.angle, ctx)]; %#ok<AGROW>
			end
			if strcmp(t.type, 'matern')
				s = [s ', nu=' printNode(t.shape, ctx)]; %#ok<AGROW>
			elseif strcmp(t.type, 'stable')
				s = [s ', alpha=' printNode(t.shape, ctx)]; %#ok<AGROW>
			end
			parts{k} = [s ')'];
	end
end
text = strjoin(parts, ' + ');
end

function validateContext(ctx)
% Client-side checks of what is cheap: unbound names and constant (scalar) values.
scalarNames = {};
scalarValues = [];
for k = 1:numel(ctx.bindNames)
	value = ctx.bindValues{k};
	if isempty(value)
		fail(sprintf('variable ''%s'' has no value; use g2sCov.var(''%s'', value) or g2sCov.parse(text, ''%s'', value)', ...
			ctx.bindNames{k}, ctx.bindNames{k}, ctx.bindNames{k}));
	end
	if isnumeric(value) && isscalar(value)
		scalarNames{end + 1} = ctx.bindNames{k}; %#ok<AGROW>
		scalarValues(end + 1) = value; %#ok<AGROW>
	end
end
for m = 1:numel(ctx.models)
	terms = ctx.models{m}.Terms;
	for k = 1:numel(terms)
		t = terms{k};
		if strcmp(t.type, 'map')
			if any(strcmp(scalarNames, varName(t.table, ctx)))
				fail(sprintf('map(%s) needs a covariance map image, not a number', varName(t.table, ctx)));
			end
			continue
		end
		range = knownValues(t.range, scalarNames, scalarValues);
		angle = knownValues(t.angle, scalarNames, scalarValues);
		shape = [];
		if ~isempty(t.shape)
			shape = evalNode(t.shape, scalarNames, scalarValues);
		end
		problem = checkValues(t.type, evalNode(t.sill, scalarNames, scalarValues), range, angle, shape);
		if ~isempty(problem)
			fail([t.type ': ' problem]);
		end
	end
end
end

function values = knownValues(nodes, names, scalars)
values = [];
for k = 1:numel(nodes)
	v = evalNode(nodes{k}, names, scalars);
	if ~isempty(v)
		values(end + 1) = v; %#ok<AGROW>
	end
end
end

function texts = coverTexts(texts)
% A text starting with '-' would be read as a flag by the interface; the C++ ignores a leading space.
for k = 1:numel(texts)
	if startsWith(texts{k}, '-')
		texts{k} = [' ' texts{k}];
	end
end
end

function a = cvarArgs(ctx)
a = {};
for k = 1:numel(ctx.bindNames)
	name = ctx.bindNames{k};
	value = ctx.bindValues{k};
	if isnumeric(value) && isscalar(value)
		text = fmt(value);
		if startsWith(text, '-')
			a = [a, {'-cvar', [name '=' text]}]; %#ok<AGROW>  '-30' alone would be read as a flag
		else
			a = [a, {'-cvar', name, text}]; %#ok<AGROW>
		end
	else
		a = [a, {'-cvar', name, value}]; %#ok<AGROW>
	end
end
end

% --------------------------------------------------------------------- evaluation through covEval

function out = covEvalCall(obj, h, variogram, options)
pos = [];
ndim = [];
verbose = false;
extra = {};
k = 1;
while k <= numel(options)
	key = options{k};
	if (ischar(key) || (isstring(key) && isscalar(key))) && any(strcmpi(char(key), {'pos', 'ndim', 'verbose'})) && k < numel(options)
		switch lower(char(key))
			case 'pos', pos = options{k + 1};
			case 'ndim', ndim = options{k + 1};
			case 'verbose', verbose = logical(options{k + 1});
		end
		k = k + 2;
	else
		extra{end + 1} = key; %#ok<AGROW>
		k = k + 1;
	end
end
if strcmp(obj.Kind, 'set') && isempty(pos)
	fail('a set of regional models needs ''pos'' to say where to evaluate it');
end
a = args(obj);
h = double(h);
if ~isempty(ndim)
	h = reshape(h.', ndim, []).';
elseif isvector(h)
	h = h(:);
end
if ndims(h) > 2 || isempty(h)
	fail('h must be a scalar, a vector of 1D lags or an N x ndim matrix');
end
nd = size(h, 2);
count = size(h, 1);
call = [{'-a', 'covEval'}, a, {'-lag', h}];
if ~isempty(pos)
	p = double(pos);
	if nd == 1
		p = p(:);
	elseif isvector(p) && numel(p) == nd
		p = p(:).';
	end
	if size(p, 2) ~= nd || ~(size(p, 1) == 1 || size(p, 1) == count)
		fail(sprintf('pos must have %d component(s): one node for every lag, or one per lag', nd));
	end
	call = [call, {'-pos', p}];
end
if variogram
	call{end + 1} = '-vario';
	key = 'variogram';
else
	key = 'covariance';
end
if ~verbose
	call{end + 1} = '-silent';
end
call = [call, {'-returnFormat', 'schema'}, extra];
if ~any(exist('g2s', 'file') == [2 3])
	error('g2sCov:server', ['model.eval/variogram evaluate the covariance with the covEval algorithm of a G2S server, ' ...
		'but the g2s MEX file is not on the MATLAB path (run CompileG2S in build/matlab-build)']);
end
try
	result = g2s(call{:});
catch err
	if contains(err.message, 'off-line') || contains(lower(err.message), 'timeout')
		error('g2sCov:server', ['model.eval/variogram evaluate the covariance with the covEval algorithm of a running ' ...
			'G2S server, and none answered (start one with ''g2s server'', or pass ''-sa'', host, ''-p'', port): %s'], err.message);
	end
	error('g2sCov:server', 'covEval (on the G2S server) failed: %s', err.message);
end
if isstruct(result) && isfield(result, key)
	out = double(result.(key));
elseif isnumeric(result) && ~variogram
	out = double(result);
else
	error('g2sCov:server', 'covEval returned no %s', key);
end
out = reshape(out, count, 1);
end

% --------------------------------------------------------------------- parser (port of ModelSpec::parse)

function tf = isDigit(c)
tf = c >= '0' && c <= '9';
end

function tf = isAlpha(c)
tf = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
end

function t = mkToken(kind, text, value, position)
t = struct('kind', kind, 'text', text, 'value', value, 'position', position);
end

function tokens = tokenize(text)
% Tokens with 0-based character positions (column = position + 1), ending with an 'end' token.
tokens = repmat(mkToken('end', '', 0, 0), 1, 0);
i = 1;
n = numel(text);
whitespace = sprintf(' \t\n\v\f\r');
while i <= n
	c = text(i);
	if any(c == whitespace)
		i = i + 1;
		continue
	end
	if isDigit(c) || (c == '.' && i + 1 <= n && isDigit(text(i + 1)))
		j = i;
		while j <= n && isDigit(text(j)), j = j + 1; end
		if j <= n && text(j) == '.'
			j = j + 1;
			while j <= n && isDigit(text(j)), j = j + 1; end
		end
		if j <= n && (text(j) == 'e' || text(j) == 'E')
			k = j + 1;
			if k <= n && (text(k) == '+' || text(k) == '-'), k = k + 1; end
			if k > n || ~isDigit(text(k))
				fail(sprintf('invalid number ''%s'' at column %d', text(i:k - 1), i));
			end
			while k <= n && isDigit(text(k)), k = k + 1; end
			j = k;
		end
		if j <= n && (isAlpha(text(j)) || text(j) == '_' || text(j) == '.')
			fail(sprintf('invalid number ''%s'' at column %d', text(i:j), i));
		end
		literal = text(i:j - 1);
		value = str2double(literal);
		if ~isfinite(value)
			fail(sprintf('number ''%s'' is out of range', literal));
		end
		tokens(end + 1) = mkToken('number', literal, value, i - 1); %#ok<AGROW>
		i = j;
	elseif isAlpha(c) || c == '_'
		j = i;
		while j <= n && (isAlpha(text(j)) || isDigit(text(j)) || text(j) == '_'), j = j + 1; end
		tokens(end + 1) = mkToken('name', text(i:j - 1), 0, i - 1); %#ok<AGROW>
		i = j;
	elseif any(c == '+-*/()[],=')
		tokens(end + 1) = mkToken('symbol', c, 0, i - 1); %#ok<AGROW>
		i = i + 1;
	else
		fail(sprintf('unexpected character ''%s'' at column %d', c, i));
	end
end
tokens(end + 1) = mkToken('end', '', 0, n);
end

% ---- expression parser over tokens(pos:stop-1); state st = struct(tokens, pos, stop)

function t = peekToken(st)
if st.pos < st.stop
	t = st.tokens(st.pos);
elseif st.stop <= numel(st.tokens)
	t = st.tokens(st.stop);
else
	t = mkToken('end', '', 0, 0);
end
end

function tf = isSymbolAt(st, symbol)
tf = false;
if st.pos < st.stop
	t = st.tokens(st.pos);
	tf = strcmp(t.kind, 'symbol') && strcmp(t.text, symbol);
end
end

function failAt(st, message)
t = peekToken(st);
fail(sprintf('%s (column %d)', message, t.position + 1));
end

function s = describe(st)
t = peekToken(st);
if st.pos >= st.stop || strcmp(t.kind, 'end')
	s = 'end of the expression';
else
	s = ['''' t.text ''''];
end
end

function st = expectSymbol(st, symbol)
if ~isSymbolAt(st, symbol)
	failAt(st, sprintf('expected ''%s'' but found %s', symbol, describe(st)));
end
st.pos = st.pos + 1;
end

function [n, st] = parseSum(st)
[n, st] = parseProduct(st);
while isSymbolAt(st, '+') || isSymbolAt(st, '-')
	if strcmp(st.tokens(st.pos).text, '+'), op = 'add'; else, op = 'sub'; end
	st.pos = st.pos + 1;
	[right, st] = parseProduct(st);
	n = mkOp(op, n, right);
end
end

function [n, st] = parseProduct(st)
[n, st] = parseUnary(st);
while isSymbolAt(st, '*') || isSymbolAt(st, '/')
	if strcmp(st.tokens(st.pos).text, '*'), op = 'mul'; else, op = 'div'; end
	st.pos = st.pos + 1;
	[right, st] = parseUnary(st);
	n = mkOp(op, n, right);
end
end

function [n, st] = parseUnary(st)
if isSymbolAt(st, '-')
	st.pos = st.pos + 1;
	[inner, st] = parseUnary(st);
	n = mkOp('neg', inner);
	return
end
[n, st] = parsePrimary(st);
end

function [n, st] = parsePrimary(st)
if st.pos >= st.stop
	failAt(st, ['expected a value but found ' describe(st)]);
end
t = st.tokens(st.pos);
if strcmp(t.kind, 'number')
	st.pos = st.pos + 1;
	n = mkConst(t.value);
	return
end
if isSymbolAt(st, '(')
	st.pos = st.pos + 1;
	[n, st] = parseSum(st);
	st = expectSymbol(st, ')');
	return
end
if strcmp(t.kind, 'name')
	lower_ = lower(t.text);
	if any(strcmp(lower_, {'min', 'max', 'sqrt', 'exp', 'log', 'abs'}))
		st.pos = st.pos + 1;
		st = expectSymbol(st, '(');
		[first, st] = parseSum(st);
		if any(strcmp(lower_, {'min', 'max'}))
			st = expectSymbol(st, ',');
			[second, st] = parseSum(st);
			n = mkOp(lower_, first, second);
		else
			n = mkOp(lower_, first);
		end
		st = expectSymbol(st, ')');
		return
	end
	if strcmp(lower_, 'h')
		failAt(st, 'the lag ''h'' cannot be used in a parameter: parameters are constant or vary with the node, never with the lag');
	end
	if reserved(t.text)
		failAt(st, sprintf('''%s'' is a reserved word and cannot be used as a variable', t.text));
	end
	st.pos = st.pos + 1;
	n = mkVar(t.text, [], 0);
	return
end
failAt(st, ['expected a value but found ' describe(st)]);
end

% ---- model parser. P = struct(tokens, nTok, match); MATLAB index = C++ index + 1 and
% nTok (the index of the 'end' token) plays the role of the C++ "end".

function tf = pSym(P, i, symbol)
tf = i < P.nTok && strcmp(P.tokens(i).kind, 'symbol') && strcmp(P.tokens(i).text, symbol);
end

function tf = pOpen(P, i)
tf = pSym(P, i, '(') || pSym(P, i, '[');
end

function tf = pStructure(P, i)
tf = strcmp(P.tokens(i).kind, 'name') && any(strcmp(lower(P.tokens(i).text), ...
	{'nugget', 'spherical', 'exponential', 'gaussian', 'cubic', 'matern', 'stable', 'map'}));
end

function pFail(P, index, message)
fail(sprintf('%s (column %d)', message, P.tokens(index).position + 1));
end

function e = pExpression(P, from, to, what)
if from >= to
	pFail(P, min(from, P.nTok), ['missing value for ' what]);
end
st = struct('tokens', P.tokens, 'pos', from, 'stop', to);
[e, st] = parseSum(st);
if st.pos ~= to
	pFail(P, st.pos, sprintf('unexpected ''%s'' in %s', P.tokens(st.pos).text, what));
end
end

function parts = pSplit(P, from, to, symbol)
% Splits [from, to) at the top-level occurrences of symbol (',' or '+'): rows [start, stop).
parts = zeros(0, 2);
first = from;
i = from;
while i < to
	if pOpen(P, i)
		i = P.match(i) + 1;
		continue
	end
	if pSym(P, i, symbol)
		parts(end + 1, :) = [first, i]; %#ok<AGROW>
		first = i + 1;
	end
	i = i + 1;
end
parts(end + 1, :) = [first, to];
end

function m = parseModel(text)
tokens = tokenize(text);
nTok = numel(tokens);
if nTok == 1
	fail('empty model');
end
structureList = 'nugget, spherical, exponential, gaussian, cubic, matern, stable or map';
P = struct('tokens', tokens, 'nTok', nTok, 'match', zeros(1, nTok));

% matching bracket of each '(' or '['
stack = [];
for i = 1:nTok - 1
	if pOpen(P, i)
		stack(end + 1) = i; %#ok<AGROW>
	elseif pSym(P, i, ')') || pSym(P, i, ']')
		if isempty(stack)
			pFail(P, i, sprintf('unbalanced ''%s''', tokens(i).text));
		end
		open = stack(end);
		if strcmp(tokens(open).text, '(') ~= strcmp(tokens(i).text, ')')
			pFail(P, i, sprintf('''%s'' closed by ''%s''', tokens(open).text, tokens(i).text));
		end
		P.match(open) = i;
		stack(end) = [];
	end
end
if ~isempty(stack)
	pFail(P, stack(end), sprintf('unclosed ''%s''', tokens(stack(end)).text));
end

terms = pSplit(P, 1, nTok, '+');
result = cell(1, size(terms, 1));
tables = {};
for termIndex = 1:size(terms, 1)
	from = terms(termIndex, 1);
	to = terms(termIndex, 2);
	if from >= to
		pFail(P, min(from, nTok), 'empty term (a ''+'' without a structure)');
	end
	% the structure: a structure name at the top level of the term
	structureToken = to;
	i = from;
	while i < to
		if pOpen(P, i)
			for j = i + 1:P.match(i) - 1
				if pStructure(P, j) && ~(j + 1 < nTok && pSym(P, j + 1, '='))
					pFail(P, j, sprintf('structure ''%s'' inside brackets; write sill*%s(...)', tokens(j).text, lower(tokens(j).text)));
				end
			end
			i = P.match(i) + 1;
			continue
		end
		if pStructure(P, i)
			if structureToken ~= to
				pFail(P, i, 'two structures in one term; separate them with ''+''');
			end
			structureToken = i;
		end
		i = i + 1;
	end
	if structureToken == to
		i = from;
		while i + 1 < to
			if pOpen(P, i)
				i = P.match(i) + 1;
				continue
			end
			if strcmp(tokens(i).kind, 'name') && pSym(P, i + 1, '(') && ...
					~any(strcmp(lower(tokens(i).text), {'min', 'max', 'sqrt', 'exp', 'log', 'abs'}))
				pFail(P, i, sprintf('unknown structure ''%s'' (%s)', tokens(i).text, structureList));
			end
			i = i + 1;
		end
		termText = strjoin({tokens(from:to - 1).text}, '');
		pFail(P, from, sprintf('term ''%s'' has no structure (%s)', termText, structureList));
	end
	type = lower(tokens(structureToken).text);

	% ---- partial sill
	if structureToken == from
		sill = mkConst(1);
	else
		if ~pSym(P, structureToken - 1, '*')
			pFail(P, structureToken, sprintf('expected ''*'' between the partial sill and ''%s''', tokens(structureToken).text));
		end
		sillEnd = structureToken - 1;
		% a top-level binary '-' would silently change the meaning ("1-0.5*nugget")
		i = from;
		while i < sillEnd
			if pOpen(P, i)
				i = P.match(i) + 1;
				continue
			end
			if pSym(P, i, '-') && i > from
				previous = tokens(i - 1);
				binary = any(strcmp(previous.kind, {'number', 'name'})) || ...
					(strcmp(previous.kind, 'symbol') && any(strcmp(previous.text, {')', ']'})));
				if binary
					pFail(P, i, ['a structure cannot be subtracted; parenthesise a difference in a partial sill: (a - b)*' type]);
				end
			end
			i = i + 1;
		end
		sill = pExpression(P, from, sillEnd, ['the partial sill of ' type]);
	end

	% ---- arguments
	argumentsFrom = structureToken + 1;
	argumentsTo = structureToken + 1;
	if structureToken + 1 < to
		parenthesis = pSym(P, structureToken + 1, '(');
		if ~parenthesis || P.match(structureToken + 1) + 1 ~= to
			if parenthesis
				after = P.match(structureToken + 1) + 1;
			else
				after = structureToken + 1;
			end
			if pSym(P, after, '*') || pSym(P, after, '/')
				pFail(P, after, sprintf('the partial sill must come before the structure: sill*%s(...)', type));
			end
			pFail(P, after, sprintf('unexpected ''%s'' after %s', tokens(after).text, type));
		end
		argumentsFrom = structureToken + 2;
		argumentsTo = to - 1;
	end
	hasArguments = argumentsFrom < argumentsTo;

	range = {};
	angle = {};
	shape = [];
	table = [];
	useScale = false;
	if strcmp(type, 'nugget')
		if hasArguments
			pFail(P, argumentsFrom, 'nugget takes no parameter');
		end
	elseif strcmp(type, 'map')
		if ~hasArguments || argumentsTo ~= argumentsFrom + 1 || ~strcmp(tokens(argumentsFrom).kind, 'name')
			if hasArguments, where = argumentsFrom; else, where = structureToken; end
			pFail(P, where, 'map takes one table name: map(name)');
		end
		name = tokens(argumentsFrom).text;
		if ~validName(name)
			pFail(P, argumentsFrom, sprintf('''%s'' is a reserved word and cannot name a map', name));
		end
		table = mkVar(name, [], 0);
		tables{end + 1} = name; %#ok<AGROW>
	else
		if ~hasArguments
			pFail(P, structureToken, [type ' needs range= or scale=']);
		end
		seenRange = false;
		seenAngle = false;
		seenShape = false;
		argParts = pSplit(P, argumentsFrom, argumentsTo, ',');
		for q = 1:size(argParts, 1)
			a = argParts(q, 1);
			b = argParts(q, 2);
			if a >= b
				pFail(P, min(a, nTok), sprintf('empty argument in %s(...)', type));
			end
			if ~strcmp(tokens(a).kind, 'name') || ~pSym(P, a + 1, '=')
				pFail(P, a, sprintf('expected keyword=value in %s(...)', type));
			end
			keyword = lower(tokens(a).text);
			isRange = any(strcmp(keyword, {'range', 'scale'}));
			isShape = (strcmp(keyword, 'nu') && strcmp(type, 'matern')) || (strcmp(keyword, 'alpha') && strcmp(type, 'stable'));
			if ~isRange && ~isShape && ~strcmp(keyword, 'angle')
				allowed = 'range, scale, angle';
				if strcmp(type, 'matern'), allowed = [allowed ', nu']; end %#ok<AGROW>
				if strcmp(type, 'stable'), allowed = [allowed ', alpha']; end %#ok<AGROW>
				pFail(P, a, sprintf('unknown keyword ''%s'' for %s (allowed: %s)', tokens(a).text, type, allowed));
			end
			if isRange
				if seenRange, pFail(P, a, ['range/scale given twice in ' type]); end
				seenRange = true;
			elseif isShape
				if seenShape, pFail(P, a, sprintf('''%s'' given twice in %s', keyword, type)); end
				seenShape = true;
			else
				if seenAngle, pFail(P, a, sprintf('''%s'' given twice in %s', keyword, type)); end
				seenAngle = true;
			end
			valueFrom = a + 2;
			values = {};
			if pSym(P, valueFrom, '[') && P.match(valueFrom) + 1 == b
				if isShape
					pFail(P, valueFrom, [keyword ' takes a single value']);
				end
				if P.match(valueFrom) == valueFrom + 1
					pFail(P, valueFrom, ['empty vector for ' keyword]);
				end
				elements = pSplit(P, valueFrom + 1, P.match(valueFrom), ',');
				for e = 1:size(elements, 1)
					values{end + 1} = pExpression(P, elements(e, 1), elements(e, 2), [keyword ' of ' type]); %#ok<AGROW>
				end
			else
				values{end + 1} = pExpression(P, valueFrom, b, [keyword ' of ' type]); %#ok<AGROW>
			end
			if isRange
				if numel(values) > 4
					pFail(P, valueFrom, sprintf('%s has more than 4 values', keyword));
				end
				useScale = strcmp(keyword, 'scale');
				range = values;
			elseif isShape
				shape = values{1};
			else
				if numel(values) ~= 1 && numel(values) ~= 3
					pFail(P, valueFrom, 'angle takes one value (2D) or three values (3D)');
				end
				angle = values;
			end
		end
		if ~seenRange
			pFail(P, structureToken, [type ' needs range= or scale=']);
		end
		if strcmp(type, 'matern') && ~seenShape
			pFail(P, structureToken, 'matern needs nu=');
		end
		if strcmp(type, 'stable') && ~seenShape
			pFail(P, structureToken, 'stable needs alpha=');
		end
	end

	% ---- constant checks (variable parameters are checked when bound)
	sillValue = 0;
	if strcmp(sill.op, 'const'), sillValue = sill.value; end
	shapeValue = 1;
	if ~isempty(shape) && strcmp(shape.op, 'const'), shapeValue = shape.value; end
	problem = checkValues(type, sillValue, constantsOf(range), constantsOf(angle), shapeValue);
	if ~isempty(problem)
		pFail(P, structureToken, [type ': ' problem]);
	end
	result{termIndex} = mkTerm(sill, type, range, useScale, angle, shape, table);
end

% ---- names: a name is either a map table or a scalar variable
used = {};
for k = 1:numel(result)
	leaves = collectAll(result{k});
	for j = 1:numel(leaves)
		used{end + 1} = leaves{j}.name; %#ok<AGROW>
	end
end
for k = 1:numel(tables)
	if any(strcmp(used, tables{k}))
		fail(sprintf('''%s'' is used both as a map table and as a parameter variable', tables{k}));
	end
end
m = g2sCov('model', result);
end
