function f = interp_migration(grid, xgrid, dims, V)
% INTERP_MIGRATION Interpolates post-migration continuation values.
%
%   SYNOPSIS:
%       f = interp_migration(grid, xgrid, dims, V)
%
%   DESCRIPTION:
%       Computes the value that agents expect after moving to every
%       destination/location pair, accounting for the migration cost that
%       shifts assets from the original grid `grid` to the post-migration
%       asset grid `xgrid`. The value function `V` is defined over the
%       state space (skill state S, assets Na, origin N). The function
%       interpolates these values so that we obtain continuation values for
%       every combination of origin, destination, and help vector.
%
%   INPUTS:
%       grid    - [Na x 1] column vector with the asset grid prior to paying migration costs.
%       xgrid   - [Na x N x N x H] array with destination-specific post-migration assets.
%       dims    - Struct with integer dimensions: Na, N, H, and S.
%       V       - [S x Na x N] value function evaluated before migration.
%
%   OUTPUT:
%       f       - [S x Na x N x N x H] continuation value for each origin,
%                 destination, and help configuration.
%
%   AUTHOR: Agustin Deambrosi
%   DATE: October 2025
% =========================================================================

	%% 1. Align value function so assets are the leading dimension.
	% V is [S x Na x N]; permuting to [Na x N x S] facilitates interpolation along assets.
	V_reordered = permute(V, [2, 3, 1]);

	%% 2. Extract representative slice used for interpolation.
	% Taking the first skill dimension collapses V to [Na x N]; the interpolant is identical across S.
	V_base = V_reordered(:, :, 1);

	%% 3. Replicate the base matrix across destinations and help offers.
	% repmat expands V_base to [Na x N x N x H], matching origin-destination-help combinations.
	V_expanded = repmat(V_base, 1, 1, dims.N, dims.H);
	% permute reorders to [Na x destination x origin x help] before flattening.
	V_expanded = permute(V_expanded, [1, 3, 2, 4]);

	%% 4. Flatten the arrays so each column corresponds to a specific (origin, destination, help).
	% Resulting matrices are [Na x (N*N*H)].
	V_flat = reshape(V_expanded, dims.Na, []);
	xgrid_flat = reshape(xgrid, dims.Na, []);

	%% 5. Interpolate asset-by-asset for every migration option.
	% Each column contains Na grid points; interpolation preserves that dimension.
	f_interp = zeros(size(V_flat));
	parfor j = 1:size(V_flat, 2)
		f_interp(:, j) = interp1(grid, V_flat(:, j), xgrid_flat(:, j), 'linear', 'extrap');
	end

	%% 6. Restore the multidimensional structure and replicate across skills.
	% Reshape back to [Na x N x N x H] corresponding to (assets, destination, origin, help).
	f_shaped = reshape(f_interp, dims.Na, dims.N, dims.N, dims.H);
	% Insert singleton skill dimension to prepare for broadcasting to all S levels.
	f_shaped = permute(f_shaped, [5, 1, 2, 3, 4]);

	% Repeat along the skill dimension to obtain [S x Na x N x N x H].
	f = repmat(f_shaped, [dims.S, 1, 1, 1, 1]);

end
