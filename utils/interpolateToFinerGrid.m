function f = interpolateToFinerGrid(grid, xgrid, Raux)
% INTERPOLATETOFINERGRID Interpolate value arrays onto a finer asset grid.
%
%   Remaps value matrices defined on a coarse asset grid to a finer grid used in
%   the optimization routine. All ancillary state dimensions (e.g., skill levels,
%   shock histories) are preserved during the transformation so that only the
%   asset dimension changes resolution.
%
%   INPUTS:
%       grid    	- [Na x 1] coarse asset grid indexing the 2nd dimension of Raux.
%       xgrid   	- [na x 1] finer asset grid at which interpolated values are
%                 required.
%       Raux    	- Array of size [K x Na x N1 x ...] where Na corresponds to
%                 length(grid); K and the trailing dimensions collect additional
%                 state variables.
%
%   OUTPUT:
%       f       	- Array of size [K x 1 x N1 x ... x na] delivering the same
%                 economic objects as Raux but evaluated on xgrid and arranged to
%                 match the maximization routines (asset dimension last).
%
%   AUTHOR: Agustin Deambrosi
%   LAST REVISED: October 2025
% =========================================================================
%
%   STEP 1 (Reorder for interpolation): Bring the asset dimension to the leading
%   position so linear interpolation can operate along rows while ancillary
%   dimensions remain grouped in the trailing indices.
    %% 1. Move asset grid dimension to front for interpolation
    Raux                        = permute(Raux, [2, 1, 3, 4]);  % [Na x K x N x ...]
%
%   STEP 2 (Stack ancillary dimensions): Record the full size vector before
%   collapsing the trailing indices into columns. The reshape produces a matrix
%   with Na rows (asset points) and one column per stacked state combination.
    %% 2. Flatten all other dimensions for batch interpolation
    sz                          = size(Raux);                  % Save original size
    Rflat                       = reshape(Raux, sz(1), []);     % [Na x (K*N*...)]
%
%   STEP 3 (Interpolate each column): Apply linear interpolation with
%   extrapolation, mapping every stacked column from grid to xgrid. The output has
%   na rows corresponding to the finer asset grid but retains the column count of
%   stacked states.
    %% 3. Interpolate using linear method + extrapolation
    Rinterp                     = interp1(grid, Rflat, xgrid, 'linear', 'extrap');  % [na x (K*N*...)]
%
%   STEP 4 (Restore dimensions): Replace the first entry of sz with na, then
%   reshape the interpolated matrix back to the multidimensional structure with
%   the finer asset resolution.
    %% 4. Reshape back to multi-dimensional form
    sz(1)                       = length(xgrid);               % Replace Na with na
    Rreshaped           = reshape(Rinterp, sz);         % [na x K x N x ...]
%
%   STEP 5 (Final orientation): Permute dimensions so the asset grid becomes the
%   last index, yielding the [K x 1 x N1 x ... x na] arrangement required by the
%   calling maximization routines.
    %% 5. Reorder: [K x 1 x N x na] for compatibility with maximization
    f                           = permute(Rreshaped, [2, 4, 3, 1]);

end
