function [vf, pol] = noHelpEqm(dims, params, grids, indexes, matrices, settings)
% NOHELPEQM Compute the no-network migration equilibrium value functions.
%
%   Solves the dynamic programming problem in which migrants receive no help
%   from their network (G = G0). The routine performs fixed-point iteration on
%   the value functions defined over the full discrete state space described by
%   indexes.sz and returns both the converged values and the associated policy
%   objects produced by updateValueAndPolicy.
%
%   INPUTS:
%       dims    	- Struct of dimension counters (e.g., number of asset nodes,
%                 labor shocks, and experience types); indexes.sz matches these
%                 counts in the order expected by value-function arrays.
%       params  	- Model parameters, including the no-help transition matrix
%                 params.G0 passed to the updater.
%       grids   	- Collection of discretized state grids used inside the
%                 transition and interpolation routines.
%       indexes 	- Struct of helper indices and reshape utilities; indexes.sz is
%                 the multi-dimensional size of value arrays V and Vn.
%       matrices	- Packaged payoff and transition matrices used during the
%                 Bellman operator application.
%       settings	- Numerical tolerances and iteration limits (fields tolV and
%                 MaxItJ) governing convergence checks.
%
%   OUTPUTS:
%       vf      	- Struct with fields .V and .Vn, each of size indexes.sz,
%                 containing the converged value functions with and without
%                 network access.
%       pol     	- Struct of optimal policy rules returned by
%                 updateValueAndPolicy at convergence.
%
%   AUTHOR: Agustin Deambrosi
%   LAST REVISED: October 2025
% =========================================================================
%
%   STEP 1 (Initialization): Establish scalar trackers diffV and itV that log the
%   convergence distance and iteration count for the value-function iteration.
%   Both are scalars while all subsequent arrays respect the multi-dimensional
%   layout stored in indexes.sz.
    %% Initialization
    diffV               = 1;
    itV                 = 0;
%
%   STEP 2 (Initial guess): Fill val.V with ones(indexes.sz), producing a
%   multi-dimensional array whose dimensions correspond to the ordered state
%   components encoded in indexes.sz. The no-network guess val.Vn mirrors this
%   shape so the subsequent updates occur element-wise over the full grid.
    % Initialize value functions uniformly
    val.V               = ones(indexes.sz);     % With network
    val.Vn              = val.V;                % Without network
%
%   STEP 3 (Value-function iteration without help): Iterate on the Bellman
%   operator until convergence or until hitting the iteration cap. Each pass
%   calls updateValueAndPolicy, which expects the current value struct and
%   returns arrays vf.V and vf.Vn of the same dimension indexes.sz along with the
%   associated policies.
    %% Value Function Iteration (no help offers)
    while (diffV > settings.tolV) && (itV < settings.MaxItJ)
        itV             = itV + 1;
%
%       STEP 3.1 (Apply Bellman operator): UpdateValueAndPolicy maps the
%       multi-dimensional value guess in val (size indexes.sz) into new value
%       functions vf.V and vf.Vn while holding the transition matrix fixed at
%       params.G0, which enforces the no-network setting.
        [vf, pol] = updateValueAndPolicy(val, dims, params, grids, indexes, matrices, params.G0);
%
%       STEP 3.2 (Convergence metric): Compute diffV as the L1 distance across
%       all grid points. The subtraction vf.Vn - val.Vn is element-wise over the
%       indexes.sz-dimensional array, and the normalization keeps magnitudes
%       comparable. Assign val = vf to feed the new iterate into the next loop.
        % Update convergence criterion
        diffV   = sum(abs(vf.Vn - val.Vn) ./ (1 + abs(vf.Vn)), 'all');
        val             = vf;
    end

end
