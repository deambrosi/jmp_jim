function [vf_path, pol_path] = PolicyDynamics(M1, vf_terminal, dims, params, grids, indexes, matrices, settings)
% POLICYDYNAMICS Evaluate transition-path value and policy functions.
%
%   [vf_path, pol_path] = PolicyDynamics(M1, vf_terminal, dims, params,
%       grids, indexes, matrices, settings)
%
%   This routine performs the backward-recursion step of the dynamic
%   equilibrium algorithm. Starting from the terminal value functions, it
%   iterates backward over the transition horizon, updating continuation
%   values and policy functions that govern savings and migration decisions.
%
%   INPUTS:
%       M1           - [N x T] guess for networked mass by location and period.
%                      Each column aggregates the share of networked agents
%                      across the N = dims.N destinations; rows sum to the
%                      measure of networked households in that period.
%       vf_terminal  - Struct containing terminal-period value arrays with
%                      dimensions [S x Na x N] for unattached agents and the
%                      analogous matrices for networked agents (.V, .Vn, etc.).
%       dims         - Struct of model dimensions, including N (locations),
%                      Na (asset grid points), S (idiosyncratic states), and H
%                      (help-vector states).
%       params       - Struct of structural parameters used by the Bellman
%                      operator; this includes the curvature parameter ggamma
%                      and objects such as params.ttau(N x N x H) with
%                      migration costs.
%       grids        - Struct of discretized asset grids. grids.agrid is
%                      [Na x 1], while grids.ahgrid indexes the finer
%                      post-decision asset grid used by the policy update.
%       indexes      - Collection of index maps used to vectorize the Bellman
%                      operator (see utils/setGridsAndIndices.m).
%       matrices     - Pre-computed payoff matrices (Ue, a_prime, A_prime) with
%                      dimensions compatible with the policy update routine.
%       settings     - Struct with iterative-solution settings, notably the
%                      transition length T = settings.T.
%
%   OUTPUTS:
%       vf_path      - Cell array of length T holding structs of value
%                      functions at each date, where each value matrix retains
%                      the [S x Na x N] dimensionality of the Bellman state.
%       pol_path     - Struct with cell arrays of transition policies. Each
%                      field (.a, .an, .mu, .mun) is a cell array of length T
%                      containing policy tensors with dimensions described in
%                      updateValueAndPolicy.m (e.g., .mu{t} is [S x Na x N x N]).
%
%   AUTHOR: Agustin Deambrosi
%   DATE:   October 2025
% =========================================================================

%% 1. Initialize storage for the backward recursion
	T				= settings.T;
	vf_path			= cell(T, 1);
	pol_path.a		= cell(T, 1);
	pol_path.an		= cell(T, 1);
	pol_path.mu		= cell(T, 1);
	pol_path.mun	= cell(T, 1);
	G_path0			= computeG(M1, params.ggamma);	% [H x T] help-vector probabilities

	vf_path{T}		= vf_terminal;	% Terminal condition anchors recursion

%% 2. Backward induction from period T-1 down to period 1
	for t = T-1:-1:1
		% Step 2A: Extract the help distribution G_t for period t. This column is
		%          [H x 1], summarizing the probability of each help vector that
		%          networked agents face when updating their migration policies.
		G_path_t		= G_path0(:, t);

		% Step 2B: Update the value and policy objects using the period-(t+1)
		%          continuation values and the help distribution G_path_t. The
		%          policy tensors returned keep their original dimensions,
		%          including the [S x Na x N x N] structure of the migration rules.
		[vf_t, pol_t]	= updateValueAndPolicy( ...
			vf_path{t+1}, dims, params, grids, indexes, matrices, G_path_t);

		% Step 2C: Store the updated objects so they are available both for the
		%          recursion at t-1 and for the forward-simulation step of the
		%          equilibrium algorithm.
		vf_path{t}		= vf_t;
		pol_path.a{t}	= pol_t.a;
		pol_path.an{t}	= pol_t.an;
		pol_path.mu{t}	= pol_t.mu;
		pol_path.mun{t}= pol_t.mun;
	end
end
