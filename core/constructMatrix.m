function matrices = constructMatrix(dims, params, grids, indexes)
% CONSTRUCTMATRIX Assemble utility and wealth tensors for dynamic choices.
%
%	SYNTAX:
%		matrices = constructMatrix(dims, params, grids, indexes)
%
%	INPUTS:
%		dims (struct)	Dimension settings containing counts for locations, asset
%			grids, productivity states, and help statuses used in the solver.
%		params (struct)	Model primitives evaluated on location indices, including
%			fields `bbi`, `A`, `B`, and the migration cost tensor `ttau`.
%		grids (struct)	Asset and amenity grids produced by `setGridsAndIndices`.
%		indexes (struct)	Index tensors produced by `setGridsAndIndices` that map
%			between reshaped arrays and their economic interpretation.
%
%	OUTPUTS:
%		matrices (struct)	Precomputed tensors used throughout the dynamic
%			program:
%				.Ue			Utility of employed households, with dimension
%						[dims.S x dims.Na x dims.N x dims.na].
%				.a_prime	Coarse asset grid net of migration costs, shaped
%						[dims.Na x dims.N x dims.N x dims.H].
%				.A_prime	Wealth tensor replicated across productivity states with
%						[dims.S x dims.Na x dims.N x dims.N x dims.H].
%
%	NOTES:
%		All computations are purely algebraic and preserve the original code
%		logic; only documentation and formatting are updated.
%
%	AUTHOR: Agustin Deambrosi
%	DATE: October 2025
% =========================================================================
	%% 1. Compute consumption and utility (Ue)
	%	Build labor income on the post-savings grid. The resulting tensor `income`
	%	shares the [dims.S x dims.Na x dims.N x dims.na] shape of `indexes.I_ep` and
	%	`indexes.I_Np`, combining benefits (`bbi`) for the unemployed and wages (`A`)
	%	for the employed with amenity adjustments from `grids.psi`.
	income	= (2 - indexes.I_ep) .* params.bbi(indexes.I_Np) + ...
			 (indexes.I_ep - 1) .* params.A(indexes.I_Np) .* ...
			 (params.lb_pr.*(1-grids.psi(indexes.I_psip)) + params.ub_pr.*grids.psi(indexes.I_psip));
	%	Compute consumption on the fine asset grid by adding gross resources and
	%	subtracting chosen savings. `cons` inherits the same dimensions as `income`.
	cons	= (1 / params.bbeta) .* grids.agrid(indexes.I_ap) + ...
			 income - grids.ahgrid(indexes.I_app);
	%	Amenity shifters weight utility according to the destination location, again
	%	producing a [dims.S x dims.Na x dims.N x dims.na] tensor.
	amenity_weight	= params.B(indexes.I_Np) .* ...
			 (params.lb_am.*(1-grids.psi(indexes.I_psip)) + params.ub_am.*grids.psi(indexes.I_psip));
	%	Initialize utility and apply the log utility where feasible consumption is
	%	positive. Infeasible points are penalized with `-realmax`.
	Ue	= zeros(size(cons));
    on  =   ones(size(cons));
	Ue(cons > 0)	= amenity_weight(cons > 0) .* log( on(cons > 0) + cons(cons > 0));
	Ue(cons <= 0)	= -realmax;	% Penalize infeasible consumption
	%% 2. Compute after-migration wealth matrices (a' and A')
	%	Migration costs are stored as a [dims.Na x dims.N x dims.N x dims.H] tensor in
	%	`params.ttau`. The permutation adds a leading singleton dimension to align with
	%	the asset grid for broadcasting when subtracting from `grids.agrid`.
	mig_costs	= permute(params.ttau, [4, 1, 2, 3]);	% Adds singleton 4th dimension: wealth
	%	The resulting `a_prime` retains dimensions [dims.Na x dims.N x dims.N x dims.H].
	a_prime	= grids.agrid - mig_costs;	% Adjust for migration cost
	%	Expand post-migration wealth to match all productivity states. The first
	%	permutation creates a singleton leading dimension, and the replication via
	%	`repmat` produces a [dims.S x dims.Na x dims.N x dims.N x dims.H] tensor.
	A_Prime	= permute(a_prime, [5, 1, 2, 3, 4]);	% Expand to 5th dimension: state
	A_Prime	= repmat(A_Prime, [dims.S, 1, 1, 1, 1]);	% Copy across productivity states
	%% 3. Output struct
	%	Store the precomputed tensors for use in the solution routines.
	matrices.Ue	= Ue;
	matrices.a_prime	= a_prime;
	matrices.A_prime	= A_Prime;
end
