function [vf, pol] = updateValueAndPolicy(val, dims, params, grids, indexes, matrices, G)
% UPDATEVALUEANDPOLICY Update value functions and policy rules under migration.
%
%   SYNOPSIS:
%       [vf, pol] = updateValueAndPolicy(val, dims, params, grids, indexes, matrices, G)
%
%   DESCRIPTION:
%       Refreshes the continuation values, asset choices, and migration
%       probabilities for the household dynamic problem. The function
%       considers both agents outside the network (n = 0) and agents within
%       the network (n = 1) who may receive help offers. It performs the
%       expectation over future shocks, interpolates the value of migrating
%       using post-migration asset grids, and maximizes over savings choices.
%
%   INPUTS:
%       val      - Struct containing current value functions:
%                     .V  [S x Na x N]    for agents without network access.
%                     .Vn [S x Na x N]    for agents inside the network.
%       dims     - Struct with dimensions: S (skill), Na (asset grid),
%                  N (locations), H (help states).
%       params   - Struct with parameters such as transition matrix P [S x S],
%                  discount factor bbeta, help re-entry cchi, scale CONS,
%                  and extreme-value shape nnu.
%       grids    - Struct of grids: agrid [Na x 1] and ahgrid [Nah x 1] (finer asset grid).
%       indexes  - Struct of index helpers including II selecting the stay-put
%                  destination and reshaping helpers for expectations.
%       matrices - Struct with precalculated matrices:
%                     Ue        [S x Na x N x Nah] period utility including asset choice.
%                     a_prime   [Na x N x N x H] post-migration assets per origin/destination/help.
%                     A_prime   [Na x N x N x H] feasibility indicator for migration assets.
%       G        - [H x 1] probability mass over help offers for network agents.
%
%   OUTPUTS:
%       vf       - Struct with updated value and continuation functions:
%                     .V   [S x Na x N]
%                     .Vn  [S x Na x N]
%                     .R   [S x Na x N]
%                     .Rn  [S x Na x N]
%       pol      - Struct with policy indices and probabilities:
%                     .a   [S x Na x N]    optimal asset index (n = 0)
%                     .an  [S x Na x N]    optimal asset index (n = 1)
%                     .mu  [S x Na x N x N] migration distribution without help
%                     .mun [S x Na x N x N x H] migration distribution with help
%
%   AUTHOR: Agustin Deambrosi
%   DATE: October 2025
% =========================================================================

	%% 1. Continuation values when staying in the origin location
    PT = permute(params.P, [2 1 3]);
    
    % pagemtimes multiplies P [S x S] with V along the skill dimension, yielding [S x Na x N].
	cont_no_mig = pagemtimes(PT, val.V);
	% Weighted combination of network and non-network values before transition: still [S x Na x N].
	cont_no_mig_net = pagemtimes(PT, (1 - params.cchi) * val.Vn + params.cchi * val.V);

	%% 2. Continuation values when migrating to any destination.
	% interp_migration returns [S x Na x N x N x H] over post-migration assets.
	cont_mig = interp_migration(grids.agrid, matrices.a_prime, dims, val.V);
	cont_mig_net = interp_migration(grids.agrid, matrices.a_prime, dims, (1 - params.cchi) * val.Vn + params.cchi * val.V);

	% Overwrite diagonal elements (origin equals destination) with stay values across help states.
	cont_mig(indexes.II) = repmat(cont_no_mig, 1, 1, 1, dims.H);
	cont_mig_net(indexes.II) = repmat(cont_no_mig_net, 1, 1, 1, dims.H);

	% Penalize asset combinations that are infeasible: set value to -Inf for every associated state.
	cont_mig(matrices.A_prime < 0) = -Inf;
	cont_mig_net(matrices.A_prime < 0) = -Inf;

	%% 3. Migration probabilities using the generalized extreme value formulation.
	% Exponentiation keeps dimension [S x Na x N x N x H]; normalization later collapses help dimension.
	exp_vals = (exp(cont_mig / params.CONS)).^(1 / params.nnu);
	exp_vals_net = (exp(cont_mig_net / params.CONS)).^(1 / params.nnu);

	% Normalize over the destination dimension (fourth axis) to obtain probabilities.
	mmuu = exp_vals ./ sum(exp_vals, 4);
	mmuu_net = exp_vals_net ./ sum(exp_vals_net, 4);

	%% 4. Expected continuation values integrating over migration choices and help offers.
	% For non-network agents only the first help state exists, leaving [S x Na x N x N].
	exp_value = mmuu(:,:,:,:,1) .* cont_mig(:,:,:,:,1);
	exp_value(isnan(exp_value)) = 0;

	% Network agents integrate over H help states; reshape G to broadcast across dimensions.
	exp_value_net = mmuu_net .* cont_mig_net;
	exp_value_net(isnan(exp_value_net)) = 0;
	exp_value_net = permute(exp_value_net, [5, 1, 2, 3, 4]);
	exp_value_net = sum(exp_value_net .* reshape(G, [dims.H, 1, 1, 1, 1]), 1);
	exp_value_net = permute(exp_value_net, [2, 3, 4, 5, 1]);

	%% 5. Discounted expected values for each origin state.
	vf.R = params.bbeta * sum(exp_value, 4);
	vf.Rn = params.bbeta * sum(exp_value_net, 4);

	%% 6. Optimal asset policy on the finer grid.
	% interpolateToFinerGrid maps R from [Na] to the finer grid size Nah.
	interp_R = interpolateToFinerGrid(grids.agrid, grids.ahgrid, vf.R);
	total_val = matrices.Ue + interp_R;
	[vf.V, pol.a] = max(total_val, [], 4);

	interp_Rn = interpolateToFinerGrid(grids.agrid, grids.ahgrid, vf.Rn);
	total_valn = matrices.Ue + interp_Rn;
	[vf.Vn, pol.an] = max(total_valn, [], 4);

	%% 7. Migration policy outputs for both agent types.
	% Non-network policy collapses the help dimension; retain [S x Na x N x N].
	pol.mu = mmuu(:,:,:,:,1);
	% Network policy keeps the full [S x Na x N x N x H] distribution.
	pol.mun = mmuu_net;

end
