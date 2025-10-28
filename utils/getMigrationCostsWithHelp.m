function tau_h = getMigrationCostsWithHelp(tau, alpha)
% GETMIGRATIONCOSTSWITHHELP Build cost tensor conditional on help offers.
%
%   The baseline migration cost matrix tau ∈ ℝ^{N×N} encodes the expense of
%   moving from origin i to destination j. When a help vector h ∈ {0,1}^N is in
%   effect, the destination-specific costs are discounted by the scalar factor
%   alpha whenever h(j) = 1. This function expands the two-dimensional costs into
%   a three-dimensional tensor tau_h(:, :, h_idx) that lists the effective cost
%   for every help configuration indexed by h_idx = 1,...,2^N.
%
%   INPUTS
%       tau     [N x N]     Baseline migration cost matrix.
%       alpha   [scalar]    Proportional cost reduction (0 < alpha ≤ 1).
%
%   OUTPUTS
%       tau_h   [N x N x 2^N]  Cost tensor indexed by help-vector identifier.
%
%   AUTHOR: Agustin Deambrosi
%   DATE:   October 2025
% =========================================================================

%% Step 1: Dimensions and help enumeration
	N	= size(tau, 1);	% Number of locations inferred from tau
	H	= 2^N;	% Total count of binary help vectors
	hmat	= dec2bin(0:H-1) - '0';	% [H x N] matrix of binary offers by configuration

%% Step 2: Initialise the destination-specific cost tensor
	tau_h	= repmat(tau, 1, 1, H);	% Replicate tau across the third dimension → [N x N x H]

%% Step 3: Apply discounts for destinations receiving help
	for h_idx = 1:H
		h_vec	= hmat(h_idx, :);	% 1 x N binary vector for configuration h_idx
		for j = 1:N
			if h_vec(j)
				tau_h(:, j, h_idx)	= alpha * tau(:, j);	% Replace destination-j column with its helped cost profile
			end
		end
	end
end
