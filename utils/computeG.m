function G = computeG(M, ggamma)
% COMPUTEG Construct the probability mass function over binary help offers.
%
%   This routine evaluates the independent Bernoulli offer process implied by
%   the model. For each time period t = 1,...,T it enumerates every binary help
%   vector h ∈ {0,1}^N (where N is the number of locations) and stores the joint
%   probability G(h | M_t). The offer probability for location j is
%   π^j(M^j_t) = (M^j_t)^γ with elasticity parameter γ = ggamma. The returned
%   matrix G therefore stacks the 2^N probabilities for each period column-wise.
%
%   INPUTS
%       M       [N x T]    Migrant network masses by location and time.
%       ggamma  [scalar]   Elasticity governing help probabilities.
%
%   OUTPUTS
%       G       [2^N x T]  Probability mass over help vectors for each period.
%
%   AUTHOR: Agustin Deambrosi
%   DATE:   October 2025
% =========================================================================

%% Step 1: Retrieve dimensions and pre-compute offer probabilities
	[N, T]	= size(M);	% N locations and T periods implied by M ∈ ℝ^{N×T}
	P	= M.^ggamma;	% Element-wise π(M) evaluations preserving the [N x T] shape

%% Step 2: Enumerate binary help vectors
	Hmat	= dec2bin(0:(2^N - 1)) - '0';	% [2^N x N] matrix of all binary help configurations

%% Step 3: Allocate storage for the probability mass function
	G	= zeros(2^N, T);	% Preallocate [2^N x T] array for joint probabilities

%% Step 4: Loop over periods and help configurations
	for t = 1:T
		pi_t	= P(:, t)';	% 1 x N row of offer probabilities for period t
		one_minus	= 1 - pi_t;	% 1 x N complements for non-help events

		for h_idx = 1:2^N
			h	= Hmat(h_idx, :);	% 1 x N binary help vector corresponding to index h_idx
			G(h_idx, t)	= prod(pi_t.^h .* one_minus.^(1 - h));	% Scalar joint probability mass G(h_idx, t)
		end
	end
end
