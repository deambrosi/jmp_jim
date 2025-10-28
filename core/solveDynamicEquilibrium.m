function [pol_eqm, M_eqm, it_count, vf_path] = solveDynamicEquilibrium(M0, vf_terminal, m0, dims, params, grids, indexes, matrices, settings, scenario)
% SOLVEDYNAMICEQUILIBRIUM Solve for the model's dynamic transition equilibrium.
%
%   [pol_eqm, M_eqm, it_count, vf_path] = solveDynamicEquilibrium(M0,
%       vf_terminal, m0, dims, params, grids, indexes, matrices, settings,
%       scenario)
%
%   This routine implements the fixed-point problem that delivers the
%   transition dynamics of the model. Starting from a guess M0 of the share of
%   networked agents by destination and date ([N x T]), it alternates between:
%       1. Backward iteration of value and policy functions via PolicyDynamics,
%          preserving the [S x Na x N] dimensionality of value arrays and the
%          [S x Na x N x N] / [S x Na x N x N x H] structure of migration rules.
%       2. Forward simulation of agent histories using one of the simulate*()
%          routines, which produce a new distribution path M_new.
%       3. A weighted convergence criterion that emphasizes early transition
%          periods when measuring ||M_new - M_eqm||.
%
%   INPUTS:
%       M0           - [N x T] initial guess for the networked-mass path.
%                      Each column corresponds to a period; each row to a
%                      location. Rows sum to the total networked share in that
%                      period.
%       vf_terminal  - Struct with terminal-period value functions (fields .V,
%                      .Vn, .R, .Rn, etc.), each stored as [S x Na x N]
%                      matrices consistent with PolicyDynamics.
%       m0           - [Nagents x 1] struct array describing the initial agent
%                      population used for the forward simulations.
%       dims         - Struct with model dimensions (fields N, Na, S, H, ...).
%       params       - Parameter struct (preferences, transition matrices,
%                      migration costs, etc.).
%       grids        - Struct with state-space grids.
%       indexes      - Struct of index mappings used inside PolicyDynamics.
%       matrices     - Struct holding precomputed payoff matrices (Ue,
%                      a_prime, A_prime) used to accelerate the Bellman step.
%       settings     - Struct with numerical settings (T, tolerance levels,
%                      maximum iterations, Nagents, etc.).
%       scenario     - Struct describing the policy experiment. If omitted or
%                      empty, the routine defaults to the benchmark case.
%                      Supported scenario.type values:
%                        'benchmark'  -> uses simulateAgents.m
%                        'transport'  -> uses simulateAgentsTransportAid.m with
%                                         additional fields (startPeriod,
%                                         wealthThreshold, budget, massIncrease)
%                        'shelter'    -> uses simulateAgentsShelterAid.m with
%                                         its associated parameters.
%
%   OUTPUTS:
%       pol_eqm   - Struct with the converged policy cell arrays (.a, .an, .mu,
%                   .mun) over the transition horizon.
%       M_eqm     - [N x T] matrix representing the fixed-point networked-mass
%                   path.
%       it_count  - Number of equilibrium iterations performed.
%       vf_path   - Cell array of value-function structs from the final
%                   iteration (length T).
%
%   AUTHOR: Agustin Deambrosi
%   DATE:   October 2025
% =========================================================================

if nargin < 10 || isempty(scenario)
scenario = struct();
end

if ~isfield(scenario, 'type') || isempty(scenario.type)
scenario.type = 'benchmark';
end

%% 1. Initialization
	T           = settings.T;
	diffM       = 1;
	it_count    = 0;
	M_eqm       = M0;              % [N x T] current guess for networked masses

	% Exponential weights place more emphasis on early periods when measuring
	% convergence of the [N x T] distribution path.
	beta_weight  = 0.7;
	time_weights = beta_weight .^ (0:T-1);      % [1 x T]
	time_weights = time_weights / sum(time_weights);

%% 2. Iteration Loop
	while (diffM > settings.tolM) && (it_count < settings.MaxItJ)
		it_count = it_count + 1;

		% Step 1: Update policy functions via backward induction. PolicyDynamics
		%         returns vf_path (cell of [S x Na x N] arrays) and pol_new, whose
		%         migration blocks retain their [S x Na x N x N (x H)] shapes.
		[vf_path, pol_new] = PolicyDynamics(M_eqm, vf_terminal, dims, params, grids, indexes, matrices, settings);

		% Step 2: Build the help distribution path implied by M_eqm and run the
		%         appropriate forward simulation. G_path is [H x T], where each
		%         column enumerates the probability of the H help vectors.
		G_path = zeros(dims.H, T);
		for t = 1:T
			G_path(:, t) = computeG(M_eqm(:, t), params.ggamma);
		end

		switch lower(string(scenario.type))
		case "transport"
			G_aug = computeTransportAidHelpPath(M_eqm, params, scenario, T);
			[~, M_new, ~, ~] = simulateAgentsTransportAid(m0, pol_new, ...
				G_path, G_aug, dims, params, grids, settings, scenario);

		case "shelter"
			[~, M_new, ~, ~] = simulateAgentsShelterAid(m0, pol_new, ...
				G_path, dims, params, grids, settings, scenario);

		otherwise
			[~, M_new, ~] = simulateAgents(m0, pol_new, G_path, ...
				dims, params, grids, settings);
		end

		% Step 3: Compute the weighted norm of differences period by period.
		%         diff_per_t is [1 x T]; dividing by (1 + |M_new|) dampens large
		%         deviations in low-mass locations.
		diff_per_t = sum(abs(M_eqm - M_new) ./ (1 + abs(M_new)), 1);
		diffM      = sum(time_weights .* diff_per_t);

		% Step 4: Refresh the guess for the next iteration.
		M_eqm = M_new;
	end

	% Final output: pol_new already stores the last set of policy functions and
	% vf_path corresponds to that iteration's backward solution.
	pol_eqm = pol_new;
	end

%% ------------------------------------------------------------------------
	function G_aug = computeTransportAidHelpPath(M_path, params, scenario, T)
	% COMPUTETRANSPORTAIDHELPPATH Construct augmented help distributions for aid.
	%
	%   G_aug = computeTransportAidHelpPath(M_path, params, scenario, T)
	%
	%   INPUTS:
	%       M_path    - [N x T] matrix with the current guess of networked masses.
	%       params    - Parameter struct providing params.ggamma and params.G0.
	%       scenario  - Transport-aid scenario struct with fields:
	%                      .massIncrease (scalar or [N x 1])
	%                      .startPeriod  (integer in 1..T)
	%       T         - Transition length (scalar consistent with settings.T).
	%
	%   OUTPUTS:
	%       G_aug     - [H x T] augmented help probability matrix used by
	%                   simulateAgentsTransportAid.m. Prior to startPeriod it
	%                   matches the benchmark probabilities implied by M_path.
	%
	%   AUTHOR: Agustin Deambrosi
	%   DATE:   October 2025
	% =========================================================================

	if ~isfield(scenario, 'massIncrease') || isempty(scenario.massIncrease)
		error('Transportation aid scenario must include a massIncrease field.');
	end

	if ~isfield(scenario, 'startPeriod') || isempty(scenario.startPeriod)
		error('Transportation aid scenario must include a startPeriod field.');
	end

	massIncrease = scenario.massIncrease(:);
	if numel(massIncrease) == 1
		massIncrease = repmat(massIncrease, size(M_path, 1), 1);
	elseif numel(massIncrease) ~= size(M_path, 1)
		error('massIncrease must be either a scalar or an N x 1 vector.');
	end

	startPeriod = max(1, min(T, scenario.startPeriod));

	G_aug = zeros(size(params.G0, 1), T);
	for t = 1:T
		if t >= startPeriod
			G_aug(:, t) = computeG(M_path(:, t) + massIncrease, params.ggamma);
		else
			G_aug(:, t) = computeG(M_path(:, t), params.ggamma);
		end
	end
	end
