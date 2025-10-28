function [M_history, MIN_history, agentData, stats] = simulateAgentsTransportAid(m0, pol, G_base, G_aug, dims, params, grids, settings, program)
% SIMULATEAGENTSTRANSPORTAID Simulate dynamics under a transport-aid program.
%
%   [M_history, MIN_history, agentData, stats] = simulateAgentsTransportAid(
%       m0, pol, G_base, G_aug, dims, params, grids, settings, program)
%
%   This forward-simulation routine mirrors simulateAgents.m but augments the
%   migration step with the transportation-aid experiment used in the paper.
%   Benchmark help probabilities G_base (dimension [H x T]) describe the
%   timing and composition of network assistance. The program introduces an
%   alternative help path G_aug (also [H x T]) that reflects the artificial
%   mass of helpers financed by the aid budget. When the program is active
%   networked agents compare benchmark and augmented outcomes, receive a
%   subsidy whenever the augmented help expands the feasible destination set,
%   and update the program budget accordingly.
%
%   INPUTS:
%       m0        - [Nagents x 1] struct array of agent states at t = 1.
%                   Each entry stores .location (1..N), .wealth (1..Na),
%                   .state (1..S), and .network ∈ {0,1}.
%       pol       - Policy struct with fields .a, .an, .mu, .mun representing
%                   transition policies. Each field is either time-invariant
%                   or a cell array over t = 1,...,T-1. pol.mu{t} has
%                   dimension [S x Na x N x N]; pol.mun{t} has
%                   dimension [S x Na x N x N x H].
%       G_base    - [H x T] matrix of benchmark help probabilities for
%                   networked agents.
%       G_aug     - [H x T] matrix with augmented help probabilities implied
%                   by the transport-aid program.
%       dims      - Struct of model dimensions (fields N, Na, S, H).
%       params    - Parameter struct. This function uses params.ttau (N x N x H
%                   migration costs), params.aalpha (share paid by migrants),
%                   and params.cchi (probability of network attrition).
%       grids     - Struct with asset grids (grids.agrid [Na x 1],
%                   grids.ahgrid for the fine grid).
%       settings  - Struct with simulation settings: T (horizon) and
%                   Nagents (number of simulated households).
%       program   - Struct describing the transport program with fields:
%                     .type            = 'transport',
%                     .startPeriod     = first period when aid is available,
%                     .wealthThreshold = wealth-index eligibility cutoff,
%                     .budget          = total subsidy resources (model units),
%                     .massIncrease    = artificial helper mass (scalar or [N x 1]).
%
%   OUTPUTS:
%       M_history   - [N x T] matrix of location shares across periods.
%       MIN_history - [N x T] matrix of networked shares across periods.
%       agentData   - Struct with simulated trajectories:
%                       .location, .wealth, .state, .network,
%                       .helpIndex (index on the H help vectors),
%                       .aidAccepted (logical flag per agent-period).
%       stats       - Struct summarizing aid implementation (budget flows and
%                     acceptance counts).
%
%   AUTHOR: Agustin Deambrosi
%   DATE:   October 2025
% =========================================================================

%% 1. Validate inputs and prepare policies
	if ~isfield(program, 'startPeriod') || ~isfield(program, 'wealthThreshold') || ...
			~isfield(program, 'budget')
		error('Transportation aid program must include startPeriod, wealthThreshold, and budget fields.');
	end

	if ~iscell(pol.a)
		pol.a= repmat({pol.a}, settings.T-1, 1);
		pol.an= repmat({pol.an}, settings.T-1, 1);
		pol.mu= repmat({pol.mu}, settings.T-1, 1);
		pol.mun= repmat({pol.mun}, settings.T-1, 1);
	end

%% 2. Basic dimensions and storage
	T= settings.T;
	numAgents= settings.Nagents;
	numLocations= dims.N;
	H= size(G_base, 1);

	locationTraj= zeros(numAgents, T);
	wealthTraj= zeros(numAgents, T);
	stateTraj= zeros(numAgents, T);
	networkTraj= zeros(numAgents, T);
	helpIndexTraj= ones(numAgents, T);% default help index corresponds to no additional support
	aidAccepted= false(numAgents, T);

	tauBase= params.ttau(:, :, 1);
	alpha= params.aalpha;

	% Mapping from help index to binary vector of length N (columns are destinations)
	helpMatrix= dec2bin(0:H-1, numLocations) - '0';
	powersOfTwo = 2.^((numLocations-1):-1:0);

	% Load initial conditions from m0 into the trajectory arrays
	for agentIdx = 1:numAgents
		locationTraj(agentIdx, 1)= m0(agentIdx).location;
		wealthTraj(agentIdx, 1)= m0(agentIdx).wealth;
		stateTraj(agentIdx, 1)= m0(agentIdx).state;
		networkTraj(agentIdx, 1)= m0(agentIdx).network;
	end

%% 3. Aid bookkeeping
	stats = struct();
	stats.initialBudget= program.budget;
	stats.remainingBudget= program.budget;
	stats.totalAidSpent= 0;
	stats.acceptedMoves= 0;
	stats.acceptedByDestination = zeros(numLocations, 1);
	stats.aidSpendingTimeline= zeros(1, T);

	startPeriod= max(1, min(T, program.startPeriod));
	wealthThreshold = program.wealthThreshold;

%% 4. Simulation: loop over time, then agents
	for t = 1:(T-1)
		isProgramActive = (t >= startPeriod) && (stats.remainingBudget > 0);
		for agentIdx = 1:numAgents
			loc = locationTraj(agentIdx, t);
			wea = wealthTraj(agentIdx, t);
			sta = stateTraj(agentIdx, t);
			net = networkTraj(agentIdx, t);

%% A) Saving decision on the fine grid
			if net == 1
				a_fine = pol.an{t}(sta, wea, loc);
			else
				a_fine = pol.a{t}(sta, wea, loc);
			end
			a_fine = min(max(a_fine, 1), numel(grids.ahgrid));
			[~, nextWea] = min(abs(grids.agrid - grids.ahgrid(a_fine)));

%% B) Migration decision under benchmark probabilities
			if net == 1
				u_help = rand();
				G_t    = G_base(:, t);
				cumG   = cumsum(G_t);
				h_real = find(cumG >= u_help, 1);
				if isempty(h_real)
					h_real = H;
				end

				h_real_vec = helpMatrix(h_real, :);

				migProb = squeeze(pol.mun{t}(sta, wea, loc, :, h_real));
				migProb = migProb / sum(migProb);
				u_move  = rand();
				cumMig  = cumsum(migProb);
				dest_real = find(cumMig >= u_move, 1);
				if isempty(dest_real)
					[~, dest_real] = max(migProb);
				end

				dest_effective = dest_real;
				help_effective = h_real;
				aidGranted     = false;

%% C) Evaluate counterfactual decision with augmented help
				eligible = isProgramActive && (wea <= wealthThreshold);
				if eligible
					G_aug_t = G_aug(:, t);

					% Recover marginal help probabilities and construct artificial mass
					P_base = (G_t.'    * helpMatrix);% [1 x N] probability of assistance by destination
					P_aug  = (G_aug_t.' * helpMatrix);
					denom  = max(1 - P_base, eps);
					p_art  = max(0, min(1, (P_aug - P_base) ./ denom));

					h_art_vec   = double(rand(1, numLocations) < p_art);
					h_tilde_vec = max(h_real_vec, h_art_vec);
					h_union_idx = 1 + h_tilde_vec * powersOfTwo.';
					h_union_idx = round(h_union_idx);

					migProb_tilde = squeeze(pol.mun{t}(sta, wea, loc, :, h_union_idx));
					migProb_tilde = migProb_tilde / sum(migProb_tilde);
					cumMigTilde   = cumsum(migProb_tilde);
					dest_tilde    = find(cumMigTilde >= u_move, 1);
					if isempty(dest_tilde)
						[~, dest_tilde] = max(migProb_tilde);
					end

					if (dest_tilde ~= dest_real) && (dest_tilde ~= loc)
						grossCost = tauBase(loc, dest_tilde);
						subsidy   = (1 - alpha) * grossCost;
						if (stats.remainingBudget - subsidy) >= -1e-12 && subsidy > 0
							dest_effective = dest_tilde;
							help_effective = h_union_idx;
							aidGranted     = true;

							stats.remainingBudget      = stats.remainingBudget - subsidy;
							stats.totalAidSpent        = stats.totalAidSpent + subsidy;
							stats.acceptedMoves        = stats.acceptedMoves + 1;
							stats.acceptedByDestination(dest_tilde) = stats.acceptedByDestination(dest_tilde) + 1;
							stats.aidSpendingTimeline(t)            = stats.aidSpendingTimeline(t) + subsidy;
						end
					end
				end
			else
				migProb = squeeze(pol.mu{t}(sta, wea, loc, :));
				migProb = migProb / sum(migProb);
				u_move  = rand();
				cumMig  = cumsum(migProb);
				dest_effective = find(cumMig >= u_move, 1);
				if isempty(dest_effective)
					[~, dest_effective] = max(migProb);
				end
				help_effective = 1;% no help when outside the network
				aidGranted     = false;
			end

%% D) Wealth and state transitions given the chosen destination
			if dest_effective ~= loc
				migCost   = params.ttau(loc, dest_effective, help_effective);
				newAssets = grids.agrid(nextWea) - migCost;
				[~, nextWea] = min(abs(grids.agrid - newAssets));
				sta       = 1;% reset shock indices after migration
			else
				transCdf = cumsum(params.P(:, sta, loc));
				sta      = find(transCdf >= rand(), 1);
			end

%% E) Network status update
			if net == 1 && dest_effective ~= 1
				if rand() < params.cchi
					net = 0;
				end
			end

%% F) Record next-period states
			locationTraj(agentIdx, t+1)= dest_effective;
			wealthTraj(agentIdx, t+1)= nextWea;
			stateTraj(agentIdx, t+1)= sta;
			networkTraj(agentIdx, t+1)= net;
			helpIndexTraj(agentIdx, t+1) = help_effective;
			aidAccepted(agentIdx, t+1)= aidGranted;
		end
	end

%% 5. Aggregate location shares
	M_history= zeros(numLocations, T);
	MIN_history= zeros(numLocations, T);
	for t = 1:T
		locs = locationTraj(:, t);
		nets = networkTraj(:, t);
		M_history(:, t)= accumarray(locs, 1, [numLocations, 1]) / numAgents;
		MIN_history(:, t)= accumarray(locs(nets == 1), 1, [numLocations, 1]) / numAgents;
	end

%% 6. Prepare agent-level output
	agentData.location= locationTraj;
	agentData.wealth= wealthTraj;
	agentData.state= stateTraj;
	agentData.network= networkTraj;
	agentData.helpIndex= helpIndexTraj;
	agentData.aidAccepted= aidAccepted;

%% 7. Finalize statistics
	stats.finalBudget = stats.remainingBudget;
	end
