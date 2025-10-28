function [M_history, MIN_history, agentData] = simulateAgents(m0, pol, G_dist, dims, params, grids, settings)
% SIMULATEAGENTS Forward-simulate agents under a given transition policy.
%
%   [M_history, MIN_history, agentData] = simulateAgents(m0, pol, G_dist,
%       dims, params, grids, settings)
%
%   This routine executes the forward step of the dynamic equilibrium
%   algorithm. Taking the dynamic policy functions as given, it tracks the
%   evolution of a cross-section of Nagents households for T periods. For
%   each agent it records the discrete state tuple (location, wealth index,
%   productivity state, network membership) as well as any realizations of the
%   external help vector. When the policy functions are stored as cell arrays
%   of tensors, each migration decision uses the [S x Na x N x N] or
%   [S x Na x N x N x H] dimensions provided by pol.mu and pol.mun.
%
%   INPUTS:
%       m0        - [Nagents x 1] struct array with initial agent-level states.
%                   Each struct stores .state (1..S), .wealth (1..Na)
%                   .location (1..N), and .network ∈ {0,1}.
%       pol       - Struct of dynamic policies with fields .a, .an, .mu, .mun.
%                   Each field is either a time-invariant tensor or a cell
%                   array over t = 1,...,T-1. When stored as cell arrays,
%                   .a{t} and .an{t} map [S x Na x N] → indices on the fine
%                   asset grid, .mu{t} is [S x Na x N x N], and .mun{t} is
%                   [S x Na x N x N x H].
%       G_dist    - [H x T] matrix where column t contains the probability
%                   mass function over H distinct help vectors faced by
%                   networked agents in period t.
%       dims      - Struct with core dimensions (fields N, Na, S, H) used to
%                   size arrays during simulation.
%       params    - Parameter struct; this function uses params.ttau (N x N x H
%                   migration costs), params.P (S x S transition matrix), and
%                   params.cchi (probability of network loss when away from
%                   the home location).
%       grids     - Struct with asset grids. grids.agrid is [Na x 1] while
%                   grids.ahgrid indexes the finer post-decision grid used to
%                   evaluate the savings choice.
%       settings  - Struct with simulation settings including T (horizon) and
%                   Nagents (number of agents simulated in parallel).
%
%   OUTPUTS:
%       M_history   - [N x T] matrix of location masses. Column t reports the
%                     share of agents residing in each location at the start of
%                     period t.
%       MIN_history - [N x T] matrix of networked masses. Column t reports the
%                     share of agents that both reside in each location and
%                     remain networked at the start of period t.
%       agentData   - Struct collecting the simulated trajectories with fields
%                     .location, .wealth, .state, .network (each [Nagents x T]).
%
%   AUTHOR: Agustin Deambrosi
%   DATE:   October 2025
% =========================================================================

%% 1. Setup simulation horizon and policy containers
	T				= settings.T;

	isTimeInvariant	= ~iscell(pol.a);
	if isTimeInvariant
		pol.a	= repmat({pol.a}, T-1, 1);
		pol.an	= repmat({pol.an}, T-1, 1);
		pol.mu	= repmat({pol.mu}, T-1, 1);
		pol.mun	= repmat({pol.mun}, T-1, 1);
	end

	numAgents		= settings.Nagents;
	numLocations	= dims.N;

%% 2. Preallocate full agent trajectories
	locationTraj	= zeros(numAgents, T);	% [Nagents x T] location indices
	wealthTraj	= zeros(numAgents, T);	% [Nagents x T] asset-grid indices
	stateTraj		= zeros(numAgents, T);	% [Nagents x T] idiosyncratic state indices
	networkTraj	= zeros(numAgents, T);	% [Nagents x T] binary network status

%% 3. Simulation loop across agents (parallelized)
	parfor agentIdx = 1:numAgents
		% Step 3A: Pull initial state for agent agentIdx and allocate workspace
		agent	= m0(agentIdx);
		locHist	= zeros(1, T);
		weaHist	= zeros(1, T);
		staHist	= zeros(1, T);
		netHist	= zeros(1, T);

		% Step 3B: Store t = 1 initial conditions drawn from m0
		loc	= agent.location;
		wea	= agent.wealth;
		sta	= agent.state;
		net	= agent.network;

		locHist(1)	= loc;
		weaHist(1)	= wea;
		staHist(1)	= sta;
		netHist(1)	= net;

		for t = 1:(T-1)
			% Step 3C: Saving choice on the fine post-decision grid. pol.a{t} and
			%          pol.an{t} map the discrete triple (sta,wea,loc) into an index
			%          on grids.ahgrid (length = numel(grids.ahgrid)).
			if net == 1
				a_fine = pol.an{t}(sta, wea, loc);
			else
				a_fine = pol.a{t}(sta, wea, loc);
			end
			a_fine = min(max(a_fine, 1), length(grids.ahgrid));
			[~, nextWea] = min(abs(grids.agrid - grids.ahgrid(a_fine)));

			% Step 3D: Migration draw. When networked, migProb is extracted from the
			%          [S x Na x N x N x H] tensor pol.mun{t}; otherwise it comes from
			%          the [S x Na x N x N] tensor pol.mu{t}. G_dist(:, t) supplies the
			%          H-dimensional probability of each help vector at date t.
			if net == 1
				G_t	= G_dist(:, t);
				h_idx	= find(cumsum(G_t) >= rand(), 1);
				migProb = squeeze(pol.mun{t}(sta, wea, loc, :, h_idx));
			else
				migProb = squeeze(pol.mu{t}(sta, wea, loc, :));
				h_idx	= 1;	% default help index when outside the network
			end
			migProb = migProb / sum(migProb);
			nextLoc = find(cumsum(migProb) >= rand(), 1);

			% Step 3E: Guard against numerical issues that leave nextLoc empty
			if isempty(nextLoc)
				[~, nextLoc] = max(migProb);
			end

			% Step 3F: Update wealth index and productivity state. When migrating,
			%          the cost params.ttau(loc,nextLoc,h_idx) (with dimensions
			%          [N x N x H]) is deducted before mapping back to the coarse grid.
			if nextLoc ~= loc
				migCost	= params.ttau(loc, nextLoc, h_idx);
				newA	= grids.agrid(nextWea) - migCost;
				[~, nextWea] = min(abs(grids.agrid - newA));
				sta	= 1;	% reset (η, ψ) state indices after migration
			else
				transCdf = cumsum(params.P(:, sta, loc)');
				sta	= find(transCdf >= rand(), 1);
			end

			% Step 3G: Potentially lose network membership when away from home
			if net == 1 & nextLoc ~= 1
				if rand() < params.cchi
					net = 0;
				end
			end

			% Step 3H: Record next-period state variables for agentIdx
			loc = nextLoc;
			wea = nextWea;

			locHist(t+1)	= loc;
			weaHist(t+1)	= wea;
			staHist(t+1)	= sta;
			netHist(t+1)	= net;
		end

		% Step 3I: Write agentIdx trajectories into the master matrices
		locationTraj(agentIdx, :)	= locHist;
		wealthTraj(agentIdx, :)	= weaHist;
		stateTraj(agentIdx, :)	= staHist;
		networkTraj(agentIdx, :)	= netHist;
	end

%% 4. Aggregate location histories into shares
	M_history	= zeros(numLocations, T);
	MIN_history	= zeros(numLocations, T);

	for t = 1:T
		locs	= locationTraj(:, t);
		nets	= networkTraj(:, t);

		% accumarray returns an [N x 1] vector counting agents in each location;
		% dividing by Nagents converts counts into shares for both aggregates.
		M_history(:, t)		= accumarray(locs, 1, [numLocations, 1]) / numAgents;
		MIN_history(:, t)	= accumarray(locs(nets == 1), 1, [numLocations, 1]) / numAgents;
	end

%% 5. Output full agent trajectories for downstream analysis
	agentData.location	= locationTraj;
	agentData.wealth	= wealthTraj;
	agentData.state		= stateTraj;
	agentData.network	= networkTraj;
end
