function [M_history, MIN_history, agentData, stats] = simulateAgentsShelterAid(m0, pol, G_dist, dims, params, grids, settings, program)
% SIMULATEAGENTSSHELTERAID Simulate dynamics with food-and-shelter transfers.
%
%   Implements the counterfactual wealth-transfer program that targets agents
%   located in destination 2. Eligible agents receive an asset transfer with a
%   fixed probability provided that program funds remain. The simulation loops
%   over time (outer loop) and agents (inner loop) to avoid prioritising early
%   agents in the queue.
%
%   INPUTS:
%       m0        - Initial agent states (struct array)
%       pol       - Policy struct (.a, .an, .mu, .mun)
%       G_dist    - [H x T] help distribution path used for network agents
%       dims      - Dimension struct
%       params    - Parameter struct
%       grids     - Grid struct
%       settings  - Simulation settings
%       program   - Struct with fields:
%                       .type            = 'shelter'
%                       .startPeriod     - first active period
%                       .wealthThreshold - eligibility wealth index
%                       .transferAmount  - asset transfer when aid is granted
%                       .grantProbability- probability of receiving aid
%                       .budget          - total resources available
%
%   OUTPUTS:
%       M_history, MIN_history - location shares (all agents / networked)
%       agentData              - trajectories including aid receipts
%       stats                  - summary of transfers and remaining budget
%
%   AUTHOR: Agustin Deambrosi
%   DATE:   October 2025
% =========================================================================

    %% 1. Validate program fields and prepare policies
    requiredFields = {'startPeriod','wealthThreshold','transferAmount','grantProbability','budget'};
    for k = 1:numel(requiredFields)
        if ~isfield(program, requiredFields{k})
            error('Shelter aid program missing required field "%s".', requiredFields{k});
        end
    end

    if ~iscell(pol.a)
        pol.a   = repmat({pol.a}, settings.T-1, 1);
        pol.an  = repmat({pol.an}, settings.T-1, 1);
        pol.mu  = repmat({pol.mu}, settings.T-1, 1);
        pol.mun = repmat({pol.mun}, settings.T-1, 1);
    end

    %% 2. Dimensions and storage
    T            = settings.T;
    numAgents    = settings.Nagents;
    numLocations = dims.N;
    H            = size(G_dist, 1);

    locationTraj = zeros(numAgents, T);
    wealthTraj   = zeros(numAgents, T);
    stateTraj    = zeros(numAgents, T);
    networkTraj  = zeros(numAgents, T);
    helpIndexTraj= ones(numAgents, T);
    aidReceipt   = false(numAgents, T);

    for agentIdx = 1:numAgents
        locationTraj(agentIdx, 1) = m0(agentIdx).location;
        wealthTraj(agentIdx, 1)   = m0(agentIdx).wealth;
        stateTraj(agentIdx, 1)    = m0(agentIdx).state;
        networkTraj(agentIdx, 1)  = m0(agentIdx).network;
    end

    %% 3. Aid bookkeeping
    stats = struct();
    stats.initialBudget       = program.budget;
    stats.remainingBudget     = program.budget;
    stats.totalAidSpent       = 0;
    stats.transfersGranted    = 0;
    stats.aidSpendingTimeline = zeros(1, T);

    startPeriod     = max(1, min(T, program.startPeriod));
    wealthThreshold = program.wealthThreshold;
    transferAmount  = program.transferAmount;
    grantProb       = program.grantProbability;

    %% 4. Simulation loop (time outer, agents inner)
    for t = 1:(T-1)
        programActive = (t >= startPeriod) && (stats.remainingBudget > 0);
        for agentIdx = 1:numAgents
            loc = locationTraj(agentIdx, t);
            wea = wealthTraj(agentIdx, t);
            sta = stateTraj(agentIdx, t);
            net = networkTraj(agentIdx, t);

            %% A) Saving decision
            if net == 1
                a_fine = pol.an{t}(sta, wea, loc);
            else
                a_fine = pol.a{t}(sta, wea, loc);
            end
            a_fine = min(max(a_fine, 1), numel(grids.ahgrid));
            [~, nextWea] = min(abs(grids.agrid - grids.ahgrid(a_fine)));

            %% B) Potential transfer before migration choice
            aidGranted = false;
            if programActive && loc == 2 && wea <= wealthThreshold && stats.remainingBudget >= transferAmount
                if rand() < grantProb
                    newAssets = grids.agrid(nextWea) + transferAmount;
                    [~, nextWea] = min(abs(grids.agrid - newAssets));
                    stats.remainingBudget = stats.remainingBudget - transferAmount;
                    stats.totalAidSpent   = stats.totalAidSpent + transferAmount;
                    stats.transfersGranted= stats.transfersGranted + 1;
                    stats.aidSpendingTimeline(t) = stats.aidSpendingTimeline(t) + transferAmount;
                    aidGranted = true;
                end
            end

            %% C) Migration decision (baseline help probabilities)
            if net == 1
                u_help = rand();
                G_t    = G_dist(:, t);
                cumG   = cumsum(G_t);
                h_idx  = find(cumG >= u_help, 1);
                if isempty(h_idx)
                    h_idx = H;
                end
                migProb = squeeze(pol.mun{t}(sta, wea, loc, :, h_idx));
                migProb = migProb / sum(migProb);
            else
                h_idx  = 1;
                migProb = squeeze(pol.mu{t}(sta, wea, loc, :));
                migProb = migProb / sum(migProb);
            end

            u_move = rand();
            cumMig = cumsum(migProb);
            nextLoc = find(cumMig >= u_move, 1);
            if isempty(nextLoc)
                [~, nextLoc] = max(migProb);
            end

            %% D) Wealth and state transitions
            if nextLoc ~= loc
                migCost = params.ttau(loc, nextLoc, h_idx);
                newAssets = grids.agrid(nextWea) - migCost;
                [~, nextWea] = min(abs(grids.agrid - newAssets));
                sta = 1;
            else
                transCdf = cumsum(params.P(:, sta, loc));
                sta      = find(transCdf >= rand(), 1);
            end

            %% E) Network status update
            if net == 1 && nextLoc ~= 1
                if rand() < params.cchi
                    net = 0;
                end
            end

            %% F) Record next-period states
            locationTraj(agentIdx, t+1) = nextLoc;
            wealthTraj(agentIdx, t+1)   = nextWea;
            stateTraj(agentIdx, t+1)    = sta;
            networkTraj(agentIdx, t+1)  = net;
            helpIndexTraj(agentIdx, t+1)= h_idx;
            aidReceipt(agentIdx, t+1)   = aidGranted;
        end
    end

    %% 5. Aggregate location shares
    M_history   = zeros(numLocations, T);
    MIN_history = zeros(numLocations, T);
    for t = 1:T
        locs = locationTraj(:, t);
        nets = networkTraj(:, t);
        M_history(:, t)   = accumarray(locs, 1, [numLocations, 1]) / numAgents;
        MIN_history(:, t) = accumarray(locs(nets == 1), 1, [numLocations, 1]) / numAgents;
    end

    %% 6. Prepare agent-level output
    agentData.location   = locationTraj;
    agentData.wealth     = wealthTraj;
    agentData.state      = stateTraj;
    agentData.network    = networkTraj;
    agentData.helpIndex  = helpIndexTraj;
    agentData.aidReceipt = aidReceipt;

    %% 7. Finalise statistics
    stats.finalBudget = stats.remainingBudget;
end
