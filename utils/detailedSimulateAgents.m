function [M_history, MIN_history, agentData] = detailedSimulateAgents( ...
        m0, pol, G_dist, dims, params, grids, settings)
% DETAILEDSIMULATEAGENTS Simulate agents while recording additional statistics.
%
%   This version mirrors simulateAgents but keeps richer information that is
%   useful for moment computation.  In addition to locations, wealth, states,
%   and network status, it records per period:
%       - Realized income
%       - Whether migration help was used in the move executed that period
%       - Whether the agent reached the current destination using help
%       - Whether the agent reached the current destination directly from
%         Venezuela
%
%   The calling convention matches simulateAgents for easy drop-in
%   replacement once the dynamic equilibrium has been solved.

    T = settings.T;

    isTimeInvariant = ~iscell(pol.a);
    if isTimeInvariant
        pol.a   = repmat({pol.a},  T-1, 1);
        pol.an  = repmat({pol.an}, T-1, 1);
        pol.mu  = repmat({pol.mu}, T-1, 1);
        pol.mun = repmat({pol.mun},T-1, 1);
    end

    numAgents    = settings.Nagents;
    numLocations = dims.N;

    locationTraj            = zeros(numAgents, T);
    wealthTraj              = zeros(numAgents, T);
    stateTraj               = zeros(numAgents, T);
    networkTraj             = zeros(numAgents, T);
    incomeTraj              = zeros(numAgents, T);
    helpArrivalTraj         = zeros(numAgents, T);
    arrivalWithHelpTraj     = zeros(numAgents, T);
    arrivalFromVenTraj      = zeros(numAgents, T);

    helpMatrix = dec2bin(0:(dims.H - 1)) - '0';

    parfor agentIdx = 1:numAgents
        agent = m0(agentIdx);

        locHist      = zeros(1, T);
        weaHist      = zeros(1, T);
        staHist      = zeros(1, T);
        netHist      = zeros(1, T);
        incHist      = zeros(1, T);
        helpHist     = zeros(1, T);
        arrHelpHist  = zeros(1, T);
        arrFromHist  = zeros(1, T);

        loc = agent.location;
        wea = agent.wealth;
        sta = agent.state;
        net = agent.network;

        locHist(1)  = loc;
        weaHist(1)  = wea;
        staHist(1)  = sta;
        netHist(1)  = net;
        incHist(1)  = computeIncomeFromState(loc, sta, params, grids, dims);
        helpHist(1)    = 0;
        arrHelpHist(1) = 0;
        arrFromHist(1) = 0;

        for t = 1:(T-1)
            % Asset policy
            if net == 1
                a_fine = pol.an{t}(sta, wea, loc);
            else
                a_fine = pol.a{t}(sta, wea, loc);
            end
            a_fine = min(max(a_fine, 1), length(grids.ahgrid));
            [~, nextWea] = min(abs(grids.agrid - grids.ahgrid(a_fine)));

            % Migration decision
            if net == 1
                G_t     = G_dist(:, t);
                h_idx   = find(cumsum(G_t) >= rand(), 1);
                migProb = squeeze(pol.mun{t}(sta, wea, loc, :, h_idx));
            else
                migProb = squeeze(pol.mu{t}(sta, wea, loc, :));
                h_idx   = 1;
            end
            migProb = migProb / sum(migProb);
            nextLoc = find(cumsum(migProb) >= rand(), 1);
            if isempty(nextLoc)
                [~, nextLoc] = max(migProb);
            end

            helpVec   = helpMatrix(h_idx, :);
            usedHelp  = (net == 1) && (helpVec(nextLoc) == 1);
            cameFromV = (loc == 1);

            % Wealth and state transitions
            if nextLoc ~= loc
                migCost     = params.ttau(loc, nextLoc, h_idx);
                newAssets   = grids.agrid(nextWea) - migCost;
                [~, nextWea] = min(abs(grids.agrid - newAssets));
                sta          = 1;  % Reset state after moving
            else
                transCdf     = cumsum(params.P(:, sta, loc));
                sta          = find(transCdf >= rand(), 1);
            end

            % Network status
            if net == 1 && nextLoc ~= 1
                if rand() < params.cchi
                    net = 0;
                end
            end

            prevLoc = loc;
            loc     = nextLoc;
            wea     = nextWea;

            if loc ~= prevLoc
                arrivalHelp = usedHelp;
                arrivalFrom = cameFromV;
            else
                arrivalHelp = arrHelpHist(t);
                arrivalFrom = arrFromHist(t);
            end

            locHist(t+1)      = loc;
            weaHist(t+1)      = wea;
            staHist(t+1)      = sta;
            netHist(t+1)      = net;
            incHist(t+1)      = computeIncomeFromState(loc, sta, params, grids, dims);
            helpHist(t+1)     = usedHelp;
            arrHelpHist(t+1)  = arrivalHelp;
            arrFromHist(t+1)  = arrivalFrom;
        end

        locationTraj(agentIdx, :)        = locHist;
        wealthTraj(agentIdx, :)          = weaHist;
        stateTraj(agentIdx, :)           = staHist;
        networkTraj(agentIdx, :)         = netHist;
        incomeTraj(agentIdx, :)          = incHist;
        helpArrivalTraj(agentIdx, :)     = helpHist;
        arrivalWithHelpTraj(agentIdx, :) = arrHelpHist;
        arrivalFromVenTraj(agentIdx, :)  = arrFromHist;
    end

    M_history   = zeros(numLocations, T);
    MIN_history = zeros(numLocations, T);

    for t = 1:T
        locs = locationTraj(:, t);
        nets = networkTraj(:, t);

        M_history(:, t)   = accumarray(locs, 1, [numLocations, 1]) / numAgents;
        MIN_history(:, t) = accumarray(locs(nets == 1), 1, [numLocations, 1]) / numAgents;
    end

    agentData.location           = locationTraj;
    agentData.wealth             = wealthTraj;
    agentData.state              = stateTraj;
    agentData.network            = networkTraj;
    agentData.income             = incomeTraj;
    agentData.helpArrival        = helpArrivalTraj;
    agentData.arrivalWithHelp    = arrivalWithHelpTraj;
    agentData.arrivalFromVenezuela = arrivalFromVenTraj;
end

