function wel = computeWalfare(params, vf, agentData, T_tilde)
% COMPUTEWALFARE Compute total discounted welfare for a transition scenario.
%
%   wel = computeWalfare(params, vf, agentData, T_tilde)
%
%   INPUTS:
%       params     - Parameter struct containing the intertemporal discount
%                    factor params.bbeta used throughout the model.
%       vf         - Cell array of value-function structs over the transition
%                    horizon (output vf_path). Each entry stores .V and .Vn
%                    matrices of size [S x Na x N].
%       agentData  - Struct of simulated agent trajectories with fields
%                    .location, .wealth, .state, .network (each [Nagents x T]).
%       T_tilde    - Positive integer specifying the horizon over which welfare
%                    is computed (t = 1,...,T_tilde).
%
%   OUTPUT:
%       wel        - Scalar equal to the sum over agents of their discounted
%                    realized values across the horizon.
%
%   NOTE:
%       The horizon is truncated to the minimum length shared by vf and the
%       simulated trajectories if T_tilde exceeds either.

    % Determine the effective evaluation horizon given the available data.
    horizon = min([T_tilde, numel(vf), size(agentData.location, 2)]);

    % Preallocate per-agent welfare accumulator and construct discount weights.
    numAgents = size(agentData.location, 1);
    agentWelfare = zeros(numAgents, 1);
    betaWeights = params.bbeta .^ (1:horizon);

    % Loop over periods, retrieving realized values conditional on network status.
    for t = 1:horizon
        vf_t = vf{t};

        states    = agentData.state(:, t);
        wealthIdx = agentData.wealth(:, t);
        locIdx    = agentData.location(:, t);
        netMask   = agentData.network(:, t) ~= 0;

        linearIdx = sub2ind(size(vf_t.V), states, wealthIdx, locIdx);

        periodValues = vf_t.V(linearIdx);
        if any(netMask)
            periodValues(netMask) = vf_t.Vn(linearIdx(netMask));
        end

        agentWelfare = agentWelfare + betaWeights(t) .* periodValues;
    end

    % Sum across agents to obtain total welfare for the scenario.
    wel = sum(agentWelfare);
end
