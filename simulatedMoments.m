function [moments, M] = simulatedMoments(dims, params, grids, indexes, settings, m0)
% SIMULATEDMOMENTS Compute simulated moments for the benchmark equilibrium.
%
%   moments = SimulatedMoments(dims, params)
%
%   The routine reproduces the first five steps of Main.m but stops right
%   after simulating the benchmark equilibrium.  It solves the no-help
%   problem, computes the transitional dynamics, runs a detailed agent
%   simulation, and aggregates the statistics needed for moment matching.
%
%   INPUTS
%       dims    : structure created by setDimensionParam
%       params  : structure created by SetParameters
%
%   OUTPUT
%       moments : structure with the simulated moments described in the
%                 documentation of the project
%
%   The function leaves counterfactual analysis to the calling context so it
%   can be reused for estimation routines.
%
    % === Step 1. Prepare grids, matrices, and simulation settings ==========
    matrices         = constructMatrix(dims, params, grids, indexes);


    % === Step 2. Solve the no-help (G = G0) problem ========================
    [vf_noHelp, ~] = noHelpEqm(dims, params, grids, indexes, matrices, settings);

    % === Step 3. Initialize the guess for the network distribution =========
    M_init          = zeros(dims.N, 1);
    M_init(1)       = 1;
    M0              = repmat(M_init, 1, settings.T);

    % === Step 4. Transition equilibrium ===================================
    [pol_eqm, M_eqm] = solveDynamicEquilibrium(M0, vf_noHelp, m0, ...
        dims, params, grids, indexes, matrices, settings, []);

    % === Step 5. Benchmark simulation =====================================
    G_path = zeros(dims.H, settings.T);
    for t = 1:settings.T
        G_path(:, t) = computeG(M_eqm(:, t), params.ggamma);
    end

    [M, ~, agentData] = detailedSimulateAgents(m0, pol_eqm, G_path, ...
        dims, params, grids, settings);

    % Aggregate the requested moments
    moments = computeSimulationMoments(agentData, dims, settings);
end