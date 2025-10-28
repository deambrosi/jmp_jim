%% Main Script: Solve the Steady-State and Transition Dynamics of the Model
% This script performs the following:
%   1. Initializes model parameters, grids, and functional forms.
%   2. Solves for the steady-state value and policy functions assuming no help (G = G0).
%   3. Solves the dynamic equilibrium through backward induction and forward simulation.
%
% AUTHOR: Agustin Deambrosi
% DATE: April 2025
% =========================================================================

clc; clear; close all;

%% 1. Initialize Model Parameters, Grids, and Settings
fprintf('Initializing model parameters and grids...\n');
try
    dims                = setDimensionParam();                            % Dimensions (N, K, B, etc.)
    params              = SetParameters(dims);                            % Structural parameters
    [grids, indexes]    = setGridsAndIndices(dims);                       % Grids and index matrices
    matrices            = constructMatrix(dims, params, grids, indexes);  % Precomputed utility and wealth matrices
    settings            = IterationSettings();                            % Iteration controls and simulation length
    m0                  = createInitialDistribution(dims, settings);      % Initial agent distribution
    fprintf('Initialization completed successfully.\n');
catch ME
    error('Error during initialization: %s', ME.message);
end

tic;

%% 2. Solve No-Help Equilibrium (Steady-State)
fprintf('\nComputing No-Help Value and Policy Functions...\n');
try
    [vf_nh, pol_nh] = noHelpEqm(dims, params, grids, indexes, matrices, settings);  % Solve with G = G0
    fprintf('No-Help Value and Policy Functions found successfully.\n');
catch ME
    error('Error computing No-Help equilibrium: %s', ME.message);
end

%% 3. Initialize Guess for Dynamic Network Agent Distribution
% We assume all agents begin in location 1 and are part of the network.
M_init          = zeros(dims.N, 1);
M_init(1)       = 1;
M0              = repmat(M_init, 1, settings.T);  % [N x T] initial guess for M1

%% 4. Solve Dynamic Equilibrium (Transition Path)
fprintf('\nSolving Dynamic Equilibrium via Backward Induction and Simulation...\n');
try
    [pol_eqm, M_eqm, it_count, vf_path_eqm] = solveDynamicEquilibrium(M0, vf_nh, m0, dims, params, grids, indexes, matrices, settings, []);
    fprintf('Dynamic equilibrium solved successfully in %d iterations.\n', it_count);
catch ME
    error('Error solving dynamic equilibrium: %s', ME.message);
end

%% 5. Simulate Final Trajectories Using Converged Policy
fprintf('\nSimulating Agent Paths Using Converged Policies...\n');
try
    % Compute final G_t path from converged M_eqm
    G_dist = zeros(dims.H, settings.T);
    for t = 1:settings.T
        G_dist(:, t) = computeG(M_eqm(:, t), params.ggamma);
    end

    [M_total, M_network, agentData] = simulateAgents(m0, pol_eqm, G_dist, dims, params, grids, settings);
    fprintf('Simulation completed successfully.\n');
catch ME
    error('Error during final simulation: %s', ME.message);
end

plotOutcomeCase(M_total, M_network, agentData, dims, settings, 'benchmark');
t=toc