
%% Main Script: Solve the Steady-State and Transition Dynamics of the Model
% This script performs the following:
%   1. Initializes model parameters, grids, and functional forms.
%   2. Solves for the steady-state value and policy functions assuming no help (G = G0).
%   3. Solves the dynamic equilibrium through backward induction and forward simulation.
%
% AUTHOR: Agustin Deambrosi
% DATE: April 2025
% VERSION: 2.1
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

wel = computeWalfare(params, vf_path_eqm, agentData, settings.T_tilde);
wel = wel/settings.Nagents;

%% 6. Transportation Aid Counterfactual
fprintf('\nEvaluating transportation-aid counterfactual...\n');

transportAid = struct();
transportAid.type            = 'transport';
transportAid.startPeriod     = 1;                               % Aid becomes available from period 5 onwards
transportAid.wealthThreshold = 13;                              % Eligible if asset index ≤ 4
transportAid.massIncrease    = 0.90* ones(dims.N, 1);           % Additional migrant mass outside destination 2
transportAid.massIncrease(2) = 0;                               % No artificial mass for location 2
transportAid.massIncrease(1) = 0;                               % No artificial mass for location 2
transportAid.budget          = 3000;                            % Total funds E for the program

try
    [pol_transport, M_transport, it_transport, vf_transport] = solveDynamicEquilibrium( ...
        M0, vf_nh, m0, dims, params, grids, indexes, matrices, settings, transportAid);
    fprintf('Transportation-aid equilibrium solved in %d iterations.\n', it_transport);
catch ME
    error('Error solving transport-aid counterfactual: %s', ME.message);
end

G_transport_base = zeros(dims.H, settings.T);
G_transport_aug  = zeros(dims.H, settings.T);
for t = 1:settings.T
    G_transport_base(:, t) = computeG(M_transport(:, t), params.ggamma);
    if t >= transportAid.startPeriod
        G_transport_aug(:, t) = computeG(M_transport(:, t) + transportAid.massIncrease, params.ggamma);
    else
        G_transport_aug(:, t) = G_transport_base(:, t);
    end
end

[M_total_transport, M_network_transport, agentDataTransport, statsTransport] = ...
    simulateAgentsTransportAid(m0, pol_transport, G_transport_base, G_transport_aug, ...
    dims, params, grids, settings, transportAid);

plotOutcomeCase(M_total_transport, M_network_transport, agentDataTransport, ...
    dims, settings, 'transport_aid');

fprintf('  Accepted moves with aid: %d\n', statsTransport.acceptedMoves);
fprintf('  Aid spent: %.2f (budget remaining: %.2f)\n', ...
    statsTransport.totalAidSpent, statsTransport.finalBudget);



wel_transport = computeWalfare(params, vf_transport, agentDataTransport, settings.T_tilde);
wel_transport = wel_transport/settings.Nagents;




%% 7. Food-and-Shelter Aid Counterfactual
fprintf('\nEvaluating food-and-shelter aid counterfactual...\n');

shelterAid = struct();
shelterAid.type             = 'shelter';
shelterAid.startPeriod      = 1;             % Transfers available from period 4
shelterAid.wealthThreshold  = 10;             % Eligible if asset index ≤ 5
shelterAid.transferAmount   = 1.2;          % Wealth transfer when aid is granted
shelterAid.grantProbability = 0.90;          % Probability of receiving a transfer
shelterAid.budget           = 3000;           % Total funds for the program

try
    [pol_shelter, M_shelter, it_shelter, vf_shelter] = solveDynamicEquilibrium( ...
        M0, vf_nh, m0, dims, params, grids, indexes, matrices, settings, shelterAid);
    fprintf('Shelter-aid equilibrium solved in %d iterations.\n', it_shelter);
catch ME
    error('Error solving shelter-aid counterfactual: %s', ME.message);
end

G_shelter = zeros(dims.H, settings.T);
for t = 1:settings.T
    G_shelter(:, t) = computeG(M_shelter(:, t), params.ggamma);
end

[M_total_shelter, M_network_shelter, agentDataShelter, statsShelter] = ...
    simulateAgentsShelterAid(m0, pol_shelter, G_shelter, dims, params, grids, settings, shelterAid);

plotOutcomeCase(M_total_shelter, M_network_shelter, agentDataShelter, ...
    dims, settings, 'shelter_aid');

fprintf('  Transfers granted: %d\n', statsShelter.transfersGranted);
fprintf('  Aid spent: %.2f (budget remaining: %.2f)\n', ...
    statsShelter.totalAidSpent, statsShelter.finalBudget);



wel_shelter = computeWalfare(params, vf_shelter, agentDataShelter, settings.T_tilde);
wel_shelter = wel_shelter/settings.Nagents;

%% 8. later Transportation Aid Counterfactual
fprintf('\nEvaluating Later transportation-aid counterfactual...\n');

transportAid = struct();
transportAid.type            = 'transport';
transportAid.startPeriod     = 5;                               % Aid becomes available from period 5 onwards
transportAid.wealthThreshold = 13;                              % Eligible if asset index ≤ 4
transportAid.massIncrease    = 0.98* ones(dims.N, 1);           % Additional migrant mass outside destination 2
transportAid.massIncrease(2) = 0;                               % No artificial mass for location 2
transportAid.massIncrease(1) = 0;                               % No artificial mass for location 1
transportAid.budget          = 3000;                            % Total funds E for the program

try
    [pol_lateTransport, M_lateTransport, it_lateTransport, vf_lateTransport] = solveDynamicEquilibrium( ...
        M0, vf_nh, m0, dims, params, grids, indexes, matrices, settings, transportAid);
    fprintf('Transportation-aid equilibrium solved in %d iterations.\n', it_transport);
catch ME
    error('Error solving transport-aid counterfactual: %s', ME.message);
end

G_transport_base = zeros(dims.H, settings.T);
G_transport_aug  = zeros(dims.H, settings.T);
for t = 1:settings.T
    G_transport_base(:, t) = computeG(M_lateTransport(:, t), params.ggamma);
    if t >= transportAid.startPeriod
        G_transport_aug(:, t) = computeG(M_lateTransport(:, t) + transportAid.massIncrease, params.ggamma);
    else
        G_transport_aug(:, t) = G_transport_base(:, t);
    end
end

[M_total_lateTransport, M_network_lateTransport, agentDataLateTransport, statsLateTransport] = ...
    simulateAgentsTransportAid(m0, pol_lateTransport, G_transport_base, G_transport_aug, ...
    dims, params, grids, settings, transportAid);

plotOutcomeCase(M_total_lateTransport, M_network_lateTransport, agentDataLateTransport, ...
    dims, settings, 'Late_transport_aid');

fprintf('  Accepted moves with aid: %d\n', statsLateTransport.acceptedMoves);
fprintf('  Aid spent: %.2f (budget remaining: %.2f)\n', ...
    statsLateTransport.totalAidSpent, statsLateTransport.finalBudget);



wel_lateTransport = computeWalfare(params, vf_lateTransport, agentDataLateTransport, settings.T_tilde);
wel_lateTransport = wel_lateTransport/settings.Nagents;


%% 9. Later Food-and-Shelter Aid Counterfactual
fprintf('\nEvaluating food-and-shelter aid counterfactual...\n');

shelterAid = struct();
shelterAid.type             = 'shelter';
shelterAid.startPeriod      = 5;            % Transfers available from period 4
shelterAid.wealthThreshold  = 10;           % Eligible if asset index ≤ 5
shelterAid.transferAmount   = 1.2;          % Wealth transfer when aid is granted
shelterAid.grantProbability = 0.90;         % Probability of receiving a transfer
shelterAid.budget           = 3000;         % Total funds for the program

try
    [pol_shelter, M_shelter, it_shelter, vf_shelter] = solveDynamicEquilibrium( ...
        M0, vf_nh, m0, dims, params, grids, indexes, matrices, settings, shelterAid);
    fprintf('Shelter-aid equilibrium solved in %d iterations.\n', it_shelter);
catch ME
    error('Error solving shelter-aid counterfactual: %s', ME.message);
end

G_shelter = zeros(dims.H, settings.T);
for t = 1:settings.T
    G_shelter(:, t) = computeG(M_shelter(:, t), params.ggamma);
end

[M_total_shelter, M_network_shelter, agentDataShelter, statsShelter] = ...
    simulateAgentsShelterAid(m0, pol_shelter, G_shelter, dims, params, grids, settings, shelterAid);

plotOutcomeCase(M_total_shelter, M_network_shelter, agentDataShelter, ...
    dims, settings, 'shelter_aid');

fprintf('  Transfers granted: %d\n', statsShelter.transfersGranted);
fprintf('  Aid spent: %.2f (budget remaining: %.2f)\n', ...
    statsShelter.totalAidSpent, statsShelter.finalBudget);



wel_shelter = computeWalfare(params, vf_shelter, agentDataShelter, settings.T_tilde);
wel_shelter = wel_shelter/settings.Nagents;
%% 9. Report total runtime
elapsedTime = toc;
fprintf('\nFull script completed in %.2f seconds.\n', elapsedTime);
