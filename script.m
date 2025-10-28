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


dims                = setDimensionParam();                            % Dimensions (N, K, B, etc.)
params              = SetParameters(dims);                            % Structural parameters
[grids, indexes]    = setGridsAndIndices(dims);                       % Grids and index matrices
settings            = IterationSettings();                            % Iteration controls and simulation length
m0                  = createInitialDistribution(dims, settings);      % Initial agent distribution




tic;
[moments, M] = simulatedMoments(dims, params, grids, indexes, settings, m0);
t=toc

% 1. Plot aggregate location shares

fig = figure('Visible','on');
plot(M', 'LineWidth', 2);
title(sprintf('Share of Agents'));
xlabel('Time Period'); ylabel('Share of All Agents');
legend(arrayfun(@(i) sprintf('Location %d', i), 1:dims.N, 'UniformOutput', false));
grid on;