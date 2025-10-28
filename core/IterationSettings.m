function settings = IterationSettings()
% ITERATIONSETTINGS Configure algorithmic controls for the dynamic equilibrium solver.
%
%   OUTPUT:
%       settings    Struct scalar gathering iteration and simulation controls:
%           .it        Scalar double. Running counter for the outer loop, seeded at 0.
%           .diffV     Scalar double. Initial sup-norm gap across the value function.
%           .tolV      Scalar double. Convergence tolerance for value-function updates.
%           .tolM      Scalar double. Convergence tolerance for migration probabilities.
%           .MaxItV    Scalar double. Upper bound on value-function iterations per outer loop.
%           .MaxItJ    Scalar double. Upper bound on marriage-market iterations per loop.
%           .MaxIter   Scalar double. Maximum admissible iterations for the master routine.
%           .Nagents   Scalar double. Number of simulated agents replicated in Monte Carlo.
%           .T         Scalar double. Total simulated periods in the agent panel.
%           .burn      Scalar double. Observations discarded as burn-in before computing moments.
%
%   The struct is consumed by routines across /utils and /core to orchestrate iterative
%   convergence in the dynamic equilibrium algorithm.
%
%   AUTHOR: Agustin Deambrosi
%   DATE: October 2025
% =========================================================================

	%% Step 1: Initialize counters that track outer-iteration progress
	settings.it		= 0;      % Scalar; iteration index k = 0 before updates start
	settings.diffV	= 1;      % Scalar; initial sup-norm gap normalised to unity

	%% Step 2: Set convergence tolerances for value and migration dynamics
	settings.tolV	= 0.5;    % Scalar; admissible sup-norm error for value iteration
	settings.tolM	= 1e-2;   % Scalar; admissible sup-norm deviation for migration distribution

	%% Step 3: Cap inner-loop iterations to guarantee termination
	settings.MaxItV	= 40;    % Scalar; max Bellman iterations per outer step
	settings.MaxItJ	= 2;     % Scalar; max marriage-market iterations per outer step
	settings.MaxIter	= 100;   % Scalar; global cap on the outer fixed-point routine

	%% Step 4: Configure Monte Carlo simulation dimensions
	settings.Nagents	= 5000; % Scalar; number of agents simulated in the panel
	settings.T		= 100;   % Scalar; total simulated periods per agent trajectory
	settings.burn	= 50;    % Scalar; periods dropped so remaining panel has length T - burn

end
