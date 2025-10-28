function m0 = createInitialDistribution(dims, settings)
% CREATEINITIALDISTRIBUTION Build the initial agent distribution over discrete states.
%
%   OUTPUT:
%       m0         [settings.Nagents x 1] struct array with per-agent state descriptors:
%                      .state    Integer joint index s ∈ {1,...,dims.S} combining (η, ψ).
%                      .wealth   Integer asset index on the coarse grid {1,...,dims.Na}.
%                      .location Integer location identifier, initialised at 1 for all agents.
%                      .network  Binary indicator: 1 denotes network member, 0 otherwise.
%
%   INPUTS:
%       dims       Struct of model dimensions. Required fields:
%                      .S   Total joint (η, ψ) states; determines support of state.
%                      .Na  Total wealth grid nodes; determines support of wealth.
%       settings   Struct of simulation controls. Required field:
%                      .Nagents  Scalar double; number of Monte Carlo agents.
%
%   The routine seeds each simulated agent with an independent draw over state and wealth,
%   fixes the initial location at the home community (index 1), and assigns everyone to
%   the network, providing the initial cross-sectional distribution for the dynamic model.
%
%   AUTHOR: Agustin Deambrosi
%   DATE: October 2025
% =========================================================================

	%% Step 1: Read simulation scale from settings
	numAgents	= settings.Nagents;  % Scalar; total number of simulated agents

	%% Step 2: Preallocate the agent state container with default zeros
	m0	= repmat(struct('state', 0, 'wealth', 0, 'location', 0, 'network', 0), numAgents, 1);
	% m0 is [numAgents x 1]; each struct entry stores four scalar identifiers per agent

	%% Step 3: Populate each agent with randomised state and wealth assignments
	for i = 1:numAgents
		m0(i).state	= randi(dims.S);   % Scalar in [1, dims.S]; draws joint (η, ψ) index
		m0(i).wealth	= randi(dims.Na);  % Scalar in [1, dims.Na]; selects wealth grid node
		m0(i).location	= 1;             % Scalar; everyone starts in origin location 1
		m0(i).network	= 1;             % Scalar; network membership initialised to active
	end

end
