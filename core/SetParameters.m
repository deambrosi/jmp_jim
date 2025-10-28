function params = SetParameters(dims, overrides)
% SETPARAMETERS Initializes all structural parameters used in the model.
%
%   Sets calibrated parameters for preferences, location characteristics,
%   migration costs, network effects, and stochastic transitions.
%
% INPUTS:
%   dims      - Struct with model dimensions (from setDimensionParam)
%   overrides - (Optional) Struct with parameter values to override defaults
%
% OUTPUT:
%   params - Struct of model parameters:
%       Preferences:
%           .bbeta   - Discount factor
%           .ssigma  - Risk aversion coefficient
%           .CONS    - Scaling constant for value functions
%       
%       Grid bounds:
%           .lb_a, .up_a, .ca     - Asset grid bounds and curvature
%           .lb_am, .ub_am        - Amenity bounds
%           .lb_pr, .ub_pr        - Productivity bounds
%       
%       Location-specific [N×1 vectors]:
%           .A       - Productivity levels by location
%           .bbi     - Unemployment benefits by location
%           .B       - Amenity levels by location
%           .f0, .f1 - Job-finding probability bounds by location
%           .g0, .g1 - Job-separation probability bounds by location
%       
%       Migration costs:
%           .tilde_ttau - Transport costs between adjacent locations [N-1×1]
%           .hat_ttau   - Fixed costs of entering each location [N×1]
%           .ttau       - Full migration cost tensor [N×N×H]
%           .aalpha     - Help discount factor (0 < alpha < 1)
%           .nnu        - Scale of idiosyncratic location preference shocks
%       
%       Network parameters:
%           .ggamma  - Elasticity of help probability w.r.t. migrant stock
%           .cchi    - Probability of losing network ties outside Venezuela
%           .G0      - Initial help distribution (when M = 0) [H×1]
%       
%       Transitions:
%           .Pb      - Integration transition matrix [B×B×N]
%           .f, .g   - Job-finding/separation by integration [B×N]
%           .P       - Joint transition matrix (employment, psi) [S×S×N]
%
% AUTHOR: Agustin Deambrosi
% DATE: October 2025
% =========================================================================

    %% Handle optional overrides
    if nargin < 2 || isempty(overrides)
        overrides = struct();
    end

    %% Preferences
    params.bbeta    = 0.996315;         % Discount factor
    params.ssigma   = 2.00;             % Risk aversion
    params.CONS     = 1e2;              % Value function scaling constant

    %% Grid construction bounds
    
    % Asset grid bounds and curvature
    params.lb_a     = 0;                % Lower bound of assets
    params.up_a     = 40;               % Upper bound of assets
    params.ca       = 3;                % Curvature for coarse grid
    
    % Amenity bounds (for utility scaling)
    params.lb_am    = 0.3;              % Lower bound on amenity multiplier
    params.ub_am    = 5;              % Upper bound on amenity multiplier
    
    % Productivity bounds (for wage calculation)
    params.lb_pr    = 1.2;              % Lower bound on productivity
    params.ub_pr    = 2.9;                % Upper bound on productivity

    %% Location-specific features [N×1 vectors]
    
    params.A        = [0.5; 1.2; 3.8; 6.5]; % [N×1] Productivity by location
    params.bbi      = 0.5 * params.A;   % [N×1] Unemployment benefits (80% of productivity)
    params.B        = [1.5; 1.4; 1.0; 0.85]; % [N×1] Amenity levels
    
    % Job-finding probability bounds by location
    params.f0       = [0.8; 0.6; 0.60; 0.60];  % [N×1] Minimum job-finding probability
    params.f1       = [0.9; 0.7; 0.95; 0.95];  % [N×1] Maximum job-finding probability
    
    % Job-separation probability bounds by location
    params.g0       = [0.05; 0.03; 0.02; 0.02]; % [N×1] Minimum separation probability
    params.g1       = [0.05; 0.03; 0.02; 0.02]; % [N×1] Maximum separation probability

    %% Migration cost primitives
    
    params.tilde_ttau = [2; 3.5; 12];         % [N-1×1] Transport costs between adjacent locations
    params.hat_ttau   = [3; 0.5; 3.5; 6];     % [N×1] Fixed entry costs by destination
    params.aalpha     = 0.50;               % Help discount factor
    params.nnu        = 0.1;                % Scale of idiosyncratic location shocks

    %% Network parameters
    
    params.ggamma   = 0.60;             % Elasticity of help probability w.r.t. stock
    params.cchi     = 0.22;             % Probability of losing network ties

    %% Markov transition parameters
    
    params.baseUp_psi = 0.08;           % Base upward transition probability for integration
    topDiff           = 1;              % Difficulty parameter for reaching high integration

    %% Apply user-provided overrides
    overrideFields = fieldnames(overrides);
    for i = 1:numel(overrideFields)
        field = overrideFields{i};
        params.(field) = overrides.(field);
    end

    %% Ensure location parameters are column vectors [N×1]
    params.A          = params.A(:);
    params.B          = params.B(:);
    params.tilde_ttau = params.tilde_ttau(:);
    params.hat_ttau   = params.hat_ttau(:);

    %% Build migration cost tensor [N×N×H]
    
    % Construct base migration cost matrix [N×N]
    % tauBase(i,j) = cost to move from location i to location j
    tauBase = zeros(dims.N);
    for ii = 1:dims.N
        for jj = ii+1:dims.N
            % Transport cost = sum of costs along the corridor
            transport = sum(params.tilde_ttau(ii:jj-1));
            
            % Total cost = transport + fixed entry cost
            tauBase(ii, jj) = params.hat_ttau(jj) + transport;
            tauBase(jj, ii) = params.hat_ttau(ii) + transport;
        end
    end
    
    % Expand to [N×N×H] tensor with help discounting
    params.ttau = getMigrationCostsWithHelp(tauBase, params.aalpha);

    %% Build integration transition matrix [B×B×N]
    
    % Single location transition matrix [B×B]
    params.Pb = buildMarkovMatrixDifficultAtTop(dims.B, params.baseUp_psi, 0, topDiff);
    
    % Replicate across all locations [B×B×N]
    params.Pb = repmat(params.Pb, 1, 1, dims.N);

    %% Build employment transition probabilities [B×N]
    
    % Job-finding probability increases linearly with integration level
    params.f = (params.f0 + (params.f1 - params.f0) .* linspace(0, 1, dims.B))'; % [B×N]
    
    % Job-separation probability (constant across integration levels here)
    params.g = (params.g0 + (params.g1 - params.g0) .* linspace(0, 1, dims.B))'; % [B×N]

    %% Build joint transition matrix [S×S×N]
    
    % Combines employment transitions (f, g) with integration transitions (Pb)
    % S = 2*B joint states (employment × integration)
    params.P = build_joint_transition(params, dims);

    %% Initialize help distribution when network is empty [H×1]
    
    % G0 = probability distribution over help vectors when M = 0
    params.G0 = computeG(zeros(dims.N, 1), params.ggamma);

end