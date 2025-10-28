function dims = setDimensionParam()
% SETDIMENSIONPARAM Initializes model dimension parameters.
%
%   Defines the core dimensionality of the state space used throughout
%   the dynamic migration equilibrium model.
%
% OUTPUT:
%   dims - Structure containing model dimensions:
%       .N  - Number of locations (including Venezuela)
%       .B  - Number of integration states (psi levels)
%       .S  - Number of joint states (employment × integration)
%       .H  - Number of possible help offer vectors (2^N)
%       .Na - Coarse asset grid points (for value iteration)
%       .na - Fine asset grid points (for policy optimization)
%
% AUTHOR: Agustin Deambrosi
% DATE: October 2025
% =========================================================================

    %% Define model dimensions
    
    dims.N      = 4;                % Number of locations
    dims.B      = 6;                % Integration states
    dims.S      = 2 * dims.B;       % Joint states: (employment, psi)
    dims.H      = 2^dims.N;         % Help offer combinations
    dims.Na     = 15;               % Coarse asset grid
    dims.na     = 5000;             % Fine asset grid

end