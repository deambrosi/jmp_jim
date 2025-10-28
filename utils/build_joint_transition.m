function T = build_joint_transition(params, dims)
% BUILD_JOINT_TRANSITION Constructs joint transition matrix over employment and integration.
%
%   Builds T for joint state s = (e, psi), e in {1=U, 2=E}, psi in {1,...,B}.
%   Columns are current states, rows are next states. Each T(:,:,n) is column-stochastic.
%
% INPUTS:
%   params.Pb - [B x B x N] integration transitions by location
%               Orientation may be row- or column-stochastic. Auto-detected.
%   params.f  - [B x N] job-finding probabilities when unemployed, by (psi, n)
%   params.g  - [B x N] job-separation probabilities when employed, by (psi, n)
%   dims.B    - number of integration states
%   dims.N    - number of locations
%   dims.S    - number of joint states, must equal 2*B
%
% OUTPUT:
%   T         - [S x S x N] joint transition over s = (e, psi)
%               Block structure with columns = current, rows = next:
%                   T = [ UU  EU
%                         UE  EE ]
%               where UU: U->U, UE: U->E, EU: E->U, EE: E->E
%
% AUTHOR: Agustin Deambrosi
% DATE: October 2025
% =========================================================================

    % Extract and validate sizes
    B = dims.B;
    N = dims.N;
    S = dims.S;
    if S ~= 2*B
        error('dims.S must equal 2*dims.B. Got dims.S=%d, dims.B=%d.', S, B);
    end

    Pb = params.Pb;   % [B x B x N]
    f  = params.f;    % [B x N]
    g  = params.g;    % [B x N]

    if ~isequal(size(Pb), [B, B, N])
        error('params.Pb must be [B x B x N], got %s.', mat2str(size(Pb)));
    end
    if ~isequal(size(f), [B, N])
        error('params.f must be [B x N], got %s.', mat2str(size(f)));
    end
    if ~isequal(size(g), [B, N])
        error('params.g must be [B x N], got %s.', mat2str(size(g)));
    end

    % Preallocate
    T = zeros(2*B, 2*B, N);

    % Tolerance for stochasticity checks and normalization
    tol = 1e-12;

    for n = 1:N
        Ppsi = Pb(:, :, n);   % integration transitions at location n

        % Detect whether Ppsi is row- or column-stochastic
        col_err = max(abs(sum(Ppsi, 1) - 1));
        row_err = max(abs(sum(Ppsi, 2) - 1));

        % We want a matrix whose columns index CURRENT psi and rows index NEXT psi.
        % If Ppsi is row-stochastic with rows = current, columns = next,
        % then we should use Puse = Ppsi' so that columns = current.
        % If Ppsi is column-stochastic already, we can use it directly.
        if row_err <= col_err
            % Treat as row-stochastic: rows sum to 1
            Puse = Ppsi.';   % now columns = current psi, rows = next psi
        else
            % Treat as column-stochastic: columns sum to 1
            Puse = Ppsi;     % columns = current psi, rows = next psi
        end

        % Extract employment transition probabilities by current psi
        p = f(:, n);         % job-finding when unemployed
        q = g(:, n);         % job-separation when employed

        % Sanity bound to [0,1]
        p = min(max(p, 0), 1);
        q = min(max(q, 0), 1);

        % Build the four BxB blocks by scaling columns of Puse
        % Column j corresponds to current psi = j
        UU = Puse * diag(1 - p);   % U->U
        UE = Puse * diag(p);       % U->E
        EU = Puse * diag(q);       % E->U
        EE = Puse * diag(1 - q);   % E->E

        % Assemble the SxS matrix with rows = next states, cols = current states
        T(:, :, n) = [UU, EU;
                      UE, EE];

        % Numerical cleanup: clip tiny negatives and renormalize columns
        T(:, :, n) = max(T(:, :, n), 0);
        colSums = sum(T(:, :, n), 1);
        need_fix = abs(colSums - 1) > tol;
        if any(need_fix)
            T(:, need_fix, n) = bsxfun(@rdivide, T(:, need_fix, n), colSums(need_fix));
        end
    end
end
