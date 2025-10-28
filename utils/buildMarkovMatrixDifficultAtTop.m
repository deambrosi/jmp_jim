function P = buildMarkovMatrixDifficultAtTop(K, baseUp, shockProb, topDifficulty)
% Creates a KxK transition matrix in which:
%  - Probability of moving "down" from interior states: shockProb
%  - Probability of moving "up" from interior states: upProb(i), 
%         which starts near baseUp when i=1 and decreases
%         with i according to topDifficulty.
%  - Probability of staying put: 1 - upProb(i) - shockProb (for interior states)
%
% The function upProb(i) can be anything you like, but here we define a 
% simple linear decrease with i, such that:
%    upProb(1)   = baseUp
%    upProb(K-1) = max(0, baseUp * (1 - topDifficulty))   (roughly)
%    upProb(K)   = 0 (can't go higher than K)
% If topDifficulty=0, upProb(i) ~ baseUp for all i, 
% meaning the "very high" states are not that hard to reach.
%
% Boundary conditions:
%   - If i=1 (lowest state), can't move down => P(1,1)=1 - upProb(1), P(1,2)=upProb(1).
%   - If i=K (highest state), can't move up => P(K,K)=1 - shockProb, P(K,K-1)=shockProb.
%   - For 2<=i<=K-1, 
%        P(i,i-1) = shockProb
%        P(i,i+1) = upProb(i)
%        P(i,i)   = 1 - shockProb - upProb(i).

    if baseUp + shockProb > 1
        error('baseUp + shockProb must be <= 1.0 to keep probabilities valid.');
    end

    % Preallocate
    P = zeros(K,K);

    % We'll define a small helper function for the "state-dependent" up probability:
    function pU = upProb(i)
        % i in {1,2,...,K}
        % We want i=K => upProb=0 always.
        if i >= K
            pU = 0;
            return;
        end

        % One simple approach: 
        % upProb(i) = baseUp * [1 - topDifficulty*(i-1)/(K-1)]
        % so at i=1 => upProb(1)= baseUp * [1 - 0] = baseUp
        %    i=K-1 => upProb(K-1)= baseUp * [1 - topDifficulty*(K-2)/(K-1)]
        frac = (i-1)/(K-1);
        factor = 1 - topDifficulty*frac;  % linearly reduces from 1 to (1-topDifficulty)
        if factor < 0
            factor = 0;  % don't allow negative probabilities
        end
        pU = baseUp * factor;
    end

    % Fill the matrix row by row
    for i = 1:K

        if i == 1
            % Lowest state: no down
            p_up   = upProb(i);   % from i=1
            P(i,i)   = 1 - p_up;
            if K>1
                P(i,i+1) = p_up;
            end

        elseif i == K
            % Highest state: no up
            % Probability of going down = shockProb
            P(i,i)   = 1 - shockProb;
            P(i,i-1) = shockProb;

        else
            % Interior state
            p_up      = upProb(i);   % depends on i
            % "down" = shockProb
            % "up"   = p_up
            % "stay" = 1 - shockProb - p_up
            P(i,i-1) = shockProb;
            P(i,i+1) = p_up;
            stayProb = 1 - shockProb - p_up;
            if stayProb < 0
                % In case topDifficulty made p_up large, or we had large shockProb
                error('Markov probabilities < 0 at state %d. Adjust your parameters.', i);
            end
            P(i,i) = stayProb;
        end
    end

    % You can print or check your matrix if you like:
    % disp('Constructed Markov Transition Matrix P:'); disp(P);

end
