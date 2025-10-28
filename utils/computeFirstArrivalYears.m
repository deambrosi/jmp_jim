function arrivalYears = computeFirstArrivalYears(locationPath, destinations)
    % Compute the first arrival year (in model years) for each agent/location.

    numAgents    = size(locationPath, 1);
    totalPeriods = size(locationPath, 2);
    arrivalYears = NaN(numAgents, numel(destinations));

    for locIdx = 1:numel(destinations)
        loc = destinations(locIdx);

        arrivals = (locationPath(:, 2:totalPeriods) == loc) & ...
                   (locationPath(:, 1:(totalPeriods - 1)) ~= loc);

        for agentIdx = 1:numAgents
            % First arrival identifies the cohort year for this destination
            firstArrival = find(arrivals(agentIdx, :), 1, 'first');
            if ~isempty(firstArrival)
                arrivalYears(agentIdx, locIdx) = ceil((firstArrival + 1) / 2);
            end
        end
    end
end