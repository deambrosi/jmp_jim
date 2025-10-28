function moments = computeSimulationMoments(agentData, dims, settings)
% COMPUTESIMULATIONMOMENTS  Aggregate simulated agent histories into moments.
%
%   The routine builds cross-sectional statistics by destination and year.
%   It reuses a stayer mask that removes agents who have just arrived in a
%   destination during the current period so that unemployment, income, and
%   arrival composition are all computed on a consistent sample.

    % === Step 1. Configure the horizon and storage ======================
    periodsPerYear = 2;   % Each model period represents a semester
    maxYearsUsed   = 5;   % Moments are computed for the first five years

    totalPeriodsAvailable = min(settings.T, maxYearsUsed * periodsPerYear);
    numYears       = floor(totalPeriodsAvailable / periodsPerYear);
    analysisPeriods = numYears * periodsPerYear;

    destLocations = 2:dims.N;   % Exclude Venezuela (location 1)
    numDest       = numel(destLocations);

    avgIncome     = NaN(numDest, numYears);
    unempRate     = NaN(numDest, numYears);
    shareFromVen  = NaN(numDest, numYears);
    shareWithHelp = NaN(numDest, numYears);

    % === Step 2. Compute yearly cross-sections if at least one full year ==
    if analysisPeriods > 0
        periodIdx = 1:analysisPeriods;

        % Extract the relevant panel slices only once to avoid repeated indexing.
        locationPanel = agentData.location(:, periodIdx);
        statePanel    = agentData.state(:, periodIdx);
        incomePanel   = agentData.income(:, periodIdx);
        helpPanel     = agentData.arrivalWithHelp(:, periodIdx);
        fromVenPanel  = agentData.arrivalFromVenezuela(:, periodIdx);

        % Previous-period location: replicate column 1 because period 0 is undefined.
        prevLocationPanel = agentData.location(:, [periodIdx(1), periodIdx(1:end-1)]);

        for destIdx = 1:numDest
            dest = destLocations(destIdx);

            for yearIdx = 1:numYears
                % Columns corresponding to the two semesters of this model year.
                cols = ((yearIdx - 1) * periodsPerYear + 1) : (yearIdx * periodsPerYear);

                % Agents physically located in the destination during the year.
                inDest = (locationPanel(:, cols) == dest);

                % Same-period arrivals: in destination now but not in previous period.
                arrivals = inDest & (prevLocationPanel(:, cols) ~= dest);

                % Stayers are agents present in the previous period as well.
                stayers = inDest & ~arrivals;

                % Employment status among stayers.
                employedStayers   = stayers & (statePanel(:, cols) > dims.B);
                unemployedStayers = stayers & (statePanel(:, cols) <= dims.B);

                % Average income among employed stayers (new arrivals are excluded).
                employedCount = sum(employedStayers(:));
                if employedCount > 0
                    totalIncome = sum(incomePanel(employedStayers));
                    avgIncome(destIdx, yearIdx) = totalIncome / employedCount;
                end

                % Unemployment rate computed only among stayers, weighted by headcount.
                stayersPerPeriod = sum(stayers, 1);
                validMask = stayersPerPeriod > 0;
                if any(validMask)
                    unemployedCounts = sum(unemployedStayers(:, validMask), 1);
                    periodRates = unemployedCounts ./ stayersPerPeriod(validMask);
                    weights = stayersPerPeriod(validMask);
                    unempRate(destIdx, yearIdx) = sum(periodRates .* weights) / sum(weights);
                end

                % Composition of arrivals (shares from Venezuela / with help).
                arrivalsMaskVec = reshape(arrivals, [], 1);
                if any(arrivalsMaskVec)
                    fromVec = reshape(fromVenPanel(:, cols), [], 1);
                    helpVec = reshape(helpPanel(:, cols), [], 1);

                    validFromMask = arrivalsMaskVec & ~isnan(fromVec);
                    if any(validFromMask)
                        shareFromVen(destIdx, yearIdx) = sum(fromVec(validFromMask)) / sum(validFromMask);
                    end

                    validHelpMask = arrivalsMaskVec & ~isnan(helpVec);
                    if any(validHelpMask)
                        shareWithHelp(destIdx, yearIdx) = sum(helpVec(validHelpMask)) / sum(validHelpMask);
                    end
                end
            end
        end
    end

    % === Step 3. Cohort unemployment for long-run stayers ================
    locationFull = agentData.location;
    stateFull    = agentData.state;

    TiLookup = NaN(dims.N, 1);
    if dims.N >= 2, TiLookup(2) = 6; end
    if dims.N >= 3, TiLookup(3) = 4; end
    if dims.N >= 4, TiLookup(4) = 6; end

    arrivalYears     = computeFirstArrivalYears(locationFull, destLocations);
    totalPeriodsFull = size(locationFull, 2);

    stayerUnemployment = cell(numDest, 1);
    for destIdx = 1:numDest
        dest = destLocations(destIdx);
        Ti   = TiLookup(dest);

        if isnan(Ti)
            stayerUnemployment{destIdx} = NaN;
            continue;
        end

        lastPeriod = min(2 * Ti, totalPeriodsFull);
        if lastPeriod < 2 * Ti - 1
            stayerUnemployment{destIdx} = NaN(Ti, 1);
            continue;
        end

        rates = NaN(Ti, 1);
        for yearIdx = 1:Ti
            % Identify the cohort that first arrived in this model year.
            cohortMask = (arrivalYears(:, destIdx) == yearIdx);
            if ~any(cohortMask)
                continue;
            end

            % Keep the cohort members who are still in the destination at lastPeriod.
            stayerCohort = cohortMask & (locationFull(:, lastPeriod) == dest);
            if ~any(stayerCohort)
                continue;
            end

            % Evaluate unemployment in the final two semesters of the window.
            evalPeriods = (2 * Ti - 1) : min(2 * Ti, totalPeriodsFull);
            unempShares = zeros(numel(evalPeriods), 1);
            for pIdx = 1:numel(evalPeriods)
                period = evalPeriods(pIdx);
                unemployedNow = stateFull(stayerCohort, period) <= dims.B;
                unempShares(pIdx) = sum(unemployedNow) / sum(stayerCohort);
            end

            rates(yearIdx) = mean(unempShares);
        end

        stayerUnemployment{destIdx} = rates;
    end

    % === Step 4. Assemble the output structure ===========================
    moments.locations                 = destLocations;
    moments.years                     = 1:numYears;
    moments.averageIncome             = avgIncome;
    moments.unemploymentRate          = unempRate;
    moments.arrivalShareFromVenezuela = shareFromVen;
    moments.arrivalShareWithHelp      = shareWithHelp;
    moments.arrivalShareFromOther     = 1 - shareFromVen;
    moments.arrivalShareWithoutHelp   = 1 - shareWithHelp;
    moments.stayerUnemploymentYears   = TiLookup(destLocations);
    moments.stayerUnemployment        = stayerUnemployment;
end