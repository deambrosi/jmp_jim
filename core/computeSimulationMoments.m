function moments = computeSimulationMoments(agentData, dims, settings)
% COMPUTESIMULATIONMOMENTS  Aggregate simulated agent histories into moments.
%
%   Computes yearly statistics from simulation data where:
%   - Period 1 is the initial condition (excluded from analysis)
%   - Each year consists of 2 simulation periods (semesters)
%   - Unemployment rates exclude recent arrivals and are lagged by 1 year
%   - Migration stocks are measured at year end
%
% INPUTS:
%   agentData - Struct with agent trajectories (.location, .state, .income, etc.)
%   dims      - Model dimensions struct
%   settings  - Simulation settings struct
%
% OUTPUT:
%   moments   - Struct with computed statistics by destination and year

    %% ====================================================================
    %% STEP 1: SETUP AND PERIOD MAPPING
    %% ====================================================================
    
    % Define time structure: 2 periods = 1 year
    periodsPerYear = 2;
    
    % Period 1 is initial condition - we skip it
    % Starting from period 2, we group into years:
    % Year 1 = periods 2-3
    % Year 2 = periods 4-5
    % Year 3 = periods 6-7, etc.
    
    % Total periods available (excluding initial period 1)
    availablePeriods = settings.T - 1;  % Subtract initial period
    
    % Maximum years we want to compute (paper uses first 5 years)
    maxYearsDesired = 5;
    
    % Actual years we can compute given available data
    numYears = min(maxYearsDesired, floor(availablePeriods / periodsPerYear));
    
    
    % Destination locations (exclude Venezuela which is location 1)
    destLocations = 2:dims.N;
    numDest = numel(destLocations);
    
    %% ====================================================================
    %% STEP 2: INITIALIZE OUTPUT ARRAYS
    %% ====================================================================
    
    % Income and unemployment (lagged by 1 year)
    avgIncome     = NaN(numDest, numYears);
    unempRate     = NaN(numDest, numYears);
    
    % Arrival composition
    shareFromVen  = NaN(numDest, numYears);
    shareWithHelp = NaN(numDest, numYears);
    
    % Migration stocks at year end
    migrantStock  = NaN(numDest, numYears);
    
    %% ====================================================================
    %% STEP 3: EXTRACT FULL DATA PANELS (EXCLUDING INITIAL PERIOD)
    %% ====================================================================
    
    % Extract data starting from period 2 (skip initial condition)
    locationFull = agentData.location(:, 2:end);  % Skip period 1
    stateFull    = agentData.state(:, 2:end);     % Skip period 1
    incomeFull   = agentData.income(:, 2:end);    % Skip period 1
    
    % For arrival characteristics, we need the full timeline
    helpArrivalFull = agentData.arrivalWithHelp(:, 2:end);
    fromVenFull     = agentData.arrivalFromVenezuela(:, 2:end);
    
    numAgents = size(locationFull, 1);
    
    %% ====================================================================
    %% STEP 4: COMPUTE STATISTICS FOR EACH YEAR
    %% ====================================================================
    
    for yearIdx = 1:numYears
        
        % Map year to simulation periods (remember: period 1 was skipped)
        % Year 1 corresponds to original periods 2-3, which are now indices 1-2
        % Year 2 corresponds to original periods 4-5, which are now indices 3-4
        yearPeriods = ((yearIdx - 1) * periodsPerYear + 1) : (yearIdx * periodsPerYear);
        
        
        for destIdx = 1:numDest
            dest = destLocations(destIdx);
            
            %% --------------------------------------------------------
            %% 4A. MIGRANT STOCK AT YEAR END
            %% --------------------------------------------------------
            
            % Count agents in destination at END of the year
            lastPeriodOfYear = yearPeriods(end);
            agentsInDest = locationFull(:, lastPeriodOfYear) == dest;
            migrantStock(destIdx, yearIdx) = sum(agentsInDest) / numAgents;
            

            %% --------------------------------------------------------
            %% 4B. ARRIVAL COMPOSITION DURING THE YEAR
            %% --------------------------------------------------------
            
            % Identify NEW arrivals during this year
            % An arrival is someone who wasn't in destination in previous period
            newArrivals = false(numAgents, periodsPerYear);
            
            for pIdx = 1:periodsPerYear
                periodIdx = yearPeriods(pIdx);
                
                % Currently in destination
                inDestNow = locationFull(:, periodIdx) == dest;
                
                % Was in destination in previous period?
                if periodIdx == 1
                    % For the very first analyzed period, check original period 1
                    wasInDestBefore = agentData.location(:, 1) == dest;
                else
                    wasInDestBefore = locationFull(:, periodIdx - 1) == dest;
                end
                
                % New arrival = in destination now but wasn't before
                newArrivals(:, pIdx) = inDestNow & ~wasInDestBefore;
            end
            
            % Aggregate arrivals across the year
            arrivedThisYear = any(newArrivals, 2);
            numArrivals = sum(arrivedThisYear);
            
            if numArrivals > 0
                % Among arrivals, what share came from Venezuela?
                % Use the first period of arrival to determine origin
                for agentIdx = find(arrivedThisYear)'
                    firstArrivalPeriod = find(newArrivals(agentIdx, :), 1);
                    globalPeriodIdx = yearPeriods(firstArrivalPeriod);
                    
                    % The fromVen and help flags are set when they first arrive
                    % These should persist as characteristics of their arrival
                end
                
                % Calculate shares using the arrival flags
                % Take maximum across year periods since flag persists
                fromVenFlags = fromVenFull(arrivedThisYear, yearPeriods);
                withHelpFlags = helpArrivalFull(arrivedThisYear, yearPeriods);
                
                % An agent came from Venezuela if any period shows this
                cameFromVen = any(fromVenFlags, 2);
                cameWithHelp = any(withHelpFlags, 2);
                
                shareFromVen(destIdx, yearIdx) = mean(cameFromVen);
                shareWithHelp(destIdx, yearIdx) = mean(cameWithHelp);
                
                
            end
            
            %% --------------------------------------------------------
            %% 4C. UNEMPLOYMENT RATE (LAGGED AND EXCLUDING RECENT ARRIVALS)
            %% --------------------------------------------------------
            
            % Unemployment for year Y is measured at the END of year Y+1
            % and excludes agents who arrived in the previous 2 periods
            
            if yearIdx < numYears  % Can't compute lagged unemployment for last year
                
                % Measurement periods: end of NEXT year
                measureYear = yearIdx + 1;
                measurePeriods = ((measureYear - 1) * periodsPerYear + 1) : ...
                                 (measureYear * periodsPerYear);
                
                % For each measurement period, identify "established" agents
                % Established = in destination and arrived more than 2 periods ago
                
                periodUnempRates = [];
                periodWeights = [];
                
                for pIdx = 1:length(measurePeriods)
                    measPeriod = measurePeriods(pIdx);
                    
                    % Who is in destination during measurement?
                    inDestAtMeasure = locationFull(:, measPeriod) == dest;
                    
                    % When did they first arrive? Need to check arrival timing
                    % Agent is "established" if they've been there > 2 periods
                    establishedMask = false(numAgents, 1);
                    
                    for agentIdx = 1:numAgents
                        if inDestAtMeasure(agentIdx)
                            % Find when this agent first arrived at destination
                            % Look backwards from measurement period
                            firstInDest = find(locationFull(agentIdx, 1:measPeriod) == dest, 1);
                            
                            if isempty(firstInDest)
                                % Check if they were there in initial period
                                if agentData.location(agentIdx, 1) == dest
                                    firstInDest = 0;  % Was there from beginning
                                end
                            end
                            
                            % Established if arrived more than 2 periods before measurement
                            if ~isempty(firstInDest) && (measPeriod - firstInDest) >= 2
                                establishedMask(agentIdx) = true;
                            end
                        end
                    end
                    
                    % Among established agents, compute unemployment
                    numEstablished = sum(establishedMask);
                    
                    if numEstablished > 0
                        % Unemployed if state <= dims.B
                        unemployedMask = establishedMask & (stateFull(:, measPeriod) <= dims.B);
                        numUnemployed = sum(unemployedMask);
                        
                        periodUnempRates(end+1) = numUnemployed / numEstablished;
                        periodWeights(end+1) = numEstablished;
                        
                        
                    end
                end
                
                % Average unemployment across measurement periods, weighted by sample size
                if ~isempty(periodUnempRates)
                    unempRate(destIdx, yearIdx) = sum(periodUnempRates .* periodWeights) / ...
                                                  sum(periodWeights);
                    
                    
                end
                
                %% --------------------------------------------------------
                %% 4D. AVERAGE INCOME (ALSO LAGGED, AMONG EMPLOYED ESTABLISHED)
                %% --------------------------------------------------------
                
                % Similar logic: measure at end of next year, exclude recent arrivals
                incomeSum = 0;
                incomeCount = 0;
                
                for pIdx = 1:length(measurePeriods)
                    measPeriod = measurePeriods(pIdx);
                    
                    % Reuse establishedMask from unemployment calculation
                    inDestAtMeasure = locationFull(:, measPeriod) == dest;
                    establishedMask = false(numAgents, 1);
                    
                    for agentIdx = 1:numAgents
                        if inDestAtMeasure(agentIdx)
                            firstInDest = find(locationFull(agentIdx, 1:measPeriod) == dest, 1);
                            if isempty(firstInDest) && agentData.location(agentIdx, 1) == dest
                                firstInDest = 0;
                            end
                            if ~isempty(firstInDest) && (measPeriod - firstInDest) >= 2
                                establishedMask(agentIdx) = true;
                            end
                        end
                    end
                    
                    % Among established, count only EMPLOYED agents
                    employedMask = establishedMask & (stateFull(:, measPeriod) > dims.B);
                    
                    if sum(employedMask) > 0
                        incomeSum = incomeSum + sum(incomeFull(employedMask, measPeriod));
                        incomeCount = incomeCount + sum(employedMask);
                    end
                end
                
                if incomeCount > 0
                    avgIncome(destIdx, yearIdx) = incomeSum / incomeCount;
                    
                end
            end
        end
    end
    
    %% ====================================================================
    %% STEP 5: COHORT-BASED STAYER UNEMPLOYMENT
    %% ====================================================================
    
    % This tracks specific arrival cohorts over time
    TiLookup = NaN(dims.N, 1);
    if dims.N >= 2, TiLookup(2) = 6; end
    if dims.N >= 3, TiLookup(3) = 4; end
    if dims.N >= 4, TiLookup(4) = 6; end
    
    % Compute first arrival years (using full data including period 1)
    arrivalYears = computeFirstArrivalYears(agentData.location, destLocations);
    
    stayerUnemployment = cell(numDest, 1);
    
    for destIdx = 1:numDest
        dest = destLocations(destIdx);
        Ti = TiLookup(dest);
        
        if isnan(Ti)
            stayerUnemployment{destIdx} = NaN;
            continue;
        end
        
        % Need enough periods to track cohorts
        lastPeriod = min(2 * Ti, settings.T);
        if lastPeriod < 2 * Ti - 1
            stayerUnemployment{destIdx} = NaN(Ti, 1);
            continue;
        end
        
        rates = NaN(Ti, 1);
        
        for cohortYear = 1:Ti
            % Find agents who first arrived in this year
            cohortMask = (arrivalYears(:, destIdx) == cohortYear);
            if ~any(cohortMask)
                continue;
            end
            
            % Among cohort, keep only those still in destination at evaluation
            stayerCohort = cohortMask & (agentData.location(:, lastPeriod) == dest);
            if ~any(stayerCohort)
                continue;
            end
            
            % Measure unemployment at end of tracking period
            evalPeriods = (2 * Ti - 1) : min(2 * Ti, settings.T);
            unempShares = zeros(numel(evalPeriods), 1);
            
            for pIdx = 1:numel(evalPeriods)
                period = evalPeriods(pIdx);
                unemployedNow = agentData.state(stayerCohort, period) <= dims.B;
                unempShares(pIdx) = sum(unemployedNow) / sum(stayerCohort);
            end
            
            rates(cohortYear) = mean(unempShares);
        end
        
        stayerUnemployment{destIdx} = rates;
    end
    
    %% ====================================================================
    %% STEP 6: ASSEMBLE OUTPUT STRUCTURE
    %% ====================================================================
    
    moments.locations                 = destLocations;
    moments.years                     = 1:numYears;
    moments.averageIncome             = avgIncome;
    moments.unemploymentRate          = unempRate;
    moments.migrantStock              = migrantStock;  % NEW: stock at year end
    moments.arrivalShareFromVenezuela = shareFromVen;
    moments.arrivalShareWithHelp      = shareWithHelp;
    moments.arrivalShareFromOther     = 1 - shareFromVen;
    moments.arrivalShareWithoutHelp   = 1 - shareWithHelp;
    moments.stayerUnemploymentYears   = TiLookup(destLocations);
    moments.stayerUnemployment        = stayerUnemployment;
    

    
end
