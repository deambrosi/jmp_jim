function plotOutcomeCase(M_total, M_network, agentData, dims, settings, caseName)
% PLOTOUTCOMECASE Generate and save plots for a given simulation case.
%
%   plotOutcomeCase(M_total, M_network, agentData, dims, settings, caseName)
%
%   Saves three figures to the 'results' folder:
%       - location shares over time
%       - networked shares over time
%       - location evolution by wealth bin
%
%   AUTHOR: Agustin Deambrosi
% -------------------------------------------------------------------------
    if ~exist('results','dir'); mkdir('results'); end

    % 1. Aggregate location shares
    fig = figure('Visible','on');
    plot(M_total', 'LineWidth', 2);
    title(sprintf('Share of Agents - %s', caseName));
    xlabel('Time Period'); ylabel('Share of All Agents');
    legend(arrayfun(@(i) sprintf('Location %d', i), 1:dims.N, 'UniformOutput', false));
    grid on;
    saveas(fig, fullfile('results', sprintf('loc_share_%s.png', caseName)));

    % 2. Networked shares
    fig = figure('Visible','on');
    plot(M_network', 'LineWidth', 2);
    title(sprintf('Share of Networked Agents - %s', caseName));
    xlabel('Time Period'); ylabel('Share of Networked Agents');
    legend(arrayfun(@(i) sprintf('Location %d', i), 1:dims.N, 'UniformOutput', false));
    grid on;
    saveas(fig, fullfile('results', sprintf('net_share_%s.png', caseName)));

    % 3. Evolution by wealth bins
    locHist = agentData.location;
    wealthHist = agentData.wealth;
    bin1 = wealthHist(:,1) <= 5;
    bin2 = wealthHist(:,1) > 5 & wealthHist(:,1) <= 10;
    bin3 = wealthHist(:,1) > 10;

    compute_location_shares = @(mask) ...
        cell2mat(arrayfun(@(t) ...
            accumarray(locHist(mask, t), 1, [dims.N, 1]) / sum(mask), ...
            1:settings.T, 'UniformOutput', false));

    share_bin1 = compute_location_shares(bin1);
    share_bin2 = compute_location_shares(bin2);
    share_bin3 = compute_location_shares(bin3);

    fig = figure('Visible','on');
    subplot(3,1,1); plot(share_bin1'); title('Bin 1: Initial Wealth Index 1-5');
    xlabel('Time'); ylabel('Share'); grid on;
    subplot(3,1,2); plot(share_bin2'); title('Bin 2: Initial Wealth Index 6-10');
    xlabel('Time'); ylabel('Share'); grid on;
    subplot(3,1,3); plot(share_bin3'); title('Bin 3: Initial Wealth Index 11-15');
    xlabel('Time'); ylabel('Share'); grid on;
    legend(arrayfun(@(i) sprintf('Location %d', i), 1:dims.N, 'UniformOutput', false), 'Location', 'eastoutside');
    saveas(fig, fullfile('results', sprintf('bin_evolution_%s.png', caseName)));
end
