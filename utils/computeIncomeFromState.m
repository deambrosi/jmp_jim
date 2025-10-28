function income = computeIncomeFromState(locationIdx, stateIdx, params, grids, dims)
    psiIdx  = mod(stateIdx - 1, dims.B) + 1;
    income  = params.A(locationIdx) .* ...
			 (params.lb_pr.*(1-grids.psi(psiIdx)) + params.ub_pr.*grids.psi(psiIdx));
end