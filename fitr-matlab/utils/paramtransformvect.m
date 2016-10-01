% Accepts a single column vector of parameters and performs transformation element-wise

function y = paramtransformvect(params, rng, transformtype)

    pt = @(x) paramtransform(x, {rng}, transformtype);
    y = arrayfun(pt, params);

end
