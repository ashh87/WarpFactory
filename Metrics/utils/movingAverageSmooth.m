function c = movingAverageSmooth(y, span)
%MOVINGAVERAGESMOOTH Drop-in replacement for smooth(y, span) ('moving' method).
%   Same result as the Curve Fitting Toolbox function: window width is the largest
%   odd integer <= span, shrinking symmetrically near the ends, returned as a column.
%   Uses cumulative sums, so the cost is O(n) independent of span (smooth/filter is
%   O(n*span)), and it needs no toolbox.

y = y(:);
n = numel(y);
span = min(floor(span), n);
w = span - 1 + mod(span, 2);
if w <= 1
    c = y;
    return
end

halfWidth = (w - 1)/2;
k = (1:n)';
hk = min(min(k - 1, n - k), halfWidth);   % symmetric half-width at each sample
cs = [0; cumsum(y)];
c = (cs(k + hk + 1) - cs(k - hk)) ./ (2*hk + 1);
end
