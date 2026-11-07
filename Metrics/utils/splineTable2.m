function V = splineTable2(xg, yg, Z, xq, yq, yParity)
%% SPLINETABLE2: C^2 interpolation of a uniform 2-D table at scattered points
%
%   V = splineTable2(xg, yg, Z, xq, yq, yParity)
%
%   Tensor-product cubic B-spline interpolant of Z(i,j) = f(xg(i), yg(j)), evaluated at the points
%   (xq, yq) (any shape, same size). xg and yg must be uniformly spaced. Unlike interp2's 'linear'
%   or 'pchip', the interpolant has continuous second derivatives, so finite-difference curvature
%   computed from it has no spikes at the table nodes. Queries outside the table are clamped to it.
%
%   yParity (optional) treats yg as a polar angle from 0 to pi whose ends are the axis: the table
%   is extended across both ends by reflection, f(-y) = yParity*f(y) and
%   f(2*pi - y) = yParity*f(y), before fitting (+1 for even functions such as A_r, -1 for odd ones
%   such as A_theta). This makes the interpolant smooth across the axis. Without it, natural end
%   conditions are used.

if nargin < 6
    yParity = [];
end
xg = xg(:); yg = yg(:);
hx = xg(2) - xg(1); hy = yg(2) - yg(1);
if max(abs(diff(xg) - hx)) > 1e-9*abs(hx) || max(abs(diff(yg) - hy)) > 1e-9*abs(hy)
    error('splineTable2: table axes must be uniformly spaced');
end

npad = 0;
if ~isempty(yParity)
    npad = 4;
    if abs(yg(1)) > 1e-9*hy || abs(yg(end) - pi) > 1e-9*hy
        error('splineTable2: with yParity, yg must run from 0 to pi');
    end
    Z = [yParity*Z(:, npad+1:-1:2), Z, yParity*Z(:, end-1:-1:end-npad)];
end
y0 = yg(1) - npad*hy;

% coefficients: Z = Bx * C * By', solved one direction at a time
C = bsplineCoefficients(Z);
C = bsplineCoefficients(C.').';

% evaluation: c(i-1..i+2) with the cubic B-spline weights of the fractional position
sz = size(xq);
u = (xq(:) - xg(1))/hx; w = (yq(:) - y0)/hy;
u = min(max(u, 0), numel(xg) - 1); w = min(max(w, 0), size(Z, 2) - 1);
iu = min(floor(u), numel(xg) - 2); iw = min(floor(w), size(Z, 2) - 2);
bu = bsplineWeights(u - iu); bw = bsplineWeights(w - iw);
nr = size(C, 1);
V = zeros(numel(u), 1);
for a = 0:3
    for b = 0:3
        % coefficient index: node i has coefficient row i+2 (row 1 is the extra c_{-1})
        V = V + bu(:, a+1).*bw(:, b+1).*C(iu + a + 1 + (iw + b)*nr);
    end
end
V = reshape(V, sz);

end


function C = bsplineCoefficients(F)
% Columns of F sampled at n uniform nodes -> n+2 coefficients c_{-1}..c_n per column with
% f_i = (c_{i-1} + 4 c_i + c_{i+1})/6 and natural ends (c_{-1} - 2 c_0 + c_1 = 0, same at n).
n = size(F, 1);
e = ones(n, 1);
M = spdiags([e, 4*e, e]/6, [0, 1, 2], n, n+2);
M = [sparse([1, 1, 1], [1, 2, 3], [1, -2, 1], 1, n+2); M; ...
     sparse([1, 1, 1], [n, n+1, n+2], [1, -2, 1], 1, n+2)];
C = M \ [zeros(1, size(F, 2)); F; zeros(1, size(F, 2))];
end


function B = bsplineWeights(t)
B = [(1 - t).^3, 3*t.^3 - 6*t.^2 + 4, -3*t.^3 + 3*t.^2 + 3*t + 1, t.^3]/6;
end
