function [Metric] = metricGet_WarpShellGeneralShift(gridSize,worldCenter,m,R1,R2,smoothFactor,shift,gridScaling,form)


%% METRICGET_WARPSHELLGENERALSHIFT: Warp Shell with a general axisymmetric stationary shift
%
%   The static shell of metricGet_WarpShellComoving (same density, pressure, lapse and radial
%   metric) plus a stationary gravitomagnetic potential A = g_0i/(-g_00) that may depend on both
%   the distance r from the shell centre and the angle theta from the +x axis, and may have both
%   a radial and a polar component. The Comoving builder only allows A along x with a radial
%   profile, A_x = -vWarp*S(r)/alpha^2.
%
%   INPUTS:
%   gridSize - 1x4 array. world size in [t, x, y, z], double type.
%
%   worldCenter - 1x4 array. world center location in [t, x, y, z], double type.
%
%   m, R1, R2, smoothFactor - static shell parameters, as in metricGet_WarpShellComoving.
%
%   shift - struct describing A on a polar table about the +x axis:
%       shift.r   - 1 x Nr uniformly spaced radii (starting at 0 is fine)
%       shift.th  - 1 x Nth uniformly spaced polar angles from 0 to pi (from the +x axis)
%       shift.Ar  - Nr x Nth covariant radial component A_r(r, theta)
%       shift.Ath - Nr x Nth covariant polar component A_theta(r, theta) (optional, default 0).
%                   Must vanish on the axis for A to be regular there.
%       shift.amp - scalar multiplying A (optional, default 1)
%       The table is interpolated with C^2 cubic B-splines (splineTable2, mirrored across the
%       axis), so the second derivatives the solver takes have no interpolation kinks. A is set to zero
%       for r > max(shift.r), so the table should extend to where A is zero or pure gauge.
%
%   gridScaling - scaling of the grid in [t, x, y, z]. double type.
%
%   form - how A enters the metric (optional):
%       "stationary" (default): ds^2 = -alpha^2 (dt - A_i dx^i)^2 + h_ij dx^i dx^j with h the
%           static shell's spatial metric, i.e. g_00 = -alpha^2, g_0i = alpha^2 A_i,
%           g_ij = h_ij - alpha^2 A_i A_j. The field equations then depend on A only through
%           dA, so a re-slicing A -> A + grad f leaves the spacetime unchanged at all orders.
%       "shift": g_00 = -alpha^2, g_0i = alpha^2 A_i, g_ij = h_ij (the Comoving builder's
%           convention; A_i A_j terms of second order then depend on the gauge of A).
%
%   OUTPUTS:
%   metric - metric struct object.

%%

if nargin < 8
    gridScaling = [1,1,1,1];
end
if nargin < 9
    form = "stationary";
end
if ~isfield(shift, 'Ath') || isempty(shift.Ath)
    shift.Ath = zeros(size(shift.Ar));
end
if ~isfield(shift, 'amp') || isempty(shift.amp)
    shift.amp = 1;
end
if ~any(strcmp(form, {"stationary", "shift"}))
    error('form must be "stationary" or "shift"');
end

% static shell: the Comoving builder without its shift
Metric = metricGet_WarpShellComoving(gridSize,worldCenter,m,R1,R2,0,0,smoothFactor,0,0,gridScaling);
Metric.name = "Warp Shell, general axisymmetric shift";
Metric.params.shift = shift;
Metric.params.form = form;

% polar coordinates about the +x axis
[I, J, K] = ndgrid(1:gridSize(2), 1:gridSize(3), 1:gridSize(4));
x = I*gridScaling(2) - worldCenter(2);
y = J*gridScaling(3) - worldCenter(3);
z = K*gridScaling(4) - worldCenter(4);
s = sqrt(y.^2 + z.^2);                  % distance from the x axis
r = sqrt(x.^2 + s.^2);
theta = atan2(s, x);

% A on the grid
inside = r <= max(shift.r);
Ar = zeros(size(r)); Ath = Ar;
rq = r(inside); tq = theta(inside);
Ar(inside) = splineTable2(shift.r, shift.th, shift.Ar, rq, tq, 1);
Ath(inside) = splineTable2(shift.r, shift.th, shift.Ath, rq, tq, -1);
Ar = shift.amp*Ar; Ath = shift.amp*Ath;

% Cartesian covariant components: A_i = A_r dr/dx^i + A_theta dtheta/dx^i, with
% dtheta/dx = -s/r^2 and dtheta/d(y,z) = x (y,z)/(s r^2). On the axis (s = 0) A_theta must
% vanish; there the A_theta/s factor is taken as zero.
rs = r; rs(rs == 0) = 1;
ss = s; ss(ss == 0) = 1;
At_over_s = Ath./(ss.*rs.^2); At_over_s(s == 0) = 0;
Acart = cell(1, 3);
Acart{1} = Ar.*x./rs - Ath.*s./rs.^2;
Acart{2} = Ar.*y./rs + At_over_s.*x.*y;
Acart{3} = Ar.*z./rs + At_over_s.*x.*z;

shape = [1, gridSize(2:4)];
alpha2 = -Metric.tensor{1,1};
for i = 1:3
    Acart{i} = reshape(Acart{i}, shape);
    Metric.tensor{1,i+1} = alpha2.*Acart{i};
    Metric.tensor{i+1,1} = Metric.tensor{1,i+1};
end
if strcmp(form, "stationary")
    for i = 1:3
        for j = 1:3
            Metric.tensor{i+1,j+1} = Metric.tensor{i+1,j+1} - alpha2.*Acart{i}.*Acart{j};
        end
    end
end

end
