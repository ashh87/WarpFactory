function [Metric] = metricGet_WarpShellComoving(gridSize,worldCenter,m,R1,R2,Rbuff,sigma,smoothFactor,vWarp,doWarp,gridScaling)


%% METRICGET_WARPSHELLCOMOVING: Builds the Warp Shell metric in a comoving frame
% https://iopscience.iop.org/article/10.1088/1361-6382/ad26aa
%
%   INPUTS: 
%   gridSize - 1x4 array. world size in [t, x, y, z], double type.
%   
%   worldCenter - 1x4 array. world center location in [t, x, y, z], double type.
%
%   m - total mass of the warp shell
%
%   R1 - inner radius of the shell
%
%   R2 - outer radius of the shell
%
%   Rbuff - buffer distance between the shell wall and when the shift
%   starts to change
%
%   sigma - sharpness parameter of the shift sigmoid
%
%   smoothfactor - factor by which to smooth the walls of the shell
% 
%   vWarp - speed of the warp drive in factors of c, along the x direction, double type.
% 
%   doWarp - 0 or 1, whether or not to create the warp effect inside the
%   shell
% 
%   gridScale - scaling of the grid in [t, x, y, z]. double type.
%
%   OUTPUTS: 
%   metric - metric struct object. 

%%

% input values
if nargin < 6
    Rbuff = 0;
end
if nargin < 7
    sigma = 0;
end
if nargin < 8
    smoothFactor = 1;
end
if nargin < 9
    vWarp = 0;
end
if nargin < 10
    doWarp = 0;
end
if nargin < 11
    gridScaling = [1,1,1,1];
end

Metric.type = "metric";
Metric.name = "Comoving Warp Shell";
Metric.scaling = gridScaling;
Metric.coords = "cartesian";
Metric.index = "covariant";
Metric.date = date;

% declare radius array
worldSize = sqrt((gridSize(2)*gridScaling(2)-worldCenter(2))^2+(gridSize(3)*gridScaling(3)-worldCenter(3))^2+(gridSize(4)*gridScaling(4)-worldCenter(4))^2);
rSampleRes = 10^5;
rsample = linspace(0,worldSize*1.2,rSampleRes);

% construct rho profile
rho = zeros(1,length(rsample))+m/(4/3*pi*(R2^3-R1^3)).*(rsample>R1 & rsample<R2);
Metric.params.rho = rho;

[~, maxR] = min(diff(rho>0));
maxR = rsample(maxR);

% construct mass profile
M = cumtrapz(rsample, 4*pi.*rho.*rsample.^2);

% construct pressure profile
P = TOVconstDensity(R2,M,rho,rsample);
Metric.params.P = P;

% smooth functions
rho = movingAverageSmooth(movingAverageSmooth(movingAverageSmooth(movingAverageSmooth(rho,1.79*smoothFactor),1.79*smoothFactor),1.79*smoothFactor),1.79*smoothFactor);
rho = rho';
Metric.params.rhosmooth = rho;

P = movingAverageSmooth(movingAverageSmooth(movingAverageSmooth(movingAverageSmooth(P,smoothFactor),smoothFactor),smoothFactor),smoothFactor);
P = P';
Metric.params.Psmooth = P;

% reconstruct mass profile
M = cumtrapz(rsample, 4*pi.*rho.*rsample.^2);
M(M<0) = max(M);


% save varaibles
Metric.params.M = M;
Metric.params.rVec = rsample;


% set shift line vector
shiftRadialVector = compactSigmoid(rsample,R1,R2,sigma,Rbuff);
shiftRadialVector = movingAverageSmooth(movingAverageSmooth(shiftRadialVector,smoothFactor),smoothFactor);

% construct metric using spherical symmetric solution: 
% solve for B
B = (1-2*G.*M./rsample/c^2).^(-1);
B(1) = 1;

% solve for a
a = alphaNumericSolver(M,P,maxR,rsample);

% solve for A from a
A = -exp(2.*a);

% save variables to the metric.params
Metric.params.A = A;
Metric.params.B = B;

% return metric boosted and in cartesian space
Metric.tensor = cell(4);
for mu = 1:4
    for nu = 1:4
        Metric.tensor{mu,nu} = zeros(gridSize);
    end
end
epsilon = 0;

% Evaluate every grid point at once (previously a per-point triple loop).
[I, J, K] = ndgrid(1:gridSize(2), 1:gridSize(3), 1:gridSize(4));
x = I*gridScaling(2) - worldCenter(2);
y = J*gridScaling(3) - worldCenter(3);
z = K*gridScaling(4) - worldCenter(4);

r = sqrt(x.^2 + y.^2 + z.^2) + epsilon;
theta = atan2(sqrt(x.^2 + y.^2), z);
phi = atan2(y, x);

% Fractional index into rsample. rsample is uniform, so the lower bracketing sample is
% found directly instead of by a nearest-neighbour search over all samples; the two
% corrections reproduce the original rule (largest index with rsample(idx) <= r).
dr = rsample(2) - rsample(1);
idx = floor((r - rsample(1))/dr) + 1;
idx = idx - (rsample(idx) > r);
idx = idx + (rsample(idx + 1) <= r);
minIdx = idx + (r - rsample(idx))./(rsample(idx + 1) - rsample(idx));

g11_sph = legendreRadialInterp(A, minIdx);
g22_sph = legendreRadialInterp(B, minIdx);
[g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart] = sph2cartDiag(theta, phi, g11_sph, g22_sph);

shape = [1, gridSize(2:4)];
Metric.tensor{1,1} = reshape(g11_cart, shape);
Metric.tensor{2,2} = reshape(g22_cart, shape);
Metric.tensor{2,3} = reshape(g23_cart, shape);
Metric.tensor{3,2} = Metric.tensor{2,3};
Metric.tensor{2,4} = reshape(g24_cart, shape);
Metric.tensor{4,2} = Metric.tensor{2,4};
Metric.tensor{3,3} = reshape(g33_cart, shape);
Metric.tensor{3,4} = reshape(g34_cart, shape);
Metric.tensor{4,3} = Metric.tensor{3,4};
Metric.tensor{4,4} = reshape(g44_cart, shape);
ShiftMatrix = reshape(legendreRadialInterp(shiftRadialVector, minIdx), shape);

if doWarp
    Metric.tensor{1,2} = Metric.tensor{1,2}-Metric.tensor{1,2}.*ShiftMatrix - ShiftMatrix*vWarp;
    Metric.tensor{2,1} = Metric.tensor{1,2};
end

end
