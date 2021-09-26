function J = canny_edge_detection(I, sigma, Tl, Th)
%canny_edge_detection function uses Canny's algorithm to detect edges on an
%image
% Input image I is type double in range of [0,1].
% Parameter sigma is the standard deviation of the Gaussian filter used in
% the DRoG operator. 
% Tl and Th are the absolut values of the low and high
% thresholds used to distinguish weak and strong edges.
% Output image J is binary (0,1), and it only contains edges. 
% 
% J = canny_edge_detection(I, sigma) uses the default values for Tl = 0.02,
% and for Th = 0.5.
% J = canny_edge_detection(I) uses the default values for sigma = sqrt(2),
% Tl = 0.02, and for Th = 0.5.
%
% 
%See also edge

% --- Argument verification --- %
if (nargin < 1) || (nargin>4) 
    error('Error: Number of parameters sent to the function ''canny_edge_detection'' exceeds expected range');
elseif nargin == 3 
    error('Error: Function ''canny_edge_detection'' recieved only 1 threshold');
elseif nargin == 2
    Tl = 0.2;
    Th = 0.4*Tl;
elseif nargin == 1
    Tl = 0.2;
    Th = 0.4*Tl;
    sigma = sqrt(3);
end
% Checking a pixel value format of parameter I
if (~isa(I,'double')) 
    error('Error: Input argument I in ''canny_edge_detection'' function has to be a type double');
end
% Checking a if value of parameter limit exceeds demanded range
if (min(I(:))<0 || max(I(:))>1) 
    error('Error: Input argument I in ''canny_edge_detection'' function has to be in range of [0,1]');
end

% 1. and 2. Filter the input image with a Gaussian function and determine
% horizontal and vertical gradients using DRoG function
[I_dx, I_dy] = DRoG(I, sigma);

% 3. STEP: Determine the magnitude and angle of the gradient
[Id_magnitude, Id_angle] = magnitude_and_angle(I_dx, I_dy);

% 4. STEP: Quantization of the gradient
Id_angle = gradient_quantization(Id_angle);

% 5. STEP: Suppress non-representing gradient values
% local maximum
window_size = 3;
J = non_max_supression(Id_magnitude, Id_angle, window_size);

% 6. STEP: Determining maps of strong and weak edges
weak = 50/250;
strong = 1;
[J, ~, ~] = threshold(J, Tl, Th, weak, strong);

% 7. STEP: Connect the weak edges with the strong ones
window_size = 3;
J = hysteresis(J, window_size, weak, strong);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local Function: DRoG
%
function [I_dx, I_dy] = DRoG(I, sigma)
% 
% Applies the DRoG operator to input image I.
% The Gaussian operator function has a standard sigma deviation,
% a window dimension is the first odd integer greater than or equal to 6 * sigma
%
% OUTPUTS:
% I_dx: Partial gradient of the image in the x direction
% I_dy: Partial gradient of the image in the y direction

% define window dimensions
kernel_size = ceil(6*sigma);
if (mod(kernel_size, 2) == 0)
    kernel_size = kernel_size + 1;
end
if kernel_size < 3 % minimum kernel size is 3
    kernel_size = 3;
end
% creating a gaussian filter
Gauss_filter  = fspecial('gaussian', kernel_size, sigma);
% numerical gradient
[Hx, Hy] = gradient(Gauss_filter); 
Hx = Hx/sum(sum(abs(Hx)));
Hy = Hy/sum(sum(abs(Hy)));
% filtering (operator application)
I_dx = convolve(I, Hx);
I_dy = convolve(I, Hy);
%I_dx = imfilter(I, Hx, 'replicate', 'same');
%I_dy = imfilter(I, Hy, 'replicate', 'same'); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local function : magnitude_and_angle
%
function [Id_magnitude, Id_angle] = magnitude_and_angle(I_dx, I_dy)
%
%   A function that calculates the magnitude and angle of the gradient
%

Id_magnitude = sqrt(I_dx.^2 + I_dy.^2);
Id_angle = atan(I_dy./I_dx); 
Id_angle(Id_magnitude == 0) = 0; 
Id_angle = Id_angle*360/2/pi; % conversion from radians to degrees
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local function : gradient_quantization
%
function Id_angle = gradient_quantization(Id_angle)
%
%   Quantizes the gradient on 4 levels: -45, 0, 45, 90
%

Id_angle(Id_angle > -67.5 & Id_angle < -22.5) = -45;
Id_angle(Id_angle >= -22.5 & Id_angle <=22.5) = 0;
Id_angle(Id_angle > 22.5 & Id_angle < 67.5) = 45;
Id_angle(Id_angle >= 67.5 & Id_angle <= 90) = 90;
Id_angle(Id_angle >= -90 & Id_angle <= -67.5) = 90;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local function: non_max_supression
%
function J = non_max_supression(Id_magnitude,Id_angle, window_size)
%
%   Suppression of gradient values that do not represent local maxima
%

[N, M] = size(Id_magnitude);
half = (window_size-1)/2; % half of the window dimension
J = Id_magnitude; % output image initialization
Id_magnitude = padarray(Id_magnitude, [half, half], 'replicate');

for i = 1+half:N+half
    for j = 1+half:M+half
        switch Id_angle(i-half,j-half)
            case 45
                window = Id_magnitude(i-half:i+half,j-half:j+half).*[1 0 0;...
                                                                     0 1 0;...
                                                                     0 0 1];
            case 90
                window = Id_magnitude(i-half:i+half,j-half:j+half).*[0 1 0;...
                                                                     0 1 0;...
                                                                     0 1 0];
            case -45
                window = Id_magnitude(i-half:i+half,j-half:j+half).*[0 0 1;...
                                                                     0 1 0;...
                                                                     1 0 0];
            case 0
                window = Id_magnitude(i-half:i+half,j-half:j+half).*[0 0 0;...
                                                                     1 1 1;...
                                                                     0 0 0];
        end

        max_ind = find(window == max(window(:))); 
        if(max_ind ~= (window_size^2-1)/2+1) % if the pixel with the max value is not in the middle of the window
                J(i-half,j-half) = 0; % we are not to be inplate
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local function: threshold
%
function [J, Weak_edges, Strong_edges] = threshold(I, Tl, Th, weak, strong)
%
%   Determining maps of strong and weak edges based on thresholds
%
% INPUTS:
% J: gradient magnitude
% Tl: absolute value of the lower threshold
% Th: absolute value of the upper threshold
% weak: the intensity of the pixel to be assigned to the weak edges
% strong: pixel intensity to be assigned to strong edges
% OUTPUTS:
% J: an edge map that contains both strong and weak edges
% Weak_edges: an edge map that contains only weak edges
% Strong_edges: an edge folder that contains only strong edges
%
J = I;
J(J >= Th) = strong;
J(J > Tl & J < Th) = weak;
J(J <= Tl) = 0;
Weak_edges = J;
Weak_edges(J == weak) = 1;
Weak_edges(J == strong) = 0;
Strong_edges = J;
Strong_edges(J == weak) = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local function: hysteresis
%
function J = hysteresis(I, window_size, weak, strong)
%
%  All pixels that have at least one strong edge in their neighborhood
% are declared as an edge pixel
%
J = I; % output matrix initialization
[N, M] = size(J);
half = (window_size-1)/2;
changed = true;
while changed 
    changed = false;
    Edges = padarray(J, [half, half], 'replicate'); % after going through all the pixels update the edges
     for i = 1 + half: N + half% in these two loops pass through all pixels once
         for j = 1 + half: M + half
             if Edges (i, j) == weak% if the edge is weak
                 window = Edges (i-half: i + half, j-half: j + half); % environment around the weak edge
                 if ~ isempty (find (window == strong))% if there is at least one 'strong' pixel in the environment
                     J (i-half, j-half) = strong; % must not be inplace
                    changed = true;
                end
            end          
        end
    end
end
J(J == weak) = 0;% Set the remaining edge to 0
end