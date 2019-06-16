% Read noisy image
I = imread('C:/Users/Akanksha/Desktop/vision/n.jpg');
O = imread('C:/Users/Akanksha/Desktop/vision/o.jpg');
% Perform Fast Fourier Transform on the image
f = fftshift(fft2(I));

% Find Power Spectrum
psd2D = abs(f).^2 + abs(conj(f)).^2 ;

% Find the local maxima 
hLocalMax = vision.LocalMaximaFinder('MaximumNumLocalMaxima',9, ...
                                      'NeighborhoodSize',[13,13], ...
                                      'Threshold',10);
coordinates = hLocalMax(psd2D);
min = 1000;

% Finding the sigma for which noise is minimum
for sigma = 1:100
    sum_square = filtering(sigma,f, coordinates);
    if(sum_square <= min)
        loc=sigma;
        min = sum_square;
    else
        break;
    end
end

% Finding the noiseless image for that optimal sigma
[sum_square, noiseless_image] = filtering(loc, f, coordinates);

% Display image and variance value
disp("Optimal Standard Deviation : " + loc);
disp("Optimal Variance : " + loc.^2);
disp("Minimum Noise : " + sum_square);
% new_fft = fftshift(fft2(noiseless_image));
% new_log = abs(new_fft).^2 + abs(conj(new_fft)).^2;

colormap(gray)
subplot(2,2,1),imagesc(noiseless_image); title('Noiseless Image')
subplot(2,2,2),imagesc(log(psd2D)); title('Power Spectrum')
%subplot(2,2,3),imagesc(filter); title('Filter Image')
% subplot(2,2,4),imagesc(log(new_log)); title('Noiseless Power Spectrum')
annotation('textbox', [0 0.9 1 0.1], ...
    'String', 'Fourier Analysis on Clown Image', ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 15, ...
    'FontWeight', 'bold')

figure,imshow(noiseless_image,[]);
%figure,imshow(Difference);

% Function to remove noise from an image. 
% Passing value of sigma, fft and local maxima coordinates as parameters
% Getting de-noised image and averaged sum square as output
function [sum_square, noiseless_image] = filtering(sigma, f, coordinates)
    O = imread('C:/Users/Akanksha/Desktop/vision/o.jpg');
    I = imread('C:/Users/Akanksha/Desktop/vision/n.jpg');
    
    % Find size of noisy image
    [p,q] = size(I);
    
    % Assigning fft to g
    g=f;
    
    % Finding size of kernel. If it is even, add 1
    m = floor(5*sigma);
    if mod(m,2)==0
        m = m+1;
    end
    
    % Generating a mXm Gaussian Kernel
    kernel = zeros(m,m);
    centre = (1+m)/2;
    for i=1:m
        for j=1:m
            diff = (i - centre).^2 + (j - centre).^2;
            kernel(i,j) = exp(-diff/(2* sigma.^2));
        end
    end
    kernel(:) = kernel(:)/sum(kernel(:));
    
    filter = ones(p,q);
    
    %Mapping values of kernel to filter in the matrix of size mXm centred
    %at local maxima points
    for k = 2:5
        for i = -floor(m/2):floor(m/2)
            for j = -floor(m/2):floor(m/2)
                filter(coordinates(k,2)+i, coordinates(k,1)+j) = kernel(i+ceil(m/2),j+ceil(m/2));
            end
        end
    end
    
    %Gaussian filtering can be achieved with repeated average filtering
    n = floor(3*(sigma.^2));
    for i=1:n
        g = g.*filter;
    end
    
    % Perform Inverse FFT on fft obtained after filtering
    noiseless_image = ifft2(ifftshift(g));
    
    % Subtracting the de-noised image with the original image
    Difference = double(O) - double(noiseless_image);

    % Squaring and averaging the result
    sum_square=0;
    for i =1:p
        for j =1:q
            sum_square = sum_square + Difference(i,j).^2;
        end
    end
    sum_square = sum_square/(p*q);
 end