function y=mypsnr(original,image,peakval)

%This function serves to compute the psnr value of the image. image and original is the
%restored and original version respectively,

% if the reference image is unit8, set peakval=255
% if the reference image is scaled, for Poisson noise, set peakval as the maximum value of the image
[ximage,yimage]=size(image);
MSE = sum(sum((original-image).^2))/(ximage*yimage);
y=10*log10(peakval^2/MSE);

%% this function is same as the built-in function "psnr" in Matlab,  
%  usage: y=psnr(image,original,peakval)
end
