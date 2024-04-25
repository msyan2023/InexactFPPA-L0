function y=mypsnr(original,image)

%This function serves to compute the psnr value of the image. image and original is the
%restored and original version respectively,
[ximage,yimage]=size(image);
y=10*log10(...
255^2*ximage*yimage/sum(sum((original-image).^2)));


end
