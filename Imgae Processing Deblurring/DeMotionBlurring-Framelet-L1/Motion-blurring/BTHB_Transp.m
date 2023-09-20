function Img_out = BTHB_Transp(extImg,extR,extC,filter)

%% BTHB (Block Toeplitz with Hankel Blocks)
 
extImg(:,1:extC) = 0;
extImg(:,end-extC+1:end) = 0;

%% cut for 0 padding since the image is already extended
Img_out = imfilter(extImg, filter);
Img_out = Img_out(extR+1:end-extR, extC+1:end-extC);
 
end