function Img_out = BHTB_Transp(extImg,extR,extC,filter)

%% BHTB (Block Hankel with Toeplitz Blocks)

extImg(1:extR,:) = 0;
extImg(end-extR+1:end,:) = 0;

%% cut for 0 padding since the image is already extended
Img_out = imfilter(extImg, filter);
Img_out = Img_out(extR+1:end-extR, extC+1:end-extC);
 
end