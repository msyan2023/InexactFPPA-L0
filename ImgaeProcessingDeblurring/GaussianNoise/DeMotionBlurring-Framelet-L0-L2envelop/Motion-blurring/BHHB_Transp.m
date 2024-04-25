function Img_out = BHHB_Transp(extImg,extR,extC,filter)

%% BHHB (Block Hankel with Hankel Blocks)


extImg(:, extC+1:end-extC) = 0;
extImg(extR+1:end-extR,:) = 0;

%% cut for 0 padding since the image is already extended
Img_out = imfilter(extImg, filter);
Img_out = Img_out(extR+1:end-extR, extC+1:end-extC);
 
end