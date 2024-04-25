function Img_out = BTTB_Transp(Img, filter)
%% BTTB (Block Toeplitz with Toeplitz Blocks)

Img_out = imfilter(Img, filter);  

end