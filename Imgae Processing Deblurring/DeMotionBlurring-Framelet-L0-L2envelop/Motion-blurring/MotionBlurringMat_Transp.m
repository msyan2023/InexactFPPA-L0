function Img_out = MotionBlurringMat_Transp(Img, filter)
%% motion blurring matrix transpose applying on Img
[numR, numC] = size(filter);

extR = floor(numR/2);  %% extended number for row
extC = floor(numC/2);  %% extended number for column

filterR = flip(filter,1); %% flip about row
filterC = flip(filter,2); %% flip about column
filterRC = flip(flip(filter,1),2); %% flip about double directions

extImg = wextend('ar', 'sym',Img, extR);
extImg = wextend('ac', 'sym',extImg, extC);
extImg(extR+1:end-extR, extC+1:end-extC) = 0;

Img1 = BTTB_Transp(Img, filterRC);  %% zero padding with double flipped filter
Img2 = BTHB_Transp(extImg,extR,extC,filterC);   
Img3 = BHTB_Transp(extImg,extR,extC,filterR);
Img4 = BHHB_Transp(extImg,extR,extC,filter);

Img_out = Img1 + Img2 + Img3 + Img4;    %% Img_out = blurring matirx transpose * Image
end