clear
close all
clc
%%

%prepare the black-white patterns for each symbol
patterns = getPatternsUPCA();

%the digits that will be encoded in the barcode
code = [0 1 2 3 4 5 6 7 8 9 0];
% code = randi([0 9],[1,11]);

%Checksum computation
chksum = mod(10 - mod(3*sum(code(1:2:end))+sum(code(2:2:end)),10),10);
code = [code chksum];

%%

%generate the 1D barcode

stripes = [patterns{1} patterns{1}]; %initial white space
stripes = [stripes patterns{3}]; %initial guard

for i = 1:6
    tempStripe = patterns{code(i)+6}; %get the code for the corresponding left digit
    stripes = [stripes tempStripe];
end

stripes = [stripes patterns{5}]; %middle guard

for i = 7:12
    tempStripe = patterns{code(i)+16}; %get the code for the corresponding right digit
    stripes = [stripes tempStripe];
end

stripes = [stripes patterns{4}]; %end guard

stripes = [stripes patterns{2} patterns{2}]; %final white space

%% Generate the scanline


obs_noise = 0;

%obs is the observed scanline (x_n in the document)
obs = 255*stripes;
obs = obs + obs_noise * randn(size(obs));
obs(obs<0) = 0;
obs(obs>255) = 255;

%you can play with 'obs_noise'

%generate the image
bc_image = uint8((255- repmat(obs, [100 1])));

figure;
imshow(bc_image)
set(gcf,'Position',[100 100 1000 500]);
title(num2str(code));
