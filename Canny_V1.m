close all;
clear all;
clc
Img = imread('lake.png');
%i = im2double(Img);
lena = .299*Img(:,:,1) + .587*Img(:,:,2) + .114*Img(:,:,3);


figure, imshow(lena);

I = double(lena)/255;
J = canny_edge_detection(I, 1, 0.06, 0.12);
figure, imshow(J);
