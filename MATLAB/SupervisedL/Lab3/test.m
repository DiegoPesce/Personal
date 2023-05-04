global_image = imread('./images/MyImage.jpg');
blurred_image = imgaussfilt(global_image, 5);
imshow(global_image)

subplot(1,2,1)
imshow(global_image)
subplot(1,2,2)
imshow(blurred_image)