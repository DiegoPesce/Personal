%% Get images
clear

global_image = imread('./images/MyImage.jpg');
global_image = global_image(:,:,2);
template_image = imread('./images/MySubImage.jpg');
template_image = template_image(:,:,2);

global_image = imresize(global_image, 0.2);
template_image = imresize(template_image, 0.2);

[global_row, global_column] = size(global_image);
[template_row, template_column] = size(template_image);

%padding
padded_image = zeros([global_row+2*(template_row-1), global_column+2*(template_column-1)],"uint8");
padded_image( template_row:template_row+global_row-1, template_column:template_column+global_column-1 ) = global_image;
padded_size = size(padded_image);
imshow(padded_image);
%%%%%%%%%%%%%%%%%%%

%% Sum Square Differences
%error
ssd = zeros([global_row+template_row-1, global_column+template_column-1]);
ssd_sz = size(ssd);
for x = 1:ssd_sz(2)
    for y = 1:ssd_sz(1)
        %ritaglio una sottoimmagine da confrontare
        sub_image = padded_image(y:y+template_row-1, x:x+template_column-1);
        ssd(y,x) = sum( (sub_image - template_image).^2,"all" );
    end
end
%% Show error
[~, idx] = min(ssd,[], 'all');
[r, c] = ind2sub(size(ssd), idx);
tmp = ssd;
tmp(r:r+template_row, c:c+template_column) = max(ssd, [], "all") + 1000;
imagesc(tmp);  
%%%%%%%%%%%%%%%%%%%%%

%% Cross Correlation
%calculation of correlation
cross_correlation = zeros([global_row, global_column]);
for x = 1:padded_size(2)-template_column+1
    for y = 1:padded_size(1)-template_row+1
        sub_image = padded_image(y:y+template_row-1, x:x+template_column-1);
        cross_correlation(y,x) = sum(sub_image.*template_image,"all");
    end
end

%% Show correlation
imagesc(cross_correlation);

%% Normalized Cross Correlation
template_scarto = template_image - mean(template_image,"all")*ones([template_row, template_column],"uint8");
std_template = std2(template_image);
%calculation of correlation
norm_cross_correlation = zeros([global_row, global_column]);
for x = 1:padded_size(2)-template_column+1
    for y = 1:padded_size(1)-template_row+1
        sub_image = padded_image(y:y+template_row-1, x:x+template_column-1);
        sub_scarto = sub_image - mean(sub_image,"all")*ones([template_row, template_column],"uint8");
        norm_cross_correlation(y,x) = sum(sub_scarto.*template_scarto,"all")/(std2(sub_image)*std_template);
    end
end

%% Show correlation
imagesc(norm_cross_correlation);