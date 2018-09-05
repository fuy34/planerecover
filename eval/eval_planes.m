clc
clear
close all

pred_path = '../tst/plane_sgmts';
label_path = 'labels/SYN_GT_sgmt';

D = dir(fullfile(pred_path , '*.png'));
ri_all = zeros([length(D),1]);
gce_all = zeros([length(D),1]);
sc_all = zeros([length(D),1]);
vi_all = zeros([length(D),1]);

for res_idx = 1:length(D)
    imgname = D(res_idx).name;
    pred_res = imread(fullfile(pred_path, imgname));
    
    ID = imgname(1:end-4); 
    label_all = load(fullfile(label_path, [ ID, '-planes.mat']));
    label = uint8(label_all.assignment);
    
    % based on the experiment it seems upsample the pred to label size will
    % lead to a better result
    pred_res_upsample = imresize(pred_res,[380 640],'nearest');
    [ri_all(res_idx), gce_all(res_idx), vi_all(res_idx)] = ...
    compare_segmentations(label, pred_res_upsample);

    if min(min(pred_res)) < 1
        pred_res_upsample = pred_res_upsample - min(min(pred_res_upsample)) + 1;
    end
    if min(min(label)) < 1
        label = label - min(min(label)) + 1;
    end
    
    sc_all(res_idx) = compute_sc(uint8(pred_res_upsample), label);  
end

avg_ri  =  mean(ri_all);  % the higher the better
avg_vi  =  mean(vi_all);  % the lower  the better
avg_sc  =  mean(sc_all);  % the higher the better

std_ri  =  std(ri_all);
std_vi  =  std(vi_all);
std_sc  =  std(sc_all);

sprintf('ri: %.3f, voi: %.3f, sc: %.3f',avg_ri, avg_vi, avg_sc)
