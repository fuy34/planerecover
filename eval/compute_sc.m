function sc = compute_sc(seg, groundTruth)
% note: the region label should start with 1

cntP = 0;
sumP = 0;
cntR = 0;
sumR = 0;

[matches] = match_segmentations(seg, groundTruth);

% as matches contains accuracies for all ground truth labels, we have
% to take the maximum in both dimensions to get the best accuracies
matchesSeg = max(matches, [], 2);
matchesGT = max(matches, [], 1);

regionsSeg = regionprops(seg, 'Area');
regionsGT = regionprops(groundTruth, 'Area');
% cntP/sumP will be the covering of the segmentation by the ground
% truth segmentation
for r = 1 : numel(regionsSeg)
    cntP = cntP + regionsSeg(r).Area*matchesSeg(r);
    sumP = sumP + regionsSeg(r).Area;
end;

% cntR/sumR will be the covering of the ground truth segmentation by
% the segmentation
for r = 1 : numel(regionsGT)
    cntR = cntR +  regionsGT(r).Area*matchesGT(r);
    sumR = sumR + regionsGT(r).Area;
end;

sc = (cntP/sumP + cntR/sumR)/2;