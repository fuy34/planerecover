function [matches] = match_segmentations(seg, groundTruth)
% match a test segmentation to a set of ground-truth segmentations with the SEGMENTATION COVERING metric.
% based on PASCAL evaluation code:
% http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2010/index.html#devkit

% contains total number of groud truth segmentation labels over all
% ground truth segmentations
gt = groundTruth;

% max(seg(:)) is the number of labels of the segmentation
matches = zeros(max(gt(:)), max(seg(:)));

% go through all groud truth segmentations

% number of ground truth labels and segmentation labels of "seg"
num1 = max(gt(:)) + 1;
num2 = max(seg(:)) + 1;
confcounts = zeros(num1, num2);

% creates a matrix of labels as follows: for each label from 1 to num1
% in gt there are num2 "sublabels"
sumim = 1 + gt + seg*num1;

% histc computes a simple histogram counting the number of values in
% sumim to fall in the ranges given by 1:num1*num2
hs = histc(sumim(:), 1:num1*num2);

% confcounts is a mtrix of size num1 x num2 and in entry (i,j) it
% stores the number of pixels belonging to both label i-1 in ground
% truth and label j-1 in given segmentation
confcounts(:) = confcounts(:) + hs(:);

% in entry (j,i), accuracies will contain the number of pixels lying
% in the intersection of label j and i (j ground truth label; i
% segmentation label) divided by the total area of both labels
accuracies = zeros(num1-1, num2-1);
for j = 1:num1
    for i = 1:num2
        gtj = sum(confcounts(j, :));
        resj = sum(confcounts(:, i));
        gtjresj = confcounts(j, i);
        accuracies(j, i) = gtjresj / (gtj + resj - gtjresj);
    end
end

% note that MatLab is 1-based so we need to add one to cnt
% first row and first column will be zeros
matches(1:max(gt(:)), :) = accuracies(2:end, 2:end);
matches = matches';




