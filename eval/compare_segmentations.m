%A MATLAB Toolbox
%
%Compare two segmentation results using
%1. Probabilistic Rand Index
%2. Variation of Information
%3. Global Consistency Error
%
%IMPORTANT: The two input images must have the same size!
%
%Authors: John Wright, and Allen Y. Yang
%Contact: Allen Y. Yang <yang@eecs.berkeley.edu>
%
%(c) Copyright. University of California, Berkeley. 2007.
%
%Notice: The packages should NOT be used for any commercial purposes
%without direct consent of their author(s). The authors are not responsible
%for any potential property loss or damage caused directly or indirectly by the usage of the software.

function [ri,gce,vi]=compare_segmentations(sampleLabels1,sampleLabels2)

% compare_segmentations
%
%   Computes several simple segmentation benchmarks. Written for use with
%   images, but works for generic segmentation as well (i.e. if the
%   sampleLabels inputs are just lists of labels, rather than rectangular
%   arrays).
%
%   The measures:
%       Rand Index
%       Global Consistency Error
%       Variation of Information
%
%   The Rand Index can be easily extended to the Probabilistic Rand Index
%   by averaging the result across all human segmentations of a given
%   image: 
%       PRI = 1/K sum_1^K RI( seg, humanSeg_K ).
%   With a little more work, this can also be extended to the Normalized
%   PRI. 
%
%   Inputs:
%       sampleLabels1 - n x m array whose entries are integers between 1
%                       and K1
%       sampleLabels2 - n x m (sample size as sampleLabels1) array whose
%                       entries are integers between 1 and K2 (not
%                       necessarily the same as K1).
%   Outputs:
%       ri  - Rand Index
%       gce - Global Consistency Error
%       vi  - Variation of Information
%
%   NOTE:
%       There are a few formulas here that look less-straightforward (e.g.
%       the log2_quotient function). These exist to handle corner cases
%       where some of the groups are empty, and quantities like 0 *
%       log(0/0) arise...
%
%   Oct. 2006 
%       Questions? John Wright - jnwright@uiuc.edu

[imWidth,imHeight]=size(sampleLabels1);
[imWidth2,imHeight2]=size(sampleLabels2);
N=imWidth*imHeight;
if (imWidth~=imWidth2)||(imHeight~=imHeight2)
    disp( 'Input sizes: ' );
    disp( size(sampleLabels1) );
    disp( size(sampleLabels2) );
    error('Input sizes do not match in compare_segmentations.m');
end;

% note:¡¡for label and pred 0 means non-plane
% make the group indices start at 1
if min(min(sampleLabels1)) < 1
    sampleLabels1 = sampleLabels1 - min(min(sampleLabels1)) + 1;
end
if min(min(sampleLabels2)) < 1
    sampleLabels2 = sampleLabels2 - min(min(sampleLabels2)) + 1;
end

segmentcount1=max(max(sampleLabels1)); %class # in seg_1
segmentcount2=max(max(sampleLabels2)); %class # in seg_2

% compute the count matrix
%  from this we can quickly compute rand index, GCE, VOI, ect...
n=zeros(segmentcount1,segmentcount2);

for i=1:imWidth
    for j=1:imHeight
        u=sampleLabels1(i,j);
        v=sampleLabels2(i,j);
        n(u,v)=n(u,v)+1;
    end;
end;

ri = rand_index(n);
gce = global_consistancy_error(n);
vi = variation_of_information(n);

return;


% the performance measures

% the rand index, in [0,1] ... higher => better
%  fast computation is based on equation (2.2) of Rand's paper.
function ri = rand_index(n)
N = sum(sum(n)); % pixel #
n_u=sum(n,2);
n_v=sum(n,1);
N_choose_2=N*(N-1)/2;
ri = 1 - ( sum(n_u .* n_u)/2 + sum(n_v .* n_v)/2 - sum(sum(n.*n)) )/N_choose_2;

% global consistancy error (from BSDS ICCV 01 paper) ... lower => better
function gce = global_consistancy_error(n)
N = sum(sum(n));
marginal_1 = sum(n,2);
marginal_2 = sum(n,1);
% the hackery is to protect against cases where some of the marginals are
% zero (should never happen, but seems to...)
E1 = 1 - sum( sum(n.*n,2) ./ (marginal_1 + (marginal_1 == 0)) ) / N;
E2 = 1 - sum( sum(n.*n,1) ./ (marginal_2 + (marginal_2 == 0)) ) / N;
gce = min( E1, E2 );


% variation of information a "distance", in (0,vi_max] ... lower => better
function vi = variation_of_information(n)
N = sum(sum(n));
joint = n / N; % the joint pmf of the two labels
marginal_2 = sum(joint,1);  % row vector
marginal_1 = sum(joint,2);  % column vector
H1 = - sum( marginal_1 .* log2(marginal_1 + (marginal_1 == 0) ) ); % entropy of the first label
H2 = - sum( marginal_2 .* log2(marginal_2 + (marginal_2 == 0) ) ); % entropy of the second label
MI = sum(sum( joint .* log2_quotient( joint, marginal_1*marginal_2 )  )); % mutual information
vi = H1 + H2 - 2 * MI; 

% log2_quotient 
%   helper function for computing the mutual information
%   returns a matrix whose ij entry is
%     log2( a_ij / b_ij )   if a_ij, b_ij > 0
%     0                     if a_ij is 0
%     log2( a_ij + 1 )      if a_ij > 0 but b_ij = 0 (this behavior should
%                                         not be encountered in practice!
function lq = log2_quotient( A, B )
lq = log2( (A + ((A==0).*B) + (B==0)) ./ (B + (B==0)) );