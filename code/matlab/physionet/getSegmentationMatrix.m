function A = getSegmentationMatrix(assigned_states, PCG)
%
% This function constructs a segmentation matrix based on the assigned_states by running "runSpringerSegmentationAlgorithm.m" function
%
% INPUTS:
% assigned_states: the array of state values assigned to the sound recording.
% PCG: resampled sound recording with 1000 Hz.
%
% OUTPUTS:
% A: the segmentation matrix where where column values represent the beginnings of heart sound phases
% col1 = S1, col2 = systole, col3 = S2, col4 = diastole
%
%
% Written by: Chengyu Liu, January 22 2016
%             chengyu.liu@emory.edu
%
% Last modified by: Jonathan Rubin, March 17th 2016
%                   Jonathan.Rubin@parc.com
%

%% We just assume that the assigned_states cover at least 2 whole heart beat cycle
indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
    switch assigned_states(1)
        case 4
            K=1;
        case 3
            K=2;
        case 2
            K=3;
        case 1
            K=4;
    end
else
    switch assigned_states(indx(1)+1)
        case 4
            K=1;
        case 3
            K=2;
        case 2
            K=3;
        case 1
            K=0;
    end
    K=K+1;
end

indx2                = indx(K:end);
rem                  = mod(length(indx2),4);
indx2(end-rem+1:end) = [];
A                    = reshape(indx2,4,length(indx2)/4)'; % A is N*4 matrix, the 4 columns save the beginnings of S1, systole, S2 and diastole in the same heart cycle respectively
