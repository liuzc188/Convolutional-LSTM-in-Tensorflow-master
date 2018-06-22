% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This is the "main" demo
% It trains two CRBM models, one on top of the other, and then
% demonstrates data generation

%clear all; close all;
more off;   %turn off paging

%initialize RAND,RANDN to a different state
rand('state',sum(100*clock))
randn('state',sum(100*clock))

%Our important Motion routines are in a subdirectory
addpath('./Motion')

%Load the supplied training data
%Motion is a cell array containing 3 sequences of walking motion (120fps)
%skel is struct array which describes the person's joint hierarchy
load Data/data.mat

%Downsample (to 30 fps) simply by dropping frames
%We really should low-pass-filter before dropping frames
%See Matlab's DECIMATE function
dropframes;

fprintf(1,'Preprocessing data \n');

%Run the 1st stage of pre-processing
%This converts to body-centered coordinates, and converts to ground-plane
%differences
preprocess1

%Run the 2nd stage of pre-processing
%This drops the zero/constant dimensions and builds mini-batches
preprocess2ilya
numdims = size(batchdata,2); %data (visible) dimension

%Now batchdata contains all the processed sequences in matrix format
%Seqindex is a cell array which provides indices to sequences
%Here's an example of plotting the sequences
figure(3);
subplot(3,1,1); plot(batchdata(seqindex{1},:))
subplot(3,1,2); plot(batchdata(seqindex{2},:))
subplot(3,1,3); plot(batchdata(seqindex{3},:))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here you can do your learning on "batchdata", and use      %
% seqindex to form minibatches, etc without mixing sequences %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Here's an example of going back to the originable (viewable)
%representation and then displaying the motion
visible = batchdata(seqindex{1},:);

%Note postprocess assumes the data you want to plot is in "visible"
postprocess; 

%It creates "newdata" which is in the original representation
fprintf(1,'Playing sequence\n');
figure(2); expPlayData(skel, newdata, 1/30)