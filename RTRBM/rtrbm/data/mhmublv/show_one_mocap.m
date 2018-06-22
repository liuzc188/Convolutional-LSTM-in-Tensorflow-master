function show_one_mocap(visible, frame_no)
global skel

load Data/data.mat
dropframes;

fprintf(1,'Preprocessing data \n');

preprocess1

preprocess2ilya
numdims = size(batchdata,2); %data (visible) dimension


postprocess; 

fprintf(1,'Playing sequence\n');
figure(2); expPlayDataOne(frame_no, skel, newdata, 1/30)

