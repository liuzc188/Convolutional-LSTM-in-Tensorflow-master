function show_mocap_running(visible)


%load Data/data.mat
load Data/Jog1_M
%load Data/ilya_running.mat
skel.type = 'mit';
Motion = Walking;
dropframes;

fprintf(1,'Preprocessing data \n');

preprocess1

preprocess2ilya
numdims = size(batchdata,2); 


postprocess; 

fprintf(1,'Playing sequence\n');
figure(2); expPlayData(skel, newdata, 1/30)

