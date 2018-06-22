% Version 1.000 
%
% Code provided by Graham Taylor, Geoff Hinton and Sam Roweis 
%
% For more information, see:
%     http://www.cs.toronto.edu/~gwtaylor/publications/nips2006mhmublv
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program uses the 2-level CRBM to generate data

% The program assumes that the following variables are set externally:
% numframes    -- number of frames to generate
% fr           -- a starting frame from initdata (for initialization)

numGibbs = 30; %number of alternating Gibbs iterations

if fr<= n1
    error('Choose fr > n1 (order of first layer)');
end

%We have saved some initialization data in "initdata"
%Pass this through the network to get the first hidden layer

numcases = size(initdata,1)-n1; %number of hidden units we can generate
numdims = size(initdata,2);

data = zeros(numcases,numdims,n1+1); %current and past data
dataindex = n1+1:size(initdata,1);   %indexes the valid "starting" frames

data(:,:,1) = initdata(dataindex,:); %store current data
%store delayed data
for hh=1:n1
    data(:,:,hh+1) = initdata(dataindex-hh,:);
end

%Calculate contributions from directed autoregressive connections
bistar = zeros(numdims,numcases);
for hh=1:n1
    bistar = bistar +  A1(:,:,hh)*data(:,:,hh+1)' ;
end

%Calculate contributions from directed visible-to-hidden connections
bjstar = zeros(numhid1,numcases);
for hh = 1:n1
    bjstar = bjstar + B1(:,:,hh)*data(:,:,hh+1)';
end

%Calculate "posterior" probability -- hidden state being on
%Note that it isn't a true posterior
eta =  w1*(data(:,:,1)./gsd)' + ...   %bottom-up connections
    repmat(bj1, 1, numcases) + ...    %static biases on unit
    bjstar;                           %dynamic biases

hposteriors = 1./(1 + exp(-eta));     %logistic


%Now we have our next level up

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we have our visible layer, and first hidden layer
% for the initialization frames
% How much of this data do we need clamped?
% We need to hold max(n1,n2) clamped 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_clamped = max(n1,n2);

%initialize first layer; store clamped frames
visible = zeros(numframes,numdims);
visible(1:max_clamped,:) = initdata(fr:fr+max_clamped-1,:);
%initialize second layer (note the offset by n1)
hidden1 = ones(numframes,numhid1);
hidden1(1:max_clamped,:) = hposteriors(:,fr-n1:fr-n1+max_clamped-1)';
%initialize the top layer
hidden2 = ones(numframes,numhid2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%First generate a hidden sequence (top layer)
%Then go down through the first CRBM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Generating hidden states\n');
for tt=max_clamped+1:numframes  
  %initialize using the last frame + noise
  hidden1(tt,:) = hidden1(tt-1,:) + 0.01*rand(1,numhid1);
  
  %Dynamic biases aren't re-calculated during Alternating Gibbs
  %First, add contributions from autoregressive connections
  bistar = zeros(numhid1,1);  
  for hh=1:n2    
    bistar = bistar +  A2(:,:,hh)*hidden1(tt-hh,:)' ;
  end
  %Next, add contributions to hidden units from previous time steps
  bjstar = zeros(numhid2,1);
  for hh = 1:n2
    bjstar = bjstar + B2(:,:,hh)*hidden1(tt-hh,:)';
  end

  %Gibbs sampling
  for gg = 1:numGibbs
    %Calculate posterior probability -- hidden state being on (estimate)
    %add in bias
    bottomup =  w2*hidden1(tt,:)';
    eta = bottomup + ...                   %bottom-up connections
      bj2 + ...                            %static biases on unit
      bjstar;                              %dynamic biases
    
    hposteriors = 1./(1 + exp(-eta));      %logistic
    
    hidden2(tt,:) = hposteriors' > rand(1,numhid2); %Activating hiddens
    
    %Downward pass; visibles are binary logistic units     
    topdown = hidden2(tt,:)*w2;
        
    eta = topdown + ...                      %top down connections
      bi2' + ...                             %static biases
      bistar';                               %dynamic biases   
    
    hidden1(tt,:) = 1./(1 + exp(-eta));      %logistic 
  
  end

  %If we are done Gibbs sampling, then do a mean-field sample
  
  topdown = hposteriors'*w2;  %Very noisy if we don't do mean-field here

  eta = topdown + ...                        %top down connections
      bi2' + ...                             %static biases
      bistar';                               %dynamic biases
  hidden1(tt,:) = 1./(1 + exp(-eta));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Now that we've decided on the "filtering distribution", generate visible
%data through CRBM 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Generating visible data\n');
for tt=max_clamped+1:numframes
    %Add contributions from autoregressive connections
    bistar = zeros(numdims,1);
    for hh=1:n1
        bistar = bistar +  A1(:,:,hh)*visible(tt-hh,:)' ;
    end

    %Mean-field approx; visible units are Gaussian
    %(filtering distribution is the data we just generated)
    topdown = gsd.*(hidden1(tt,:)*w1);
    visible(tt,:) = topdown + ...             %top down connections
        bi1' + ...                            %static biases
        bistar';                              %dynamic biases
end
