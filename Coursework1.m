%%% Machine learning and Neural computation
%%% Coursework 1
%%% by Ahmet Narman
%%% CID: 01578741
%%% Instructor: Prof Aldo Faisal
%%% Imperial College London
%%% Nov 2018

clear all;
close all;

RunCoursework();
    
% The function that includes the coursework
function RunCoursework()
    %% Part 1
    
    CID = 01578741;
    p = 0.5 + 0.5*(4/10); % Probability to be used in the transition matrix
    gama = 0.2 + 0.5*(1/10); % The discount value

    %% Part 2
    
    [NumSt, NumAct, TrMat,RewMat, StName, AcName, AbsSt] ...
    = PersonalisedGridWorld(p);
    % For the reward and transition matrices the rows (first indice) are 
    % successor states, the columns (second indice) are current states, 
    % and the third indice is the taken action.

    % Unbiased policy function is denoted with the below matrix. This
    % matrix has the size of (NumActions, NumStates) because a policy is a
    % function of state,action pairs
    unbias = ones(NumAct, NumSt)*(1/4); % Unbiased Policy

    % Value function for the unbiased policy is found below
    Value = find_value(TrMat, RewMat, AbsSt, gama, unbias)

    %% Part 3
    
    % Part (a)
    % Sequences that were given to us
    sequence1 = [14 10 8 4 3];
    sequence2 = [11 9 5 6 6 2];
    sequence3 = [12 11 11 9 5 9 5 1 2];

    % The probabilities of these sequences for an unbiased policy
    unbiased_prob_seq1 = trace_probability(TrMat, unbias, sequence1)
    unbiased_prob_seq2 = trace_probability(TrMat, unbias, sequence2)
    unbiased_prob_seq3 = trace_probability(TrMat, unbias, sequence3)

    %Part (b)
    % Our function will modifiy the unbiased policy to improve the
    % probabilities of the three sequences given above. The policy update
    % is done three times, once for every sequence
    biased = give_bias(TrMat, unbias, sequence1);
    biased = give_bias(TrMat, biased, sequence2);
    biased = give_bias(TrMat, biased, sequence3);

    % The probabilities of these sequences for our biased policy
    biased_prob_seq1 = trace_probability(TrMat, biased, sequence1)
    biased_prob_seq2 = trace_probability(TrMat, biased, sequence2)
    biased_prob_seq3 = trace_probability(TrMat, biased, sequence3)

    %% Part 4
    
    % Part (a)
    Traces = cell(1,10); % 10 traces will be put in this cell array 
    for i = 1:length(Traces)
        % Below function will generate a trace given the parameters
        trace=generate_trace(TrMat, RewMat, unbias, StName, AcName, AbsSt);
        fprintf('%s,',upper(string(trace))); % Printing individual traces
        fprintf('\n');
        Traces(i) = {trace}; % Adding traces to the array    
    end
    
    % Part (b)  
    % Below function is the estimated val. function for the unbiased policy
    MCvalue = policy_evaluation(Traces, gama) % Val. function calculated 
    
    % Part (c)
    % The difference between the real value function and the estimated
    % value function was calculated using root mean squared error (RMS)
    diff = zeros(1,length(Traces)); % Difference between value functions
    for i = 1:length(diff)
        newValue = policy_evaluation(Traces(1:i), gama); % MC val. function
        diff(i) = sqrt(sum((Value - newValue).^2)); % RMS Err. calculation
    end
    
    figure; % Plotting the difference values
    plot(diff);
    title('Difference between value functions of Q2 & Q4');
    xlabel('Number of Traces');
    ylabel('RMS Error');
    grid on;
    
%% Part 5
    

    numEp = 150; % Number of episodes in each trial
    numTri = 20; % Number of trials
    % Below matrices hold rewards and trace lengths for every episode in
    % every trial for eps=0.1 and eps=0.75 case.
    trcMat1 = zeros(numTri,numEp); 
    rewMat1 = zeros(numTri,numEp); 
    trcMat2 = zeros(numTri,numEp); 
    rewMat2 = zeros(numTri,numEp);
    
    % Because the learning process is stochastic, we are going to run the
    % learning process for multiple trials and take the average rewards and
    % trace lengths because of the variability between trials
    for i = 1:numTri
        % Two learning processes for eps=0.1 & eps=0.75
        [OptPol1,trc1,rew1] =...
            learn(numEp,gama,0.1,TrMat,RewMat,unbias,StName,AcName,AbsSt);
        [OptPol2,trc2,rew2] =...
            learn(numEp,gama,0.75,TrMat,RewMat,unbias,StName,AcName,AbsSt);
        trcMat1(i,:) = trc1;
        trcMat2(i,:) = trc2;
        rewMat1(i,:) = rew1;
        rewMat2(i,:) = rew2;
    end
    
    % Plotting the average rewards and trace lengths across episodes
    figure;
    subplot(1,4,1);
    plot(mean(rewMat1));
    hold on;
    plot(mean(rewMat1)+std(rewMat1), 'g--');
    plot(mean(rewMat1)-std(rewMat1), 'g--');
    title('Avg. cumulative reward(\epsilon = 0.1)');
    xlabel('Episodes in Trials');
    ylabel('Average Reward');
    grid on;
    
    subplot(1,4,2);
    plot(mean(trcMat1));
    hold on
    plot(mean(trcMat1)+std(trcMat1), 'g--');
    plot(mean(trcMat1)-std(trcMat1), 'g--');
    title('Aveg. trace length(\epsilon = 0.1)');
    xlabel('Episodes in Trials');
    ylabel('Average Trace Length');
    grid on;

    subplot(1,4,3);
    plot(mean(rewMat2));
    hold on;
    plot(mean(rewMat2)+std(rewMat2), 'g--');
    plot(mean(rewMat2)-std(rewMat2), 'g--');
    title('Avg. cumulative reward(\epsilon = 0.75)');
    xlabel('Episodes in Trials');
    ylabel('Average Reward');
    grid on;
   
    subplot(1,4,4);
    plot(mean(trcMat2));
    hold on
    plot(mean(trcMat2)+std(trcMat2), 'g--');
    plot(mean(trcMat2)-std(trcMat2), 'g--');
    title('Avg. trace length(\epsilon = 0.75)');
    xlabel('Episodes in Trials');
    ylabel('Average Trace Length');
    grid on;
end

%% Functions

function Value = find_value(T, R, absorbing, disc, policy)
    Value = zeros(14, 1); % Val. func. is immidiately all zero
    prevValue = zeros(14, 1); % To calculate the rate of change each loop
    irew = T.*R; % Immediate reward matrix
    stop = 0; % To stop the loop when the system converges

    while stop == 0

        invAbsorb=-1*(absorbing'-1);% Inverse of the absorbing array 

        % Implementing an intermediate variable: Successor Value
        succValue = cat(3, disc*T(:,:,1)*Value, disc*T(:,:,2)*Value,...
            disc*T(:,:,3)*Value, disc*T(:,:,4)*Value);

        % The value function is updated here for each action
        Value=policy(1,:)'.*(succValue(:,:,1) + sum((irew(:,:,1)))');
        Value=Value+ policy(2,:)'.*(succValue(:,:,2)+ sum((irew(:,:,2)))');
        Value=Value+ policy(3,:)'.*(succValue(:,:,3)+ sum((irew(:,:,3)))');
        Value=Value+ policy(4,:)'.*(succValue(:,:,4)+ sum((irew(:,:,4)))');

        Value = Value.*invAbsorb; % Eliminating the terminate state values

        diff = sum((Value - prevValue).^2); % MSE between two iterations
        prevValue = Value; % Updating the preValue for the next cycle

        if diff<0.00001 % stop if the iteration difference is low enough
            stop = 1;
        end    
    end
end

function prob = trace_probability(T, policy, seq)
    % The "policy" input should be a matrix of size [actions x states]. 
    % For every (state,action) pair, there is a corresponding probability.

    prob = 1; % Will be multiplied with state trans. probabilities
    
    for i = 1:length(seq)-1 % No need for calculation for the last state
       
        % We will find the probability of going from one state to the next
        % for every action and sum it, which will give the total
        % probability of going from seq(i) to seq(i+1)
        tProb = policy(1,seq(i))*T(seq(i+1),seq(i),1)+...
                policy(2,seq(i))*T(seq(i+1),seq(i),2)+...
                policy(3,seq(i))*T(seq(i+1),seq(i),3)+...
                policy(4,seq(i))*T(seq(i+1),seq(i),4);
        prob = prob*tProb; % Trace probability is updated here
    end
end

function posterior = give_bias(T, prior, seq)
    % This function will take a transition matrix, a prior policy, and a
    % sequence; and will return a posterior policy for which the given
    % sequence has a higher probability of occuring. The way it does this
    % is increasing the probability of the action that gives the maximum
    % probability of going from current state to the next state in the
    % given sequence.
    
    posterior = prior; % Posterior policy will be modified
    a = 1; % Constant to adjust the probability of choosing an action
    
    for i=1:length(seq)-1
        [M, I] = max(T(seq(i+1), seq(i),:)); % Finding the optimal action
        posterior(I,seq(i))=posterior(I,seq(i))+a; % Increasing its prob.
        posterior(:, seq(i)) = posterior(:, seq(i))/(a+1); % Normalizing
    end
end

function trace = generate_trace(T, R, pol, SN, AN, Absorb)
    trace = {}; % The trace will be stored in this cell aray
    
    % Finding the initial state randomly (using uniform probability)
    s = rand*4+11; % Random number for choosing the initial state
    state = fix(s); % Now it is a random integer between 11 and 14
    
    trace = [trace SN(state,:)]; % Adding the initial state to the trace
    
    terminate = 0; % Will terminate the loop if we reach an absorbing state
    while terminate == 0     
        % Individual action probabilities will determine how likely an
        % action will be chosen in the condition below
        x = rand; % To be used in the policy when choosing an action
        if x<pol(1,state) % p(1)
            action = 1;
        elseif x<pol(2,state)+pol(1,state) % p(1)+p(2)
            action = 2;
        elseif x<pol(3,state)+pol(2,state)+pol(1,state) % p(1)+p(2)+p(3)
            action = 3;
        else % If the probability is higher, it is the 4th action
            action = 4;
        end
        
        trace = [trace AN(action)]; % Adding the action to the trace
        
        % Below line will find the successor states that has nonzero
        % probabilities so we don't have to go over all states
        nonzeroP = find(T(:,state,action)); 
        
        % Below loop will choose a successor state based on the current
        % state, the taken action and the corresponding successor state
        % probabilities on the transition matrix.
        y = rand; % Random number to be used in the transition matrix
        prob = 0; 
        for i=1:length(nonzeroP)
            % Below probability is summed until finding the successor state
            prob = prob+T(nonzeroP(i),state,action); % Checking next prob.
            if y<prob % Checking the successor state probabilities
                postState = nonzeroP(i); % Successor state was found
                break % Breaking the loop, VERY IMPORTANT!
                % If you don't break the loop, this code will always give
                % the last state in 'nonzeroP'. You don't want this
            end
        end
        
        reward = R(postState, state, action); % Corresponding reward value
        trace = [trace reward SN(postState,:)]; % Updating the trace
        
        if Absorb(postState)==1 % If the absorbing state was reached
            terminate = 1; % Stop the trace
            trace = trace(1:length(trace)-1);
        end
        state = postState; % The current state in the next cycle   
    end   
end

function Value = policy_evaluation(traces, disc)
    % This function gets a list of traces (in 'cell' format) and the
    % discount value and returns the estimated value function for states
    % according to the First-Visit Batch Monte Carlo policy evaluation
    
    len=length(traces);
    Value = zeros(14,1);
    stateReturns = cell(14,1); %Cell array was used to append state returns
    
    for i=1:len % Work this loop for each TRACE
        visitedStates = []; % To implement the first visit MC in each trace
        tau = traces{i}; % The indiced trace, will be used in the next loop
        stLen = (length(tau))/3; % How many states are there in the trace
        rew = cell2mat(tau(3:3:length(tau))); % Reward array of the trace
        
        for j=1:stLen % Work this loop for each STATE in the trace
            state = str2num(traces{i}{3*j-2}(2:3));% Indice of the state
            
            if ismember(state, visitedStates)
                % If the state is visited before, do nothing (First-visit)
            else
                % If it is the first visit, calculate and append the return
                visitedStates = [visitedStates state];
                % Total discounted rewards is given below
                Ret=sum(rew(j:stLen).*(disc.^(0:length(rew(j:stLen))-1)));
                % Adding the reward to the return array
                stateReturns{state} = [stateReturns{state} Ret]; 
            end        
        end        
    end
    
    Value = cellfun(@mean, stateReturns); % Average return for each state
    
    % If a state has never been visited, the return array will be empty and
    % averaging in will give 'NaN'. Instead, we assign them zero
    x = find(isnan(Value));
    Value(x) = 0;
end

function [finalPolicy, TraceLen, Rewards] =...
    learn(NumIter, disc, eps, T, R, pol, SN, AN, Absorb) % Function start

    % This function implements the e-greedy first-visit Monte Carlo control
    % algorithm. It returns the policy at the end of the learning, the
    % trace length array that shows how trace length changed throughout
    % learning and the reward array for showing the change in rewards

    Returns = cell(length(AN), length(SN)); %Returns for state-action pairs
    Rewards = zeros(1,NumIter); % Reward for each episode
    TraceLen = zeros(1,NumIter); % Trace length for each episode
    Qfunc = zeros(length(AN),length(SN)); % State-Action Val. function
    policy = pol; % Policy to be updated when learning
    
    for i=1:NumIter
        % An episode is generated for each iteration
        episode = generate_trace(T, R, policy, SN, AN, Absorb); 
        SAnum = (length(episode))/3; % State-Action amount in an episode
        visitedSA = []; % Visited state-action pairs in each episode
        rew =cell2mat(episode(3:3:length(episode))); %Reward of the episode
        Rewards(i) = sum(rew);
        TraceLen(i) = (length(episode))/3; % Trace length added
        
        for j=1:SAnum
            s = str2num(episode{3*j-2}(2:3)); % Curent state
            a = episode{3*j-1}; % Current action in letters (N,E,S,W)
            a = AN==a;
            a = find(a); % Current action in number (1,2,3,4)
            
            % Below notation is used to utilize the ismember() function
            % Integer value is the state, decimal value is the action
            s_a = str2num(strcat(num2str(s), '.',num2str(a))); 
            
            if ismember(s_a, visitedSA)
                % If the state-action is visited before, do nothing
            else
                % If it is the first visit, calculate and append the return
                visitedSA = [visitedSA s_a]; % Updating the visited S-A
                % Total discounted rewards found below
                ret= sum(rew(j:SAnum).*(disc.^(0:length(rew(j:SAnum))-1))); 
                Returns{a,s} = [Returns{a,s} ret]; % Appending the return
            end  
        end
        
        Qfunc = cellfun(@mean, Returns); % Q(s,a) function updated
        x = find(isnan(Qfunc)); % To eliminate 'NaN' terms
        Qfunc(x) = 0;
        
        updatedState = []; % States for which the policy is updated
        for k=1:length(visitedSA)
            saTemp = visitedSA(k);
            sNew = fix(saTemp);
            if ismember(sNew, updatedState)
                % If the state policy is already updated, do nothing
            else
                % If not, implement eps-greedy algorithm
                updatedState = [updatedState sNew];
                [M, aStar] = max(Qfunc(:,sNew)); % The optimal action
                policy(:,sNew)= eps/length(AN); % Suboptimal actions
                policy(aStar, sNew) = policy(aStar, sNew)+1-eps;
            end
        end
    end
    finalPolicy = policy; % The policy that will be returned by the func.
end