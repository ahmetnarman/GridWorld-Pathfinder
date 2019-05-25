%%% Machine learning and Neural computation
%%% Coursework 1
%%% by Ahmet Narman
%%% CID: 01578741
%%% Instructor: Prof. Aldo Faisal
%%% Imperial College London
%%% Nov 2018

    clear all;

    %% Part 1
    
    CID = 01578741;
    p = 0.5 + 0.5*(4/10); % The probability that will be used in the transition matrix
    gama = 0.2 + 0.5*(1/10); % The discount value

    %% Part 2
    
    [NumSt, NumAct, TrMat,RewMat, StName, AcName, AbsSt] ...
    = PersonalisedGridWorld(p);
    % For the reward and transition matrices the rows (first indice) are the
    % successor states, the columns (second indice) are the current states, and
    % the third indice is the taken action.

    % Unbiased policy function is denoted with the below matrix. This
    % matrix has the size of (NumActions, NumStates) because a policy is a
    % function of state,action pairs
    unbiased = ones(NumAct, NumSt)*(1/4); % Every action has the same prob. of 0.25

    % Value function for the unbiased policy is found below
    Value = find_value(TrMat, RewMat, AbsSt, gama, unbiased)

    %% Part 3
    
    % Part (a)
    % Sequences that were given to us
    sequence1 = [14 10 8 4 3];
    sequence2 = [11 9 5 6 6 2];
    sequence3 = [12 11 11 9 5 9 5 1 2];

    % The probabilities of these sequences for an unbiased policy
    unbiased_prob_seq1 = trace_probability(TrMat, unbiased, sequence1)
    unbiased_prob_seq2 = trace_probability(TrMat, unbiased, sequence2)
    unbiased_prob_seq3 = trace_probability(TrMat, unbiased, sequence3)

    %Part (b)
    % Our function will modifiy the unbiased policy to improve the
    % probabilities of the three sequences given above. The policy update
    % is done three times, once for every sequence
    biased = give_bias(TrMat, unbiased, sequence1);
    biased = give_bias(TrMat, biased, sequence2);
    biased = give_bias(TrMat, biased, sequence3);

    % The probabilities of these sequences for our biased policy
    biased_prob_seq1 = trace_probability(TrMat, biased, sequence1)
    biased_prob_seq2 = trace_probability(TrMat, biased, sequence2)
    biased_prob_seq3 = trace_probability(TrMat, biased, sequence3)

    %% Part 4
    
    % Part (a)
    Traces=cell(1,10); % 10 traces will be put in this cell array (change 10 to change trace numbers)
    for i=1:length(Traces)
        % Below function will generate a trace given the parameters
        trace = generate_trace(TrMat, RewMat, unbiased, StName, AcName, AbsSt);
%         fprintf('%s,',upper(string(trace))); % Printing individual traces
%         fprintf('\n');
        Traces(i) = {trace}; % Adding traces to the array    
    end
    
    % Part (b)  
    MCvalue = policy_evaluation(Traces, gama) % Value function calculated by MC method
    
    % Part (c)
    diff = zeros(1,length(Traces)); % Difference between original value function and the MC value function
    for i=1:length(diff)
        newValue = policy_evaluation(Traces(1:i), gama); % MC value function
        diff(i) = sum((Value - newValue).^2); % Mean Squared Error was chosen to be the measure of choice
    end
    figure;
    plot(diff);
%% Part 5
    
    OptPol1 = unbiased; % The policy will be optimised starting from the unbiased policy
    OptPol2 = unbiased; % The policy will be optimised starting from the unbiased policy
    numEp = 100; % Number of episodes in each trial
    numTri = 30; % Number of trials
    trcMat1 = zeros(numTri,numEp);
    rewMat1 = zeros(numTri,numEp);
    trcMat2 = zeros(numTri,numEp);
    rewMat2 = zeros(numTri,numEp);
    for i=1:numTri
        [OptPol1,trc1,rew1] = learn(numEp, gama, 0.1, TrMat, RewMat, OptPol1, StName, AcName, AbsSt);
        [OptPol2,trc2,rew2] = learn(numEp, gama, 0.75, TrMat, RewMat, OptPol2, StName, AcName, AbsSt);
        trcMat1(i,:) = trc1;
        trcMat2(i,:) = trc2;
        rewMat1(i,:) = rew1;
        rewMat2(i,:) = rew2;
    end


%% Functions

function Value = find_value(T, R, absorbing, disc, policy)
    Value = zeros(14, 1); % Value function is immidiately all zero
    prevValue = zeros(14, 1); % prevValue to calculate the rate of change each loop
    irew = T.*R; % Immediate reward for every (state, action, successor state) combination
    stop = 0; % To stop the loop when the system converges

    while stop == 0

        invAbsorb=-1*(absorbing'-1);% Inverse of the absorbing array (0's and 1's are changed)

        % Implementing an intermediate variable: Successor Value
        succValue = cat(3, disc*T(:,:,1)*Value, disc*T(:,:,2)*Value,...
            disc*T(:,:,3)*Value, disc*T(:,:,4)*Value);

        % The value function is updated here
        Value = policy(1,:)'.*(succValue(:,:,1) + sum((irew(:,:,1)))'); % North
        Value = Value + policy(2,:)'.*(succValue(:,:,2) + sum((irew(:,:,2)))'); % East
        Value = Value + policy(3,:)'.*(succValue(:,:,3) + sum((irew(:,:,3)))'); % South
        Value = Value + policy(4,:)'.*(succValue(:,:,4) + sum((irew(:,:,4)))'); % West

        Value = Value.*invAbsorb; % Eliminating the terminate state values

        diff = sum((Value - prevValue).^2); % difference between two iterations
        prevValue = Value; % Updating the preValue for the next cycle

        if diff<0.00001 % stop if the iteration difference is low enough
            stop = 1;
        end    
    end
end

function prob = trace_probability(T, policy, seq)
    % The "policy" input should be a matrix of size [actions x states]. 
    % For every (state,action) pair, there is a corresponding probability.

    prob = 1; % Initial trace probablility, will be multiplied with state trans. probabilities
    
    for i = 1:length(seq)-1 % Last state is the terminal state, no need for calculation
       
        % We will find the probability of going from one state to the next
        % for every action and sum it, which will give the total
        % probability of going from seq(i) to seq(i+1)
        tProb = policy(1,seq(i))*T(seq(i+1),seq(i),1)+...
                policy(2,seq(i))*T(seq(i+1),seq(i),2)+...
                policy(3,seq(i))*T(seq(i+1),seq(i)+3)+...
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
    a = 1; % Constant to adjust how much the probability of choosing an action changes
    
    for i=1:length(seq)-1
        [M, I] = max(T(seq(i+1), seq(i),:)); % Finding the optimim action
        posterior(I, seq(i)) = posterior(I, seq(i))+a; % Increasing its probability
        posterior(:, seq(i)) = posterior(:, seq(i))/(a+1); % Normalizing probabilities
    end
end

function trace = generate_trace(T, R, pol, SN, AN, Absorb)
    trace = {}; % The trace will be stored in this cell aray
    
    % Finding the initial state randomly (using uniform probability)
    s = rand*4+11; % Random number to be used for choosing the initial state
    state = fix(s); % Now it is a random integer between 11 and 14
    
    trace = [trace SN(state,:)]; % Adding the initial state to the trace
    
    terminate = 0; % Will terminate the loop if an absorbing state was reached
    
    while terminate == 0     
        % Individual action probabilities will determine how likely an
        % action will be chosen in the condition below
        x = rand; % Random number to be used in the policy when choosing an action
        if x<pol(1,state) 
            action = 1;
        elseif x<pol(2,state)+pol(1,state)
            action = 2;
        elseif x<pol(3,state)+pol(2,state)+pol(1,state)
            action = 3;
        else
            action = 4;
        end
        
        trace = [trace AN(action)]; % Adding the action to the trace
        
        nonzeroP = find(T(:,state,action)); % State transitions with nonzero probabilities
        
        % Below loop will choose a successor state based on the current
        % state, the taken action and the corresponding successor state
        % probabilities on the transition matrix.
        y = rand; % Random number to be used in the transition matrix
        prob = 0; % Reseting the variable
        for i=1:length(nonzeroP)
            prob = prob+T(nonzeroP(i),state,action); % For checking the next probability
            if y<prob % Checking the successor state probabilities
                postState = nonzeroP(i); % Successor state was found
                break % Breaking the loop, VERY IMPORTANT!
                % If you don't break the loop, this code will always give
                % the last state in 'nonzeroP'. You don't want this
            end
        end
        
        reward = R(postState, state, action); % Corresponding reward value
        
        trace = [trace reward SN(postState,:)]; %updating the trace
        
        if Absorb(postState)==1 % If the absorbing state was reached
            terminate = 1; % Stop the trace
        end
        state = postState; % Successor state will be the current state in the next cycle   
    end   
end

function Value = policy_evaluation(traces, disc)
    % This function gets a list of traces (in 'cell' format) and the
    % discount value and returns the estimated value function for states
    % according to the First-Visit Batch Monte Carlo policy evaluation
    
    len=length(traces);
    Value = zeros(14,1);
    stateReturns = zeros(14,len); % First visit: one total return for each state in each trace
    
    for i=1:len % Work this loop for each TRACE
        visitedStates = []; % To implement the first visit MC in each trace
        tau = traces{i}; % The indiced trace, will be used in the following loop
        stLen = (length(tau)-1)/3; % How many states are there in the trace (excluding terminal)
        rew = cell2mat(tau(3:3:length(tau))); % Reward array of the current trace
        
        for j=1:stLen % Work this loo for each STATE in the trace
            state = str2num(traces{i}{3*j-2}(2:3));% The numeric indice of the current state
            
            if ismember(state, visitedStates)
                % If the state is visited before, do nothing (First-visit)
            else
                % If it is the first visit, calculate and append the return
                visitedStates = [visitedStates state];
                Return = sum(rew(j:stLen).*(disc.^(0:length(rew(j:stLen))-1))); % Total discounted rewards
                stateReturns(state,i) = Return; % Adding the reward to the return array
            end        
        end        
    end
    
    Value = mean(stateReturns')'; % Average return for each state
    
    % If a state has never been visited, the return array will be empty and
    % averaging in will give 'NaN'. Instead, we assign them zero
    x = find(isnan(Value));
    Value(x) = 0;
end

function [finalPolicy, TraceLen, Reward] = learn(Iter, disc, eps, T, R, pol, SN, AN, Absorb)
    Returns = zeros(length(AN), length(SN), Iter); % Return array for each state-action pair
    Reward = zeros(1,Iter); % Reward for each episode
    TraceLen = zeros(1,Iter);
    Qfunc = zeros(length(AN),length(SN));
    policy = pol;
    
    for i=1:Iter
        episode = generate_trace(T, R, policy, SN, AN, Absorb); % Generate an episode each iteration
        SAnum = (length(episode)-1)/3; % State-Action number (also reward number) in an episode
        visitedSA = []; % To record the visited state-action pairs in each episode
        rew = cell2mat(episode(3:3:length(episode))); % Reward array of the episode
        Reward(i) = sum(rew);
        TraceLen(i) = length(episode);
        
        for j=1:SAnum
            s = str2num(episode{3*j-2}(2:3)); % Curent state
            a = episode{3*j-1}; % Current action in letters (N,E,S,W)
            a = AN==a;
            a = find(a); % Current action in number (1,2,3,4)
            
            % Below notation is used because of the nature of the ismember() function
            % Integer value denotes the state, decimal value denotes the action
            s_a = str2num(strcat(num2str(s), '.',num2str(a))); % State-Action pair representation
            
            if ismember(s_a, visitedSA)
                % If the state-action is visited before, do nothing (First-visit)
            else
                % If it is the first visit, calculate and append the return
                visitedSA = [visitedSA s_a];
                ret = sum(rew(j:SAnum).*(disc.^(0:length(rew(j:SAnum))-1))); % Total discounted rewards
                Returns(a,s,i) = ret; % Adding the reward to the return array
            end  
        end
        Qfunc = mean(Returns, 3); % Q(s,a) function updated
        x = find(isnan(Qfunc)); % To eliminate 'NaN' terms
        Qfunc(x) = 0;
        
        updatedState = [];
        for k=1:length(visitedSA)
            saTemp = visitedSA(k);
            sNew = fix(saTemp);
            if ismember(s, updatedState)
                % If the state policy is already updated, do nothing
            else
                % If not, implement eps-greedy algorithm
                [M, aStar] = max(Qfunc(:,sNew)); % aStar is the optimum action for the state
                policy(:,sNew)= eps/length(AN); % Suboptimal actions
                policy(aStar, sNew) = policy(aStar, sNew)+1-eps; % Optimal action
            end
            
            
        end
        
    end
    finalPolicy = policy;
    
%     figure
%     plot(Reward);
%     figure
%     plot(TraceLen);
end