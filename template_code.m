%%generate barcode
generate_barcode;

%%
S = length(patterns);
patternLengths = zeros(S,1);
for s = 1:S
    patternLengths(s) = length(patterns{s});
end

M = 6;
C = max(patternLengths);
NumStates = C*S*M;

%Enumerate all the states
States = zeros(NumStates,3) -1 ;
StatesInv = zeros(C,S,M) -1;

ix = 1;
for c = 1:C
    for s = 1:S
        for m = 1:M
            if( (s > 5) || (s <=5 && m ==1) )
                if( c<=patternLengths(s))
                    States(ix,:) = [c s m];
                    StatesInv(c,s,m) = ix;
                    ix = ix+1;
                end
            end
        end
    end
end

NumStates = ix -1;
States = States(1:NumStates,:);

%warning: NumStates will be less than S*M*C, because not all possible
%[s,m,c] triples are valid. 

%% Part1: Fill the transition matrix A

%mapping states to binary numbers, which will be useful for computing the
%likelihood
f_kst = zeros(NumStates,1); % the mapping function : returns 0 if white, 1 if black
A = zeros(NumStates); % transition matrix

for i = 1:NumStates
    
    c = States(i,1);
    s = States(i,2);
    m = States(i,3);
    
    patternLen = patternLengths(s);
    f_kst(i) = patterns{s}(c); %determines if this state is black or white
    
    if(s == 1) %starting quiet zone
        if(c == patternLen)
            for ss = [1 3] %the next states can only be either starting quiet zone, or the starting guard
                s_next = ss;
                c_next = 1;
                m_next = 1;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 0.5;
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
        
    elseif(s == 2) %ending quiet zone
        if (c == patternLen)
            s_next = s;
            c_next = 1;
            m_next = 1;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
    elseif(s== 3) %starting guard
        if (c == patternLen)
            for ss= 6:15
                s_next = ss;
                c_next = 1;
                m_next = m;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 0.1;
            end
        else
            c_next =c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
    elseif(s== 4) %ending guard
        
        if (c== patternLen)
            s_next = 2;
            c_next = 1;
            m_next = 1;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        else
            c_next =c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
    elseif(s== 5) %middle guard
        
        if (c == patternLen)
            for ss=16:25
                s_next = ss;
                c_next = 1;
                m_next = m;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 0.1;
            end
        else
            c_next =c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1; 
        end
        
    elseif(s>= 6 && s<=15) %left symbols
        
        if (m~=6)
            if(c==patternLen)
                for ss=6:15
                    s_next = ss;
                    c_next = 1;
                    m_next = m+1;
                    
                    nextStateIx = StatesInv(c_next,s_next,m_next);
                    A(nextStateIx,i) = 0.1;
                end
            else
                c_next = c+1;
                s_next = s;
                m_next = m;

                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
        else
            if (c==patternLen)
                s_next = 5;
                c_next = 1;
                m_next = 1;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            else
                s_next = s;
                c_next =c+1;
                m_next = m;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
        end
        
        elseif(s>= 16 && s<=25) %right symbols
            
            if (m~=6)
                if(c==patternLen)
                    for ss=16:25
                        s_next = ss;
                        c_next = 1;
                        m_next = m+1;
                        
                        nextStateIx = StatesInv(c_next,s_next,m_next);
                        A(nextStateIx,i) = 0.1;
                    end
                else
                    c_next = c+1;
                    s_next = s;
                    m_next = m;

                    nextStateIx = StatesInv(c_next,s_next,m_next);
                    A(nextStateIx,i) = 1;
                end
            else
                if (c==patternLen)
                    s_next = 4;
                    c_next = 1;
                    m_next = 1;
                    
                    nextStateIx = StatesInv(c_next,s_next,m_next);
                    A(nextStateIx,i) = 1;
                else
                    s_next = s;
                    c_next = c+1;
                    m_next = m;
                    
                    nextStateIx = StatesInv(c_next,s_next,m_next);
                    A(nextStateIx,i) = 1;
                end
            end
        
    else
        error('Unknown State!');
    end
end

%% Question 3 : Simulated observations 
T = length(obs);

mu0 = 250;
mu1 = 20;
sigma0 = sqrt(5);
sigma1 = sqrt(5);

psi = zeros(3, T); % matrix containing the [c s m] for each observation
x_n = zeros(1, T); % simulated observation vector 

p_init = zeros(NumStates,1);
p_init(StatesInv(1,1,1)) = 1; % initial prior
i = 0;
for t = 1:T 
    if t == 1
        ix = randsample(NumStates,1,true,p_init); % generate a state according to p_init
        psi(:,t) = States(ix,:); % get corresponding values of [c s m]
        i = f_kst(ix); % get the the color of the bar
    end
    if t ~=1
        c = psi(1,t-1);
        s = psi(2,t-1);
        m = psi(3,t-1);
        ix = StatesInv(c,s,m);
        p_ix = A(:,ix); % get the probability of state ix from the transition matrix
        ix_new = randsample(NumStates,1,true,p_ix);  % generate a state according to p_init
        psi(:,t) = States(ix_new,:); % get corresponding values of [c s m]
        i = f_kst(ix_new); % 1 for black 0 for white
    end
    if i == 0
        x_n(t) = normrnd(mu0,sigma0); % generate a random normal variable 
    end
    if i == 1
        x_n(t) = normrnd(mu1,sigma1); % generate a random normal variable
    end
end

bc_image = uint8((repmat(x_n, [100 1])));
imshow(bc_image)
title('Simulated bare code', 'Interpreter', 'latex');
figure;

plot(x_n, '-');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_n$', 'Interpreter', 'latex');
title('Observation $x_n$', 'Interpreter', 'latex');
figure;

plot(psi(1,:), '-');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$c_n$', 'Interpreter', 'latex');
title('$c_n$', 'Interpreter', 'latex');
figure;

plot(psi(2,:), '-');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$s_n$', 'Interpreter', 'latex');
title('$s_n$', 'Interpreter', 'latex');
figure;

plot(psi(3,:), '-');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$m_n$', 'Interpreter', 'latex');
title('$m_n$', 'Interpreter', 'latex');

%% Part2: Compute the inital probability

%the barcode *must* start with the "starting quite zone", with s_n=1. Other
%states are not possible. Fill the initial probability accordingly. 
p_init = zeros(NumStates,1);
p_init(StatesInv(1,1,1)) = 1;

%% Part3: Compute the log-likelihood
T = length(obs);
logObs = zeros(NumStates,T);
mu0 = 255;
mu1 = 0;
sigma0 = 1;
sigma1 = 1;


for t=1:T
    for i = 1:NumStates
        c = States(i,1);
        s = States(i,2);
        m = States(i,3);
        ix = f_kst(i); % 1 for black 0 for white
        x = obs(t);
        if ix == 1
            % compute the log of the normal distribution for mu0 and sigma0
            logObs(i,t) = -log(sigma0) - 1/2 * log(2*pi) - (x-mu0)^2 / 2 * sigma0^2;

        end
        if ix == 0
            % compute the log of the normal distribution for mu1 and sigma1
            logObs(i,t) = -log(sigma1) - 1/2 * log(2*pi) - (x-mu1)^2 / 2 * sigma1^2;
        end
    end
end


%% Part 4: Compute the filtering distribution via Forward recursion

log_alpha = zeros(NumStates, T);
log_alpha_predict = zeros(NumStates, T);

for t=1:T 
    if t==1
        log_alpha_predict(:,t) = log(p_init + eps);
    else
        log_alpha_predict(:,t) = state_predict(A, log_alpha(:, t-1));
    end
    log_alpha(:, t) = state_update(logObs(:,t), log_alpha_predict(:,t));
end

alpha = normalize_exp(log_alpha);

%%%%%%%%%%%%%Compute the marginals from the obtained filternig distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%
list_c = zeros(1,T);
list_s = zeros(1,T);
list_m = zeros(1,T);

for t=1:T
    ix = randsample(NumStates,1,true,alpha(:,t));
    list_c(t) = States(ix,1);
    list_s(t) = States(ix,2);
    list_m(t) = States(ix,3);
end

%%%%%%%%%%%%%%%Plot the graphs %%%%%%%%%%%%%%%%%
% plot(list_c, '-');
% xlabel('$n$', 'Interpreter', 'latex');
% ylabel('$c_n$', 'Interpreter', 'latex');
% title('$c_n~using~the~filtering~$', 'Interpreter', 'latex');
% figure;
% 
% plot(list_s, '-');
% xlabel('$n$', 'Interpreter', 'latex');
% ylabel('$s_n$', 'Interpreter', 'latex');
% title('$s_n~using~the~filtering~$', 'Interpreter', 'latex');
% figure;
% 
% plot(list_m, '-');
% xlabel('$n$', 'Interpreter', 'latex');
% ylabel('$m_n$', 'Interpreter', 'latex');
% title('$m_n~using~the~filtering~$', 'Interpreter', 'latex');


%% Part 5: Compute the smoothing distribution via Forward-Backward recursion

log_beta = zeros(NumStates, T);
log_beta_postdict = zeros(NumStates, T);
for t=T:-1:1
    if t==T
        log_beta_postdict(:,t) = zeros(NumStates,1);
    else
    log_beta_postdict(:,t) = state_postdict(A, log_beta(:, t+1));
    end
    log_beta(:, t) = state_update(logObs(:,t), log_beta_postdict(:,t));
end

log_gamma = log_alpha + log_beta_postdict;
gamma = normalize_exp(log_gamma);

%%%%%%%%%%Test if the result is OK by using log_sum_exp, we should find a
%%%%%%%%%%constant vector 
test_gamma = log_sum_exp(log_gamma);
%%%%%%%%%%%%%Compute the marginals from the obtained smoothing distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%
list_c_1 = zeros(1,T);
list_s_1 = zeros(1,T);
list_m_1 = zeros(1,T);

for t=1:T
    ix = randsample(NumStates,1,true,alpha(:,t));
    list_c_1(t) = States(ix,1);
    list_s_1(t) = States(ix,2);
    list_m_1(t) = States(ix,3);
end

% %%%%%%%%%%%%%%Plot the graphs %%%%%%%%%%%%%%%%%
% plot(list_c_1, '-');
% xlabel('$n$', 'Interpreter', 'latex');
% ylabel('$c_n$', 'Interpreter', 'latex');
% title('$c_n~using~the~Smoothing~$', 'Interpreter', 'latex');
% figure;
% 
% plot(list_s_1, '-');
% xlabel('$n$', 'Interpreter', 'latex');
% ylabel('$s_n$', 'Interpreter', 'latex');
% title('$s_n~using~the~Smoothing~$', 'Interpreter', 'latex');
% figure;
% 
% plot(list_m_1, '-');
% xlabel('$n$', 'Interpreter', 'latex');
% ylabel('$m_n$', 'Interpreter', 'latex');
% title('$m_n~using~the~Smoothing~$', 'Interpreter', 'latex');



%% Part 6: Compute the most-likely path via Viterbi algorithm

%%%%%We used here the pseudo-code provided by wikipedia %%%%%

T1 = zeros(NumStates,T);
T2 = zeros(NumStates,T);
res = zeros(3,T);

for i=1:NumStates
    % Add eps for numerical instabilities
    T1(i,1) = log(p_init(i) + eps) + logObs(i,1);
    T2(i,1)= 0;
    
end

for i=2:T
    for j=1:NumStates
        
        val = T1(:,i-1) + log(A(j,:)') + logObs(j,i);
        [val_max , val_argmax] = max(val);
        T1(j,i) = val_max ;
        T2(j,i)= val_argmax ;
        
    end
end

[val_z , z] = max(T1(:,T));
res(:,T) = States(z,:);

for i=T:-1:2
    z = T2(z,i);
    res(:,i-1) = States(z,:);
end

%% Part 7: Obtain the barcode string from the decoded states

best_cn = res(1,:); %(obtained via Viterbi)
best_sn = res(2,:); %(obtained via Viterbi)
best_mn = res(3,:); %(obtained via Viterbi)
%find the place where a new symbol starts
ix = find(best_cn ==1);

s_ix = best_sn(ix); % find the symbols of corresponding indexes
decoded_code = [];

for i = 1:length(s_ix)
    tmp = s_ix(i);
    %consider only the symbols that correspond to digits
    if(tmp>=6)
        chr = mod(tmp-6,10);
        decoded_code = [decoded_code, chr];
    end
    
end

fprintf('Real code:\t');
fprintf('%d',code);
fprintf('\n');
fprintf('Decoded code:\t');
fprintf('%d',decoded_code);
fprintf('\n');


%%%%%%%%%%%%%%%Visualize the most likely path %%%%%%%%%%%%%%%%%%%%%
% plot(best_cn, '-');
% xlabel('$n$', 'Interpreter', 'latex');
% ylabel('$c_n$', 'Interpreter', 'latex');
% title('$c_n~by~ viterbi$', 'Interpreter', 'latex');
% figure;
% 
% plot(best_sn, '-');
% xlabel('$n$', 'Interpreter', 'latex');
% ylabel('$s_n$', 'Interpreter', 'latex');
% title('$s_n ~by ~viterbi$', 'Interpreter', 'latex');
% figure;
% 
% plot(best_mn, '-');
% xlabel('$n$', 'Interpreter', 'latex');
% ylabel('$m_n$', 'Interpreter', 'latex');
% title('$m_n~by~viterbi$', 'Interpreter', 'latex');
