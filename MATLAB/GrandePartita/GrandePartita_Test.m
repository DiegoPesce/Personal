%% Params
clear 

max_valore = 100;

guerrieri = randi( [0, max_valore] , [1,30] );
guerrieri = [guerrieri, 100, 0]; % per la scala ci sono sempre 0 e 100
guerrieri = sort( guerrieri, 'descend' );

guerrieri_per_sq = 5;
Npartite = 1000;
max_sum = max_valore*guerrieri_per_sq;

%% Generate training-validation data
rng shuffle

for i = 1:Npartite

    sq1 = sort( randsample(guerrieri, guerrieri_per_sq, false), 'descend' );
    sq2 = sort( randsample(guerrieri, guerrieri_per_sq, false), 'descend' );

    diff = sum(sq1)-sum(sq2); % belongs to [-500, 500]

    % probability that strongest team wins
    p = 0.5*abs( diff/max_sum ) + 0.5; % belongs to 1/2 and 1
    rnd = rand();

    if diff > 0 % se squadra1 piu forte di squadra2
        % diff > 0 grande vuol dire che PROBABILMENTE vince sq1
        match(:,i) = [sq1(:); sq2(:); rnd > p; p; 1-p];
    else % se squadra2 > squadra1
        % diff < 0 means that sq2 probably wins
        match(:,i) = [sq1(:); sq2(:); rnd < p; 1-p; p];
    end
end

clear diff i rnd p sq1 sq2 

%%

save('GrandiPartite_train_validation', 'match')

%% See data

%first two rows are sum of teams' points, third one is the result, lasts
%are probs
WATCH_ME = [
    sum(match( 1:guerrieri_per_sq,:), 1 ); 
    sum(match( (guerrieri_per_sq+1):2*guerrieri_per_sq,:), 1 ); 
    match(2*guerrieri_per_sq+1,:);
    match(2*guerrieri_per_sq+2,:);
    match(2*guerrieri_per_sq+3,:)
    ];

% sq1_stronger_lost_count = sum( WATCH_ME(3, find( WATCH_ME(1,:) > WATCH_ME(2,:) ) ) );
% sq2_stronger_lost_count = sum( ~WATCH_ME(3, find( WATCH_ME(1,:) < WATCH_ME(2,:) ) ) );
% percentage_stronger_lose = (sq1_stronger_lost_count+sq2_stronger_lost_count)/Npartite

%distribution of the probs of sq1 to win 
histogram(WATCH_ME(4,:))

%% Generate check data

rng shuffle

for i=1:100
    sq1_eq = randsample(guerrieri, guerrieri_per_sq, false);
    sq2_eq = abs( sq1_eq + randi([-20,20], [1, guerrieri_per_sq]) );

    sq1_eq = sort( sq1_eq, 'descend' );
    sq2_eq = sort( sq2_eq, 'descend' );

    match_check(:,i) = [sq1_eq(:); sq2_eq(:)];
end

for i=101 : 150
    sq1_diff = randsample(guerrieri, guerrieri_per_sq, false);
    sq2_diff = floor( sq1_diff/(10*rand()+2) );
    sq1_diff = sort( sq1_diff, 'descend' );
    sq2_diff = sort( sq2_diff, 'descend' );
    match_check(:,i) = [sq1_diff(:); sq2_diff(:)];

    sq2_diff = randsample(guerrieri, guerrieri_per_sq, false);
    sq1_diff = floor( sq1_diff/(10*rand()+2) );
    sq1_diff = sort( sq1_diff, 'descend' );
    sq2_diff = sort( sq2_diff, 'descend' );
    match_check(:,i+50) = [sq1_diff(:); sq2_diff(:)];
end

%sum(match_check(giocatori_per_sq+1:2*giocatori_per_sq,:), 1) - sum(match_check(1:giocatori_per_sq,:), 1) 

clear sq2_diff sq1_diff sq2_eq sq1_eq 

save('GrandiPartite_check', 'match_check')



