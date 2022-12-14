%oldp = [0.99,0.009,0.001];
%oldp = [0.2,0.6,0.2];
% oldp = [0.18,0.6,0.22];
real_advantage = [-1, -1, 1];
flip_arr =  [1, 1, 1];

for epoch = 1:1
    r = randi([1 3]);
    advantage = real_advantage;
    %advantage(r) = advantage(r)*-1;
    epsilon = 0.1;
    % pitheta0 = [0.22,0.56,0.22];
    deltaY = [0,0,0];
    %loss = hpof(pitheta0 ,deltaY ,epsilon,advantage ,oldp )
    oneslike = ones(1,length(oldp));
    %psum = pitheta*oneslike'
    fun = @(pitheta)[hpof(pitheta ,[0,0,0] ,epsilon,advantage ,oldp );hpof(pitheta ,[1,0,0] ,epsilon,advantage ,oldp);hpof(pitheta ,[0,1,0],epsilon,advantage ,oldp );hpof(pitheta ,[0,0,1] ,epsilon,advantage ,oldp )];
    %constrpi:pitheta*oneslike'
    %x= fminimax(fun,x0,A,b,Aeq,beq,lb,ub)
    % Ax<=b
    % Aeqx =beq
    % ub>=x>=lb
    lb = [0,0,0];
    ub = [1,1,1];
    %x0 = [0.21,0.58,0.21];
    A = []; % No linear constraints
    b = [];
    Aeq = [1,1,1];
    beq = [1];
    %Aeq = [];
    %beq = [];
    [x,fval] = fminimax(fun,oldp,A,b,Aeq,beq,lb,ub)
    oldp = x
end
%disp(real_advantage)
%disp("oldp ",oldp)