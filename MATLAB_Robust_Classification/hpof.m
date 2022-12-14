% function [ploss] = hpof(pitheta,deltaY,epsilon,advantage,oldpi )
function [ploss] = hpof(pitheta,deltaY,epsilon,advantage,oldpi,advw )
%HPOF Summary of this function goes here
%   calculate hpo loss by given data
%ploss = max(0, epsilon - advantage*( (pitheta/oldpi)-1 ) )
oneslike = ones(1,length(pitheta));
zeroeslike = zeros(1,length(pitheta));
twolike = ones(1,length(pitheta));
epsilonlike = ones(1,length(pitheta));
for i=1:length(pitheta)
    twolike(i) = 2;
    epsilonlike = epsilon;
end
% ploss = epsilon - advantage*(1-2*deltaY)*( (pitheta/oldpi)-1 );
ptemp = epsilonlike - advantage.*(oneslike-twolike.*oneslike.*deltaY).*( (pitheta./oldpi)- oneslike );
A=[zeroeslike;ptemp.*advw];
%if ploss > 0
%    ploss = ploss;
%else
%    ploss = 0;
%end
ploss = max(A)*oneslike';
end

