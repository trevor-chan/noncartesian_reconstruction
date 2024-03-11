clear all;
gamma=2*pi*42.5764/1000;%1/(us mT)

Nshots=32;
dwelltime=10;%microsec


%t=t/to;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FoV=0.192; %m
N=192;
k_max=N/(2*FoV);
lambda=Nshots/(2*pi*FoV);%m-1
theta_max_rad=k_max/lambda

Nturns=theta_max_rad/(2*pi)

dtheta=0:360/Nshots:359;
    dtheta=dtheta*pi/180;
   




S_Ro=0.05;%(mT/m)/us
BW=1/(dwelltime);
Go=BW*2*pi/(gamma*FoV);%mT/m

beta=(gamma)*(S_Ro)/(2*pi*lambda);%[m][1/(us mT)][mT/m]/us = us^(-2)
%a2=(9*beta/4)^(1/3);
a2=exp((1/3)*log(9*beta/4));
Ts=((3*gamma*Go/(4*pi*lambda*a2^2))^3)
% Ts=Ts/10;
% Ts=10*floor(Ts)%us
Lamb=1;%to adjust S(t=0)
a=0.5*beta*(Ts).^2;
b=Lamb+(beta/(2*a2))*(Ts^(4/3));
theta_s=a./b;
%Ts=Ts/to;%%%%%%%%%%%%%%%%%%%%fig%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%n=floor(Ts/dwelltime);


Tacq=Ts+(pi*lambda)*(theta_max_rad^2-theta_s^2)/(gamma*Go);
num_samples=ceil(Tacq/dwelltime)
 t=0:dwelltime:(num_samples-1)*dwelltime;
% for t=0:n-1
theta=zeros(1,num_samples);   
% if t<n %i.e. < Ts
s=t(t<=Ts);
    a=0.5*beta*(s).^2;
    b=Lamb+(beta/(2*a2))*(s.^(4/3));
    theta1=a./b;

    
 u=t(t>=Ts);
 
     theta2=sqrt(theta1(end).^2+((gamma)*Go/(pi*lambda)).*((u-Ts)));
    
% %    end
 theta=cat(2,theta1(1:end-1),theta2(1:end-1));

k=lambda*theta;
figure; polarplot(theta, k,'.')

kx=k.*cos(theta);
ky=k.*sin(theta);

  
hold on

save('spiral_trajectory.mat', "kx","ky")

return;
