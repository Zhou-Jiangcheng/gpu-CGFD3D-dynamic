clc;
clear all;
close all;
load("seismodata1.mat");
load("seismot1.mat");
% load("seismodata2.mat");
% load("seismot2.mat");
% load("seismodata3.mat");
% load("seismot3.mat");
load("matlab.mat");
% load("matlab1.mat");
nrec = 1;
figure(1);
for irec=1:nrec
    figure(irec)
    plot(seismot1(1:10:1501,irec),seismodata1(1:10:1501,irec),'r','LineWidth',1.5);
    hold on;
    plot(seismot1(1:10:1501,irec),aaa2,'b--','LineWidth',1.5);
    bbb = seismodata1(1:10:1501,irec)';
%     dif =seismodata1(:,irec)-seismodata2(:,irec);
    dif =bbb-aaa2;
    plot(seismot1(1:10:1501,irec),dif,'r','LineWidth',1.5);
    title("Slip");
%     legend("90","70","45", FontSize=10, FontWeight='bold');
%     legend("LiHL","ZhangWQ", FontSize=10, FontWeight='bold');
end


