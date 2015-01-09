matlab_out

x = [1:length(hu1)];

figure(1);
plot(hu1, 'r');
line([0 length(x)], [0.5 0.5]);
% line([x(125) x(125)], [0, max(hu1)]);
% line([x(185) x(185)], [0, max(hu1)]);
% line([x(220) x(220)], [0, max(hu1)]);
% line([x(285) x(285)], [0, max(hu1)]);
% line([x(330) x(330)], [0, max(hu1)]);
grid on;

% figure(2);
% plot(hu2, 'r');
% line([0 length(x)], [1.75 1.75]);
% line([x(125) x(125)], [0, max(hu2)]);
% line([x(185) x(185)], [0, max(hu2)]);
% line([x(220) x(220)], [0, max(hu2)]);
% line([x(285) x(285)], [0, max(hu2)]);
% line([x(330) x(330)], [0, max(hu2)]);
% grid on;
% 
% figure(3);
% plot(hu4, 'r');
% line([0 length(x)], [0.08 0.08]);
% line([x(125) x(125)], [0, max(hu4)]);
% line([x(185) x(185)], [0, max(hu4)]);
% line([x(220) x(220)], [0, max(hu4)]);
% line([x(285) x(285)], [0, max(hu4)]);
% line([x(330) x(330)], [0, max(hu4)]);
% grid on;