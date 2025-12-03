function [P,C] = fun_signal_mvdr(x,s,lambda)
% 输入：
% x：输入信号
% s：参考信号
% lambda：正则化参数
% 输出：
% P：MVDR输出信号
% C：相关系数

B2 = sum(s.^2) / length(s);
x = [zeros(1, length(s)-1),x, zeros(1, length(s)-1)];
s_flip = flip(s);
e = conv(x, s_flip);
C = e.^2 / length(s)^2 / B2;
e1 = ones(size(s));
A2 = conv(x.^2, e1)/ length(s);
P = abs(lambda ./ (1 - C ./ (A2+lambda)));
