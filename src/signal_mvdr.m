s=exp(1j*0.1*(1:10)*pi).';
for i = 1:20
    a = exp(1j*randn(1,10)*pi).';
    Rx = a*a' + 1/100*eye(10);
    if i == 10
        Rx = s*s' + 1/100*eye(10);
    end
    p(i) = 1/(s'/Rx*s);
end
figure;
plot(abs(p))

%% 时域信号
fs = 100;
T = 1;
tk = (0: T*fs-1) / fs;
f0 = 12;
s = cos( 2 * pi * f0* tk);
x = [s, zeros(1, 9*T*fs)];
x = x + 0.1*randn(size(x));
figure;
plot(x)
lambda = 0.01;
[P,C] = fun_signal_mvdr(x,s,lambda);
figure;
plot(P/max(P))
hold on;
plot(C/max(C))

%%
HTDstr = 'HTD256';
jstr = '1';
load(sprintf("G:/试验数据处理/西太/O1T1低频拖曳/4-原始数据直接脉冲压缩信号/%s/sig300-%s-%s.mat", HTDstr, HTDstr, jstr));
ht_mes = signal_cut;

load(sprintf("G:/试验数据处理/西太/O1T1低频拖曳/1-原始数据截取信号/%s/sig300-%s-%s.mat", HTDstr, HTDstr, jstr));
    % signal_cut = fun_Band
yt_mes = signal_cut - mean(signal_cut);
yt_mes = yt_mes / abs(max(yt_mes));

load("G:\试验数据处理\西太\O1T1低频拖曳\O1T1低频拖曳\sig300.mat");
fs = 32e3;
St_r = resample(St, 1, 2);

[P,C] = fun_signal_mvdr(yt_mes,St_r,lambda);

figure;
plot(P/max(P))
hold on;
plot(C/max(C))
