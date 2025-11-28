close all;
HTDstr = 'HTD256';
jstr = '1';
load(sprintf("G:/试验数据处理/西太/O1T1低频拖曳/4-原始数据直接脉冲压缩信号/%s/sig300-%s-%s.mat", HTDstr, HTDstr, jstr));
ht_mes = signal_cut;
fs = 32e3;
tk = (0:length(ht_mes)-1)/fs;
figure;
plot(tk, ht_mes)
fsc = fft(ht_mes);
N = length(ht_mes);

fk = (0:N-1)/N*fs;
figure;
plot(fk, abs(fsc))
xlim([0 fs/2])

%%
load("G:\试验数据处理\西太\O1T1低频拖曳\O1T1低频拖曳\sig300.mat");
fs = 32e3;
St_resample = resample(St, 1, 2);
S2 = xcorr(St_resample, St_resample);
ts2 = (0:length(S2)-1)/fs;
figure;
plot(ts2, S2)
S4 = xcorr(S2, S2);
ts4 = (0:length(S4)-1) / fs;
figure;
plot(ts4, S4)

%%

fst = fft(St_resample, N);
ffy = fsc .* fst;
yt = ifft(ffy, 'symmetric');
figure;
subplot 211
plot(tk, yt)

load(sprintf("G:/试验数据处理/西太/O1T1低频拖曳/1-原始数据截取信号/%s/sig300-%s-%s.mat", HTDstr, HTDstr, jstr));
    % signal_cut = fun_Band
yt_mes = signal_cut;
subplot 212
plot(tk, yt_mes)

%% 抽样

ht_mes_resample = resample(ht_mes, 1, 10);
Ns = length(ht_mes_resample);
St_resample_10 = resample(St_resample, 1, 10);
fscr = fft(ht_mes_resample);
fstr = fft(St_resample_10, Ns);
fyr = fscr .* fstr;

ytr = ifft(fyr, 'symmetric');
tkr = (0:Ns-1)/fs*10;
figure;
subplot 211
plot(tkr, ytr)
subplot 212
plot(tk, yt_mes)

%%
fh2 = fyr .* conj(fstr);
h2 = ifft(fh2, 'symmetric');
figure;
plot(tkr, h2)

%%
ht = [[33.2678125000000,1.59407556400079]
[33.4147187500000,2.29641212028272]
[26.7893125000000,2.66890707065256]
[26.4815312500000,3.59558987819634]
[25.7670937500000,5.28970451194757]
[19.6525000000000,6.41877169258724]
[19.1984062500000,9.60834484214837]
[18.5130937500000,11.9559818058410]
[18.2311250000000,13.6210844832145]
[12.5947812500000,10.4810908765969]
[12.4628125000000,13.3561770821473]
[12.1675312500000,24.0944013211100]
[11.4950937500000,-132.958232355569]
[12.0177812500000,22.4491500455603]
[11.9989687500000,24.1292295825099]
[11.9642187500000,21.6892593499929]
[11.3805000000000,147.027299587171]];
ht(:,2) = ht(:,2)/ max(ht(:,2));
ht(:,1) = ht(:,1);
figure;
stem(ht(:,1), ht(:,2))

%%
fs = 32000;

fyhth = fst .* (ht(:,2).'*exp(-1j*2*pi*ht(:,1)*fk));

fyhth(fk<250) = 0;
fyhth(fk>350) = 0;

yhth = ifft(fyhth, 'symmetric');
figure
subplot 211
plot(tk, yhth)
subplot 212
plot(tk, yt_mes)