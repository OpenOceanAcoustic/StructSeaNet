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

yyy = fun_hilt_preprocess(ht_mes);
[pks, loks] = findpeaks(yyy, 'MinPeakHeight', max(yyy)/100, 'MinPeakDistance', 1000);
figure;
stem(loks/fs, pks)

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
ht = [loks.'/fs, pks.'];
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