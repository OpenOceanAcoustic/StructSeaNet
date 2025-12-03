clc; close all;
%% 导入信号并降采样
load("G:\试验数据处理\西太\O1T1低频拖曳\O1T1低频拖曳\sig300.mat");
St_resample = resample(St, 1, 2);

%% 处理
fs = 32e3;
fmin = 250;
fmax = 350;
HTDstr = 'HTD256';
for j = 10:394
    load(sprintf("G:/试验数据处理/西太/O1T1低频拖曳/4-原始数据直接脉冲压缩信号/%s/sig300-%s-%d.mat", HTDstr, HTDstr, j));
    ht_mes = signal_cut;

    tk = (0:length(ht_mes)-1)/fs;
    figure;
    plot(tk, ht_mes)
    fsc = fft(ht_mes);
    N = length(ht_mes);
    yyy = ht_mes;
    % yyy = fun_hilt_preprocess(ht_mes);
    [pks, loks] = findpeaks(yyy, 'MinPeakHeight', max(yyy)/100, 'MinPeakDistance', 3200);
    % figure;
    % stem(loks/fs, pks)

    fst = fft(St_resample, N);

    ht = [loks.'/fs, pks.'];
    ht(:,2) = ht(:,2);
    ht(:,1) = ht(:,1);
    figure;
    stem(ht(:,1), ht(:,2))

    fyhth = fst .* (ht(:,2).'*exp(-1j*2*pi*ht(:,1)*fk));

    fyhth(fk<fmin) = 0;
    fyhth(fk>fmax) = 0;

    yhth = ifft(fyhth, 'symmetric');

    fyy2 = fst .* fsc;
    yy2 = ifft(fyy2, 'symmetric');

    load(sprintf("G:/试验数据处理/西太/O1T1低频拖曳/1-原始数据截取信号/%s/sig300-%s-%d.mat", HTDstr, HTDstr, j));
    yt_mes = signal_cut;
    figure
    subplot 311
    plot(tk, yhth)
    xlabel('Time (s)')
    ylabel('Amplitude')
    title('冲激恢复信号')

    subplot 312
    plot(tk, yy2)
    xlabel('Time (s)')
    ylabel('Amplitude')
    title('直接恢复信号')

    subplot 313
    plot(tk, yt_mes)
    xlabel('Time (s)')
    ylabel('Amplitude')
    title('原始信号')
    saveas(gcf, sprintf("../5-冲激恢复信号/sig300-%s-%d.png", HTDstr, j));
    close all;


end