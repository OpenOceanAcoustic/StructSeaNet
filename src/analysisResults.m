clc; close all;
load('..\results\20251122_130323\test_results.mat');
fs = 32;
tk = (0:size(predictions,3)-1)/fs;

for i = 1: size(predictions,1)
    fprintf('已处理 %d / %d 个文件\n', i, size(predictions,1));
    pred_log = squeeze(predictions(i,:,:));
    targ_log = squeeze(targets(i,:,:));
    ps_max_log = max_ps_log_sp(i);
    ps_min_log = min_ps_log_sp(i);
    ph_max_log = max_ph_log_sp(i);
    ph_min_log = min_ph_log_sp(i);
    Sd_i = Sd(i);
    Rr_i = Rr(i);
    Rd_i = Rd(i);
    % 反归一
    ps_log_sp = targ_log * (ps_max_log - ps_min_log) + ps_min_log;
    ph_log_sp = pred_log * (ph_max_log - ph_min_log) + ph_min_log;
    % 反转换
    ps = 10.^(ps_log_sp/20) - 1;
    ph = 10.^(ph_log_sp/20) - 1;
    figure;
    subplot(2,1,1);
    plot(tk,targ_log)
    hold on;
    plot(tk,pred_log)
    title('Prediction (Log Scale)')
    legend('Target','Prediction')
    xlabel('Time (s)')
    ylabel('Amplitude')
    subplot(2,1,2);
    plot(tk,ps)
    hold on;
    plot(tk,ph)
    title('Prediction (Linear Scale)')
    legend('Target','Prediction')
    xlabel('Time (s)')
    ylabel('Amplitude')
    sgtitle(sprintf('Sd %.2fm Rr %.2fkm Rd %.2fm', Sd_i, Rr_i, Rd_i))
    saveas(gcf, sprintf('../results/20251122_130323/figs/pred_log_%d.png', i));
    close(gcf);
end

%% 预测结果
fmin = 250;
fmax = 350;
model = 'BELLHOP';
envfil = 'MunkB_Arr';

load("G:\试验数据处理\西太\O1T1低频拖曳\ReceiveSigInfo.mat");
load("G:\试验数据处理\西太\O1T1低频拖曳\O1T1低频拖曳\BasicInfo-太平洋.mat");
load("filter.mat");

load("G:\试验数据处理\西太\O1T1低频拖曳\O1T1低频拖曳\sig300.mat");
St_resample = resample(St, 1, 2);
fs = 32000;
T = 50;
N = fs*T;
fk = (0:N-1)/N*fs;
tk = (0:N-1)/fs;
jsonStr = fileread('..\split\theTrueTrain.json');
jsonData = jsondecode(jsonStr);
for i = 1: length(jsonData.test)
    testBinPath = jsonData.test{i};
    strSplit = split(testBinPath, '\');
    htdname = strSplit{end};
    if ~contains(htdname,'HTD256')
        continue;
    end
    
    strSplit = split(htdname, '-');
    HTDstr = strSplit{2};
    strSplit = split(strSplit{3}, '.');
    jstr = strSplit{1};
    binPath = sprintf('G:/试验数据处理/西太/dataset/ht_32kHz/%s', htdname);
    fid = fopen(binPath, 'rb');
    Ns = fread(fid, 1, 'uint32');
    Sd = fread(fid, 1, 'float32');
    Rr = fread(fid, 1, 'float32');
    Rd = fread(fid, 1, 'float32');
    signal_cut_r = fread(fid, [Ns,1], 'float32');
    ht_r = fread(fid, [Ns,1], 'float32');
    fclose(fid);
    pred_log = squeeze(predictions(i,:,:));
    ps_max_log = max_ps_log_sp(i);
    ps_min_log = min_ps_log_sp(i);
    ph_max_log = max_ph_log_sp(i);
    ph_min_log = min_ph_log_sp(i);
    ph_log_sp = pred_log * (ph_max_log - ph_min_log) + ph_min_log;

    % 反转换
    ph = double(10.^(ph_log_sp/20) - 1);
    ph_re = resample(ph, 1000, 1);
    ph_cos = ph_re.' .* exp(1j * 2* pi*300*tk);
    [A,index] = findpeaks(ph,'MinPeakHeight',max(ph)/10);
    
    st_fft = fft(St_resample, N);
    tau = index / 32;
    p_fft = fft(ph_cos, N);
    % ykkk = st_fft .* (A.' * exp(-1j*2*pi*tau * fk));
    ykkk = st_fft .* p_fft;
   
    yttt = ifft(ykkk, 'symmetric');

    yttt_scale = (yttt - mean(yttt))/max(abs(yttt - mean(yttt)));
    load(sprintf("G:/试验数据处理/西太/O1T1低频拖曳/1-原始数据截取信号/%s/sig300-%s-%s.mat", HTDstr, HTDstr, jstr));
    % signal_cut = fun_BandFilter(signal_cut);
    signal_cut = (signal_cut - mean(signal_cut))/max(abs(signal_cut - mean(signal_cut)));
    signal_cut = filter(Num, 1, signal_cut);
    signal_cut_scale = (signal_cut - mean(signal_cut))/max(abs(signal_cut - mean(signal_cut)));

    figure;
    subplot(2,1,1);
    plot(tk, signal_cut_scale)
    title(sprintf('Sd %.2fm Rr %.2fkm Rd %.2fm', Sd, Rr, Rd))
    xlabel('Time (s)')
    ylabel('Amplitude')

    subplot(2,1,2);
    plot(tk, yttt_scale)
    xlabel('Time (s)')
    ylabel('Amplitude')
    saveas(gcf, sprintf('../results/20251122_130323/figs_td2/pred_linear_%d.png', i));
    % saveas(gcf, sprintf('../results/20251122_130323/figs_td/pred_linear_%d.fig', i));
    close(gcf);
end