htdpth_list = dir('../datasets/ht_resample_3.2kHz/sig300-*.bin');
Nj = length(htdpth_list);
for j = 1:Nj
    fprintf('已处理 %d / %d 个文件\n', j, Nj);
    htdpth = fullfile('../datasets/ht_resample_3.2kHz', htdpth_list(j).name);
    htdname = htdpth_list(j).name;
    fid = fopen(htdpth, 'rb');
    Ns = fread(fid, 1, 'uint32');
    Sd = fread(fid, 1, 'float32');
    Rr = fread(fid, 1, 'float32');
    Rd = fread(fid, 1, 'float32');
    signal_cut_r = fread(fid, [Ns,1], 'float32');
    ht_r = fread(fid, [Ns,1], 'float32');
    fclose(fid);
    HSR = 10*log10(sum(ht_r.^2) / sum(signal_cut_r.^2));
    if (isnan(ht_r(1)) || HSR<-15)
        continue;
    end
    preprocessed_signal = preprocess(signal_cut_r);
    pre_ht = preprocess(ht_r);

    ps_log = 20*log10(preprocessed_signal+1);
    ph_log = 20*log10(pre_ht+1);
    % 抽样
    ps_log_sp = ps_log(1:100:end);
    ph_log_sp = ph_log(1:100:end);
    % 归一
    ps_log_sp = (ps_log_sp - min(ps_log_sp)) / (max(ps_log_sp) - min(ps_log_sp));
    ph_log_sp = (ph_log_sp - min(ph_log_sp)) / (max(ph_log_sp) - min(ph_log_sp));

    NS2 = length(ps_log_sp);
    fid2 = fopen(fullfile('../datasets/ht_denoise_log_32Hz', htdname), 'wb');
    fwrite(fid2, NS2, 'uint32');
    fwrite(fid2, Sd, 'float32');
    fwrite(fid2, Rr, 'float32');
    fwrite(fid2, Rd, 'float32');
    fwrite(fid2, ps_log_sp, 'float32');
    fwrite(fid2, ph_log_sp, 'float32');
    fclose(fid2);

end

%% 测试读取
htdname = 'HTD042';
j = 1;
base_dir = '../datasets/ht_denoise_log_32Hz';
datapth = fullfile(base_dir, sprintf("sig300-%s-%d.bin", htdname, j));
fid = fopen(datapth, 'rb');
NS2 = fread(fid, 1, 'uint32');
Sd = fread(fid, 1, 'float32');
Rr = fread(fid, 1, 'float32');
Rd = fread(fid, 1, 'float32');
ps_log_sp = fread(fid, [NS2,1], 'float32');
ph_log_sp = fread(fid, [NS2,1], 'float32');
fclose(fid);
figure;
plot(ps_log_sp);
title('ps_log_sp');
figure;
plot(ph_log_sp);
title('ph_log_sp');