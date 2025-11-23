clc; close all;
load('..\results\20251122_130323\test_results.mat');
fs = 32;
tk = (0:size(predictions,3)-1)/fs;

for i = 1: size(predictions,1)
    fprintf('已处理 %d / %d 个文件\n', i, size(predictions,1));
    pred_log = squeeze(predictions(i,:,:));
    targ_log = squeeze(targets(i,:,:));
    figure;
    plot(tk,targ_log)
    hold on;
    plot(tk,pred_log)
    title('Prediction (Log Scale)')
    legend('Target','Prediction')
    xlabel('Time (s)')
    ylabel('Amplitude')
    saveas(gcf, sprintf('../results/20251122_130323/figs/pred_log_%d.png', i));
    close(gcf);
end
