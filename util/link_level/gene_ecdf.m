% created by Bryanttxc, FUNLab

function [pow_dB_fixed] = gene_ecdf(power_sample)
%GENE_ECDF generate empirical cdf

power_dB = reshape(10.*log10(power_sample), [], 1); % W -> dB
power_dB(power_dB == -Inf) = -600; % zero item -> infinite value

[f_pow, pow_dB] = ecdf(power_dB);

f_fixed = linspace(0, 1, 1000);
pow_dB_fixed = interp1(f_pow, pow_dB, f_fixed, 'linear', 'extrap'); % interpolation

end

