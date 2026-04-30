function err = optimize_pk(pk, zk_meas, W_0_for_out)
    %OPTIMIZE_PK 此处显示有关此函数的摘要
    %   此处显示详细说明
    zk_guess  = R_reduced_to_full(pk, W_0_for_out, [], 1);
    err_vec   = zk_meas - zk_guess(1:6);
    err       = 0.5 * (err_vec'*err_vec);
end

