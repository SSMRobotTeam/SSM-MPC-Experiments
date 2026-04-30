function [F, J] = Func_Feedback_pk(yk, pk, W_0_for_FB, p_guess, opt_W)
    
    y_cal      = R_reduced_to_full(pk,W_0_for_FB,[],1);

    F0         = [(y_cal - yk);...
                  sqrt(opt_W)*(pk-p_guess)];
    F  =  F0;

end

