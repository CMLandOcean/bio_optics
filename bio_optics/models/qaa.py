def qaa(R_rs, wavelengths, lambdas=np.array([412, 443, 490, 555, 640, 670]), g0=0.089, g1=0.1245, h0=-1.146, h1=-1.366, h2=-0.469, a_w_res=[], b_bw_res=[]):

    wavelengths, idx = [utils.find_closest(wavelengths, wl) for wl in lambdas].T
    idx = idx.astype(int)

    R_rs = R_rs[idx]
    r_rs = air_water.above2below(R_rs)


    if len(a_w_res)==0:
        COMPUTE a_w_res

    if len(b_bw_res)==0:
        COMPUTE b_bw_res

    lambda0 = wavelengths[3] 

    # Step 1
    u = (-g0 + np.sqrt(g0**2 + 4*g1 * r_rs)) / 2*g1

    # Step 2 & 3
    ## QAA v5
    chi = np.log((r_rs[[1]] + r_rs[[2]]) / (r_rs[[3]] + 5*(r_rs[[5]]/r_rs[[2]]) * r_rs[[5]]))
    a_lambda_0_v5 = a_w[[3]] + 10**(h0 + h1*chi + h2*chi**2)
    b_bp_lambda_0_v5 = ((u[[3]] * a_lambda_0_v5) / (1-u[[3]])) - b_bw[[3]]
    ## QAA v6
    a_lambda_0_v6 = a_w[[5]] + 0.39 * (R_rs[[5]] / R_rs[[1]] + R_rs[[2]])**1.14
    b_bp_lambda_0_v6 = ((u[[5]] * a_lambda_0_v6) / (1-u[[3]])) - b_bw[[5]]
    # if Rrs(670) < 0.0015 sr-1: QAA_v5, else: QAA_v6
    a_lambda_0 = np.where(R_rs[[5]] < 0.0015, a_lambda_0_v5, a_lambda_0_v6)
    b_bp_lambda_0 = np.where(R_rs[[5]] < 0.0015, b_bp_lambda_0_v5, b_bp_lambda_0_v6)

    # Step 4
    eta = 2.0 * (1 - 1.2 * np.exp(-0.9 * (r_rs[[1]]/r_rs[[3]])))

    # Step 5
