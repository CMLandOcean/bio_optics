import numpy as np
import os
import pandas as pd


## PACE l2gen flag coding
def l2gen_flags_PACE(flags, flagN):
    flags = np.array(flags, dtype='int32')
    flagDict = {'STRAYLIGHT': 8,
                'MAXAERITER': 19,
                'COCCOLITH': 10,
                'HISATZEN': 5}
    exp = flagDict[flagN]
    return np.bitwise_and(flags, 2 ** exp) == 2 ** exp

def set_valid_bands_PACE(version=2):
    if version==2:
        wlstr = ['400', '403', '405', '408', '410', '413', '415', '418', '420', '422', '425',
                 '427', '430', '432', '435', '437', '440', '442', '445', '447', '450', '452',
                 '455', '457', '460', '462', '465', '467', '470', '472', '475', '477', '480',
                 '482', '485', '487', '490', '492', '495', '497', '500', '502', '505', '507',
                 '510', '512', '515', '517', '520', '522', '525', '527', '530', '532', '535',
                 '537', '540', '542', '545', '547', '550', '553', '555', '558', '560', '563',
                 '565', '568', '570', '573', '575', '578', '580', '583', '586', '588', '613', '615', '618',
                 '620', '623', '625', '627', '630', '632', '635', '637', '640', '641', '642',
                 '643', '645', '646', '647', '648', '650', '651', '652', '653', '655', '656',
                 '657', '658', '660', '661', '662', '663', '665', '666', '667', '668', '670',
                 '671', '672', '673', '675', '676', '677', '678', '682', '683',
                 '684', '686', '687', '688', '689', '691', '692', '693', '694', '696', '697',
                 '698', '699', '701', '702', '703', '704', '706', '707', '708', '709', '711',
                 '712', '713', '714', '717']
    if version == 3:
        wlstr = ['400', '403', '405', '408', '410', '413', '415', '418', '420', '422', '425',
                 '427', '430', '432', '435', '437', '440', '442', '445', '447', '450', '452',
                 '455', '457', '460', '462', '465', '467', '470', '472', '475', '477', '480',
                 '482', '485', '487', '490', '492', '495', '497', '500', '502', '505', '507',
                 '510', '512', '515', '517', '520', '522', '525', '527', '530', '532', '535',
                 '537', '540', '542', '545', '547', '550', '553', '555', '558', '560', '563',
                 '565', '568', '570', '573', '575', '578', '580', '583', '586', '588', '613',
                 '615', '618', '620', '623', '625', '627', '630', '632', '635', '637', '640',
                 '641', '642', '643', '645', '646', '647', '648', '650', '651', '652', '653',
                 '655', '656', '657', '658', '660', '661', '662', '663', '665', '666', '667',
                 '668', '670', '671', '672', '673', '675', '676', '677', '678', '679', '681',
                 '682', '683', '684', '686', '687', '688', '689', '691', '692', '693', '694',
                 '696', '697', '698', '699', '701', '702', '703', '704', '706', '707', '708',
                 '709', '711', '712', '713', '714', '717', '719']
    return wlstr


def set_wavelengths_bySensor(sensor, versionAC, maxWL=None):
    if sensor == 'PACE':
        wlstr = set_valid_bands_PACE(versionAC)

    if sensor == 'EnMAP':
        if versionAC == 'v010402':
            fname="Z:\projects\ongoing\EnsAD\workspace\data\EnMAP_extracts\OWT_NorthSea\extracts_OWT3a_North_Sea_EnMAP_20241118_Rrs_membSumFilt2_QWIPfilt.txt"
            d = pd.read_csv(fname, header=0, sep='\t')
            wlstr = [b for b in d.columns.values if
                     not b in ['idx', 'idy', 'lon', 'lat', 'OWT', 'date', 'QWIP', 'AVW', 'AVW_', 'membSum']]
        if versionAC == 'v010502L':
            fname = "D:\Documents\projects\EnsAD\EnMAP\L2A_Land\\v010502\extracts_byOWTrefined_combined\extracts_OWT3a_g_Lakes_EnMAP_20250516_Rrs_flags.txt"
            d = pd.read_csv(fname, header=0, sep='\t')
            wlstr = [b for b in d.columns.values if
                     not b in ['idx', 'idy', 'lon', 'lat', 'OWT', 'date', 'QWIP', 'AVW', 'AVW_', 'membSum']]

    if sensor == 'CHIME_Sim':
        wl = np.asarray([int(a) for a in np.arange(400, 955, 5)])
        wlstr = [str(a) for a in wl]

    wavelength = np.asarray([float(a) for a in wlstr])
    if not maxWL is None:
        ID = wavelength <maxWL
        wavelength = wavelength[ID]

    return wavelength



def read_PACE_extracts(seaName = 'Baltic Sea' , #['Baltic Sea', 'North Sea']
                       regionName = '',
                       OWTList = [],
                       versionAC = 2,
                       outpath = r"E:\Documents\projects\EnsAD\data\PACE_test\extracts_Baltic\\",
                       datasetDate = '20240925'):

    if seaName == 'Baltic Sea' and versionAC==2:
        ## Baltic Sea PACE, all OWT, all subregions combined
        path = "Z:\projects\ongoing\EnsAD\workspace\data\PACE_extracts_v2\OWT_Baltic\\"
        # OWTList = ['1', '2', '3a', '3b', '4a', '4b', '5a', '5b', '6', '7']
        # combination of OWTs with the same phytoplankton groups:
        # OWT2: C0, C1, C6, C7
        # OWT 3a + 3b: C0, C3, C4, C6
        # OWT 4a + 4b + 5a: C3, C4, C6
        # OWT 5b + 6 + 7: C1, C3, C4, C6
        # OWTList = ['3a', '3b']
        # OWTList = ['4a', '4b', '5a']
        # OWTList = ['5b', '6', '7']
        fnameList = os.listdir(path)
        fnameL = []
        OWTstr = ''
        for owt in OWTList[:]:
            [fnameL.append(fn) for fn in fnameList if owt in fn]
            OWTstr += owt + '_'
        print(fnameL)

        wlstr = set_valid_bands_PACE(versionAC)

        datasetName = 'PACE_BalticSea_OWT' + OWTstr + datasetDate
        if os.path.exists(outpath + 'extracts_' + datasetName + '.txt'):
            print('read directly')
            d = pd.read_csv(outpath + 'extracts_' + datasetName + '.txt', sep='\t', header=0)
            Rrs = d[wlstr].copy()
        else:
            Rrs = None
            meta = None
            for fn in fnameL:
                d = pd.read_csv(path + fn, sep='\t', header=0)

                if 'l2_flags' in d.columns.values:
                    flagID = l2gen_flags_PACE(d.l2_flags.values, 'HISATZEN')
                    ID = np.logical_not(flagID)
                    d = d.loc[ID, :]
                    d = d.reset_index(drop=True)

                if Rrs is None:
                    Rrs = d[wlstr].copy()
                    meta = d.copy()
                else:
                    Rrs = pd.concat([Rrs, d[wlstr]])
                    meta = pd.concat([meta, d])

            meta.to_csv(outpath + 'extracts_' + datasetName + '.txt', sep='\t', header=True, index=False)

        # simple correction of negative Rrs
        r_rs = Rrs.copy()
        minRrs = np.min(Rrs.values, axis=1)
        print(minRrs.shape)
        ID = np.array(minRrs < 0)
        rrs = Rrs.values.copy()
        for i in range(Rrs.shape[1]):  # iterate along wavelengths
            rrs[ID, i] = rrs[ID, i] - minRrs[ID]

        wavelengths = np.asarray([float(a) for a in wlstr])
        r_rs.loc[:, :] = rrs.copy()

    if seaName == 'Baltic Sea' and versionAC==3:
        # simple correction of negative Rrs
        wlstr = set_valid_bands_PACE(versionAC)
        fnamesL = os.listdir(outpath)
        fnamesL = [fn for fn in fnamesL if 'OWT'+OWTList[0] in fn and fn.startswith('extracts')]
        if len(regionName)>0:
            fnamesL = [fn for fn in fnamesL if regionName in fn]

        if len(fnamesL) == 0:
            return None, None, None, None
        datasetName = fnamesL[0].split('extracts_')[1].split('.txt')[0]
        d = pd.read_csv(outpath + fnamesL[0], sep='\t', header=0)
        Rrs = d[wlstr].copy()
        r_rs = Rrs.copy()
        minRrs = np.min(Rrs.values, axis=1)
        print(minRrs.shape)
        ID = np.array(minRrs < 0)
        rrs = Rrs.values.copy()
        for i in range(Rrs.shape[1]):  # iterate along wavelengths
            rrs[ID, i] = rrs[ID, i] - minRrs[ID]

        wavelengths = np.asarray([float(a) for a in wlstr])
        r_rs.loc[:, :] = rrs.copy()

    if seaName == 'North Sea':
        # ## North Sea PACE, all OWT, all days combined
        # path = "Z:\projects\ongoing\EnsAD\workspace\data\PACE_extracts_v2\OWT_NorthSea\\"
        # # OWTList = ['1', '2', '3a', '3b', '4a', '4b', '5a', '5b', '6', '7']
        # # combination of OWTs with the same phytoplankton groups:
        # # OWT 1: C7
        # # OWT 2 + 3a + 4a + 4b + 5a + 5b : C0, C2, C6
        # # OWT 3b: C0, C2, C5, C6
        # # OWT 6 + 7: C0, C1, C2, C6
        # # OWTList = ['1']
        # # OWTList = ['2', '3a', '4a', '4b', '5a', '5b']
        # # OWTList = ['3b']
        # # OWTList = ['5a']
        # # OWTList = ['5b']
        # # OWTList = ['6']
        # # OWTList = ['7']
        # # OWTList = ['6', '7']
        # fnameList = os.listdir(path)
        # fnameList = [fn for fn in fnameList if datasetDate in fn]
        #
        # fnameL = []
        # OWTstr = ''
        # for owt in OWTList[:]:
        #     print(owt)
        #     [fnameL.append(fn) for fn in fnameList if 'OWT' + owt in fn]
        #     OWTstr += owt + '_'
        #
        # print(fnameL)
        #
        # # dataDict = {}
        # wlstr = set_valid_bands_PACE()
        #
        # datasetName = 'PACE_NorthSea_OWT' + OWTstr + datasetDate
        # if os.path.exists(outpath + 'extracts_' + datasetName + '.txt'):
        #     print('read directly')
        #     d = pd.read_csv(outpath + 'extracts_' + datasetName + '.txt', sep='\t', header=0)
        #     Rrs = d[wlstr].copy()
        # else:
        #     Rrs = None
        #     meta = None
        #     for fn in fnameL:
        #         d = pd.read_csv(path + fn, sep='\t', header=0)
        #
        #         if 'l2_flags' in d.columns.values:
        #             flagID = l2gen_flags_PACE(d.l2_flags.values, 'HISATZEN')
        #             ID = np.logical_not(flagID)
        #             d = d.loc[ID, :]
        #             d = d.reset_index(drop=True)
        #
        #         if Rrs is None:
        #             Rrs = d[wlstr].copy()
        #             meta = d.copy()
        #         else:
        #             Rrs = pd.concat([Rrs, d[wlstr]])
        #             meta = pd.concat([meta, d])
        #
        #     meta.to_csv(outpath + 'extracts_' + datasetName + '.txt', sep='\t', header=True, index=False)

        # simple correction of negative Rrs
        wlstr = set_valid_bands_PACE(versionAC)
        fnamesL = os.listdir(outpath)
        fnamesL = [fn for fn in fnamesL if 'OWT'+OWTList[0] in fn and fn.startswith('extracts')]
        fnamesL = [fn for fn in fnamesL if seaName.replace(' ', '_') in fn]
        datasetName = fnamesL[0].split('extracts_')[1].split('.txt')[0]
        d = pd.read_csv(outpath + fnamesL[0], sep='\t', header=0)
        Rrs = d[wlstr].copy()
        r_rs = Rrs.copy()
        minRrs = np.min(Rrs.values, axis=1)
        print(minRrs.shape)
        ID = np.array(minRrs < 0)
        rrs = Rrs.values.copy()
        for i in range(Rrs.shape[1]):  # iterate along wavelengths
            rrs[ID, i] = rrs[ID, i] - minRrs[ID]

        wavelengths = np.asarray([float(a) for a in wlstr])
        r_rs.loc[:, :] = rrs.copy()

    return r_rs, wlstr, wavelengths, datasetName


def read_EnMAP_extracts(seaName = 'Baltic Sea' , #['Baltic Sea', 'North Sea']
                        regionName = '',
                        OWTList = [],
                        outpath = r"E:\Documents\projects\EnsAD\data\EnMAP_NN_training\\extracts_NorthSea\\",
                        datasetDate = '20241118',
                        datasetID = 'membSumFilt2_QWIPfilt', # Baltic: 'QWIPfilt'
                        dataType = 'orig',
                        maxWL=750.,
                        negativeCorr = True
    ):

    if dataType=='orig':
        if seaName == 'Lakes':
            path = "D:\Documents\projects\EnsAD\EnMAP\L2A_Land\\v010502\extracts_byOWTrefined_combined\\"
            fnamesL = os.listdir(outpath)
            fnamesL = [fn for fn in fnamesL if 'OWT' + OWTList[0] in fn and fn.startswith('extracts')]
            if len(regionName) > 0:
                fnamesL = [fn for fn in fnamesL if regionName in fn]

            if len(fnamesL) == 0:
                return None, None, None, None

            d = pd.read_csv(path + fnamesL[0], header=0, sep='\t')
            wlstr = [b for b in d.columns.values if
                     not b in ['idx', 'idy', 'lon', 'lat', 'OWT', 'date', 'QWIP', 'AVW', 'AVW_', 'membSum', 'Area', 'NDI']]

            wavelengths = np.asarray([float(a) for a in wlstr])
            ## reduce the number of bands in the visible
            IDwv = np.array(wavelengths < maxWL)
            wlstr = np.asarray(wlstr)[IDwv]
            wavelengths = wavelengths[IDwv]

            Rrs = d[wlstr].copy()
            r_rs = Rrs.copy()

            minRrs = np.min(Rrs.values, axis=1)
            print(minRrs.shape)
            ID = np.array(minRrs < 0)
            rrs = Rrs.values.copy()
            for i in range(Rrs.shape[1]):  # iterate along wavelengths
                rrs[ID, i] = rrs[ID, i] - minRrs[ID]

            datasetName = "EnMAP_"+ seaName+"_OWT"+OWTList[0]

            r_rs.loc[:, :] = rrs.copy()

        if seaName == 'SHLakes':
            path = "D:\Documents\projects\EnsAD\EnMAP\L2A_Land\\v010502\EnMAP_extracts_Lakes_LandAC_v010502\extracts_byOWTrefined_combined\\"
            fnamesL = os.listdir(path)
            fnamesL = [fn for fn in fnamesL if 'OWT' + OWTList[0] in fn and fn.startswith('extracts')]
            if len(regionName) > 0:
                fnamesL = [fn for fn in fnamesL if regionName in fn]

            if len(fnamesL) == 0:
                return None, None, None, None

            d = pd.read_csv(path + fnamesL[0], header=0, sep='\t')
            wlstr = [b for b in d.columns.values if
                     not b in ['idx', 'idy', 'lon', 'lat', 'OWT', 'date', 'QWIP', 'AVW', 'AVW_', 'membSum', 'Area', 'NDI']]

            wavelengths = np.asarray([float(a) for a in wlstr])
            ## reduce the number of bands in the visible
            IDwv = np.array(wavelengths < maxWL)
            wlstr = np.asarray(wlstr)[IDwv]
            wavelengths = wavelengths[IDwv]

            Rrs = d[wlstr].copy()
            r_rs = Rrs.copy()

            if negativeCorr:
                minRrs = np.min(Rrs.values, axis=1)
                print(minRrs.shape)
                ID = np.array(minRrs < 0)
                rrs = Rrs.values.copy()
                for i in range(Rrs.shape[1]):  # iterate along wavelengths
                    rrs[ID, i] = rrs[ID, i] - minRrs[ID]

                r_rs.loc[:, :] = rrs.copy()

            datasetName = "EnMAP_" + regionName + "_OWT" + OWTList[0]

        if seaName == 'North Sea':
            ## North Sea EnMAP, all OWT, all days combined, Classifiable
            path = "Z:\projects\ongoing\EnsAD\workspace\data\EnMAP_extracts\OWT_NorthSea\\"
            # combination of OWTs with the same phytoplankton groups:
            # OWT 1: C7
            # OWT 2 + 3a + 4a + 4b + 5a  : C0, C2, C6
            # OWT 3b: C0, C2, C5, C6
            # OWT 5b + 6 + 7: C0, C1, C2, C6
            ### different restrictions for inversion!
            # OWTList = ['1'] -> no data
            # OWTList = ['2'] #-> no data
            # OWTList = ['3a'] # DONE -> chl_tot_max = 10, ISM_max=10, CDOM_max=1
            # OWTList = ['3b'] # -> chl_tot_max = 10, ISM_max=10, CDOM_max=10 + cocc.
            # OWTList = ['4a'] # DONE -> chl_tot_max = 30, ISM_max=20, CDOM_max=5
            # OWTList = ['4b'] # DONE -> chl_tot_max = 300, ISM_max=100, CDOM_max=10
            # OWTList = ['5a'] # DONE -> chl_tot_max = 300, ISM_max=100, CDOM_max=100, offset!
            # OWTList = ['5b'] # DONE -> chl_tot_max = 1000, ISM_max=100, CDOM_max=100, ofset!
            # OWTList = ['6']  #  -> chl_tot_max = 500, ISM_max=1000, CDOM_max=100, no offset!
            # OWTList = ['7']  #
            # OWTList = ['6', '7']
            fnameList = os.listdir(path)

            fnameList = [fn for fn in fnameList if datasetDate in fn]
            fnameList = [fn for fn in fnameList if datasetID in fn]

            d = pd.read_csv(path + fnameList[0], header=0, sep='\t')
            wlstr = [b for b in d.columns.values if
                     not b in ['idx', 'idy', 'lon', 'lat', 'OWT', 'date', 'QWIP', 'AVW', 'AVW_', 'membSum']]

            fnameL = []
            OWTstr = ''
            for owt in OWTList[:]:
                print(owt)
                [fnameL.append(fn) for fn in fnameList if 'OWT' + owt in fn]
                OWTstr += owt + '_'

            print(fnameL)

            datasetName = 'EnMAP_NorthSea_OWT' + OWTstr + datasetDate + '_' + datasetID
            if os.path.exists(outpath + 'extracts_' + datasetName + '.txt'):
                print('read directly')
                d = pd.read_csv(outpath + 'extracts_' + datasetName + '.txt', sep='\t', header=0)
                Rrs = d[wlstr].copy()
            else:
                Rrs = None
                meta = None
                for fn in fnameL:
                    d = pd.read_csv(path + fn, sep='\t', header=0)

                    if Rrs is None:
                        Rrs = d[wlstr].copy()
                        meta = d.copy()
                    else:
                        Rrs = pd.concat([Rrs, d[wlstr]])
                        meta = pd.concat([meta, d])

                meta.to_csv(outpath + 'extracts_' + datasetName + '.txt', sep='\t', header=True, index=False)

            # simple correction of negative Rrs
            r_rs = Rrs.copy()
            minRrs = np.min(Rrs.values, axis=1)
            print(minRrs.shape)
            ID = np.array(minRrs < 0)
            rrs = Rrs.values.copy()
            for i in range(Rrs.shape[1]):  # iterate along wavelengths
                rrs[ID, i] = rrs[ID, i] - minRrs[ID]

            wavelengths = np.asarray([float(a) for a in wlstr])
            r_rs.loc[:, :] = rrs.copy()

        if seaName == 'Baltic Sea':
            ## Baltic Sea EnMAP, all OWT, all days combined, Classifiable
            path = "Z:\projects\ongoing\EnsAD\workspace\data\EnMAP_extracts\OWT_BalticSea\\"
            # combination of OWTs with the same phytoplankton groups:
            # OWT 1: C7
            # OWT 2 + 3a + 4a + 4b + 5a + 5b : C0, C2, C6
            # OWT 3b: C0, C2, C5, C6
            # OWT 6 + 7: C0, C1, C2, C6
            ### different restrictions for inversion!
            # OWTList = ['1'] -> no data
            # OWTList = ['2'] #-> chl_tot_max = 1, ISM_max=1, CDOM_max=0.1
            # OWTList = ['3a'] # -> chl_tot_max = 10, ISM_max=10, CDOM_max=1
            # OWTList = ['3b']
            # OWTList = ['4a'] # -> chl_tot_max = 30, ISM_max=10, CDOM_max=1
            # OWTList = ['4b'] # -> chl_tot_max = 300, ISM_max=10, CDOM_max=1
            # OWTList = ['5a'] # -> chl_tot_max = 300, ISM_max=40, CDOM_max=30
            # OWTList = ['5b'] # -> chl_tot_max = 1000, ISM_max=40, CDOM_max=30
            # OWTList = ['6']
            # OWTList = ['7']
            # OWTList = ['6', '7']
            fnameList = os.listdir(path)
            fnameList = [fn for fn in fnameList if datasetDate in fn]
            fnameList = [fn for fn in fnameList if datasetID in fn]

            d = pd.read_csv(path + fnameList[0], header=0, sep='\t')
            wlstr = [b for b in d.columns.values if not b in ['idx', 'idy', 'lon', 'lat', 'OWT', 'date', 'QWIP', 'AVW']]

            fnameL = []
            OWTstr = ''
            for owt in OWTList[:]:
                print(owt)
                [fnameL.append(fn) for fn in fnameList if 'OWT' + owt in fn]
                OWTstr += owt + '_'

            print(fnameL)

            datasetName = 'EnMAP_BalticSea_OWT' + OWTstr + datasetDate
            if os.path.exists(outpath + 'extracts_' + datasetName + '.txt'):
                print('read directly')
                d = pd.read_csv(outpath + 'extracts_' + datasetName + '.txt', sep='\t', header=0)
                Rrs = d[wlstr].copy()
            else:
                Rrs = None
                meta = None
                for fn in fnameL:
                    d = pd.read_csv(path + fn, sep='\t', header=0)

                    if Rrs is None:
                        Rrs = d[wlstr].copy()
                        meta = d.copy()
                    else:
                        Rrs = pd.concat([Rrs, d[wlstr]])
                        meta = pd.concat([meta, d])

                meta.to_csv(outpath + 'extracts_' + datasetName + '.txt', sep='\t', header=True, index=False)

            # simple correction of negative Rrs
            r_rs = Rrs.copy()
            minRrs = np.min(Rrs.values, axis=1)
            print(minRrs.shape)
            ID = np.array(minRrs < 0)
            rrs = Rrs.values.copy()
            for i in range(Rrs.shape[1]):  # iterate along wavelengths
                rrs[ID, i] = rrs[ID, i] - minRrs[ID]

            wavelengths = np.asarray([float(a) for a in wlstr])
            r_rs.loc[:, :] = rrs.copy()
    # if dataType=='PCA':
    #     if seaName == 'North Sea':



    return r_rs, wlstr, wavelengths, datasetName