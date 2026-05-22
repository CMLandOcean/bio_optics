import numpy as np
import pandas as pd


def calculate_QWIP_hyperspectral(Rrs, wl_, maxwl=700):
    ID2 = np.logical_and(np.array(wl_ >= 400), np.array(wl_ <= maxwl))
    wlVNIR = wl_[ID2]

    if isinstance(Rrs, pd.DataFrame):
        Rrs = Rrs.loc[:, ID2].values
    else:
        Rrs = Rrs[:, ID2]

    avw_sum = np.nansum(Rrs, axis=1)
    r_avw_sum = np.copy(Rrs)
    for i, wl in enumerate(wlVNIR):
        r_avw_sum[:, i] = r_avw_sum[:, i] / wl
    r_avw_sum = np.nansum(r_avw_sum, axis=1)

    ID665 = np.abs(wlVNIR - 665.) == np.min(np.abs(wlVNIR - 665.))
    ID490 = np.abs(wlVNIR - 490.) == np.min(np.abs(wlVNIR - 490.))

    # NDI = (Rrs.loc[:, ID665].values[:, 0] - Rrs.loc[:, ID490].values[:, 0]) / (
    #         Rrs.loc[:, ID665].values[:, 0] + Rrs.loc[:, ID490].values[:, 0])
    NDI = (Rrs[:, ID665][:, 0] - Rrs[:, ID490][:, 0]) / (
            Rrs[:, ID665][:, 0] + Rrs[:, ID490][:, 0])
    AVW = avw_sum / r_avw_sum
    NDI_pred = -8.399885 * 1.0e-9 * np.power(AVW, 4) + 1.715532 * 1.0e-5 * np.power(AVW, 3) \
               - 1.301670 * 1.0e-2 * np.power(AVW, 2) + 4.357838 * AVW - 5.449532 * 1.0e2
    qwip_score = NDI_pred - NDI
    return AVW, NDI, qwip_score


def calculate_area(Rrs, band, sensor_RGB_bands=[443, 560, 665]):
    print(Rrs.shape)
    bands_for_Area = np.array(sensor_RGB_bands)
    band_id = []
    for b in sensor_RGB_bands:
        ID = np.where(np.abs(band - b) == np.min(np.abs(band - b)))[0]
        band_id.append(ID)

    band_id = np.asarray(band_id).flatten()
    print(band_id.flatten())
    actual_bands_wl = np.asarray(band)[band_id].flatten()
    print(actual_bands_wl)
    Rrs_for_Area = Rrs[:, band_id]
    print(Rrs_for_Area.shape)
    Area = np.trapezoid(x=np.asarray(actual_bands_wl), y=Rrs_for_Area, axis=1)
    return Area

def plot_QWIP_AVW_byOWT(ndi, avw, owtDF, owt_names, colorMapDict, ax, wlmin=450., wlmax=650.):
    owt_result = owtDF['OWT']

    for i, owt in enumerate(owt_names):
        ID = owt_result == i
        if np.sum(ID)>0:
            col = colorMapDict[i]
            ax.plot(avw[ID], ndi[ID], '+', color=col, label=owt)

    delta = 0.2
    AVW = np.arange(wlmin, wlmax, 1.)
    NDI_pred = -8.399885 * 1.0e-9 * np.power(AVW, 4) + 1.715532 * 1.0e-5 * np.power(AVW, 3) \
               - 1.301670 * 1.0e-2 * np.power(AVW, 2) + 4.357838 * AVW - 5.449532 * 1.0e2
    ax.plot(AVW, NDI_pred, 'k-')
    ax.plot(AVW, NDI_pred+delta, 'k--')
    ax.plot(AVW, NDI_pred-delta, 'k--')
    ax.legend()
    ax.set_ylabel('NDI 665-490nm')
    ax.set_xlabel('AVW [nm]')
    return ax