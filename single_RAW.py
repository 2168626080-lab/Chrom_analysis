import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from fisher_py import RawFile
from fisher_py.data.business import TraceType
# from fisher_py.data.tolerance_units import ToleranceUnits
from scipy.ndimage import gaussian_filter1d
from fisher_py.data.business.mass_options import ToleranceUnits


"本文档已经经B-1~5的一级碎片检验过"

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


# BPC 峰面积排序
def AreaBasePeak(rt_bpc, int_bpc, threshold=0.0005):
    peaks_idx, _ = find_peaks(int_bpc, height=np.max(int_bpc) * threshold, distance=5)
    results_half = peak_widths(int_bpc, peaks_idx, rel_height=0.5)
    widths, width_heights, left_indices, right_indices = results_half

    left_rts = np.interp(left_indices, np.arange(len(rt_bpc)), rt_bpc)
    right_rts = np.interp(right_indices, np.arange(len(rt_bpc)), rt_bpc)
    peak_rts = rt_bpc[peaks_idx]
    peak_heights = int_bpc[peaks_idx]

    peak_areas = []
    for i in range(len(peaks_idx)):
        l_rt = left_rts[i]
        r_rt = right_rts[i]
        l_idx = np.argmin(np.abs(rt_bpc - l_rt))
        r_idx = np.argmin(np.abs(rt_bpc - r_rt))
        area = simpson(int_bpc[l_idx:r_idx + 1], rt_bpc[l_idx:r_idx + 1])
        peak_areas.append(area)

    peak_table = pd.DataFrame({
        "排名": range(1, len(peak_rts) + 1),
        "峰顶RT": peak_rts,
        # "左RT": left_rts,
        # "右RT": right_rts,
        "峰高": peak_heights,
        "峰面积": peak_areas
    })

    peak_table = peak_table.sort_values("峰面积", ascending=False).reset_index(drop=True)
    peak_table["排名"] = range(1, len(peak_table) + 1)
    return peak_table


# 不是抓取ms1，只是算强度排序<
def IntensityMs1(mz1_values, intensities, charges=None, threshold=0.0005):
    """
    非靶向代谢组学 标准母离子筛选 + 排序
    逻辑：去噪声 + 去垃圾峰 + 按强度排序 + 输出排名表
    """
    # 强度阈值：最低强度 = 最强峰的 5%（标准去噪）
    max_inten = np.max(intensities) if len(intensities) > 0 else 1
    intensity_threshold = max_inten * threshold

    # 筛选有效离子
    filtered_mz = []
    filtered_inten = []
    filtered_charge = []

    for i in range(len(mz1_values)):
        mz = mz1_values[i]
        inten = intensities[i]
        z = charges[i] if charges is not None else 0

        # 标准过滤条件
        if inten < intensity_threshold:
            continue

        filtered_mz.append(mz)
        filtered_inten.append(inten)
        filtered_charge.append(int(z))

    # 按强度降序排列（母离子排名核心依据）
    combined = list(zip(filtered_mz, filtered_inten, filtered_charge))
    combined.sort(key=lambda x: x[1], reverse=True)

    if not combined:
        return pd.DataFrame(columns=["排名", "m/z", "强度", "电荷", "相对强度%"])

    mz_out, inten_out, z_out = zip(*combined)

    # 计算相对强度（科研标准）
    rel_intensity = np.array(inten_out) / max_inten * 100

    # 输出表格
    df = pd.DataFrame({
        "排名": range(1, len(mz_out) + 1),
        "m/z": np.round(mz_out, 4),
        # "强度": np.round(inten_out, 1),
        # "电荷": z_out,
        "相对强度%": np.round(rel_intensity, 2)
    })
    return df


# 不是抓取ms2，只是算强度排序
def IntensityMs2(mz2_values, intensities, charges=None, min_intensity_ratio=None):
    """
        MS2 碎片离子排名（二级质谱子离子输出）
        按强度降序 + 输出排名表
        """
    # 强度阈值（MS2通常噪声更低，用 0.5% 更合适）
    max_inten = np.max(intensities) if len(intensities) > 0 else 1

    filtered_mz = []
    filtered_inten = []
    filtered_charge = []

    for i in range(len(mz2_values)):
        mz = mz2_values[i]
        inten = intensities[i]
        z = charges[i] if (charges is not None and i < len(charges)) else 0

        filtered_mz.append(mz)
        filtered_inten.append(inten)
        filtered_charge.append(int(z))

    # 按强度降序排列
    combined = list(zip(filtered_mz, filtered_inten, filtered_charge))
    combined.sort(key=lambda x: x[1], reverse=True)

    if not combined:
        return pd.DataFrame(columns=["排名", "m/z", "相对强度%"])

    mz_out, inten_out, _ = zip(*combined)
    rel_intensity = np.array(inten_out) / max_inten * 100

    df = pd.DataFrame({
        "排名": range(1, len(mz_out) + 1),
        "m/z": np.round(mz_out, 4),
        "相对强度%": np.round(rel_intensity, 2)
    })
    return df


# 不是抓取XIC，只是算面积
def AreaXIC(rt_xic, intensity_xic, target_rt, rt_window, sigma):
    mask = (rt_xic >= target_rt - rt_window) & (rt_xic <= target_rt + rt_window)
    rt_windowed = rt_xic[mask]
    int_windowed = intensity_xic[mask]

    int_win_smooth = gaussian_filter1d(int_windowed, sigma=sigma)

    if len(int_windowed) == 0:
        return pd.DataFrame({
            "排名": [0], "XIC峰面积": [0.0], "相对面积%": [0.0], "左RT": [0.0], "右RT": [0.0]
        })

    peaks_idx, _ = find_peaks(int_win_smooth, height=np.max(int_win_smooth) * 0.05, prominence=np.max(int_win_smooth) * 0.3)
    print(peaks_idx)
    if len(peaks_idx) == 0:
        return pd.DataFrame({
            "排名": [0], "XIC峰面积": [0.0], "相对面积%": [0.0], "左RT": [0.0], "右RT": [0.0]
        })

    # 全高峰宽
    results = peak_widths(int_win_smooth, peaks_idx, rel_height=1)
    widths, hts, l_idx_arr, r_idx_arr = results

    areas = []
    left_rts = []
    right_rts = []

    for i in range(len(peaks_idx)):
        l_idx = int(l_idx_arr[i])
        r_idx = int(r_idx_arr[i])
        l_rt = rt_windowed[l_idx]
        r_rt = rt_windowed[r_idx]

        area = simpson(int_win_smooth[l_idx:r_idx + 1], rt_windowed[l_idx:r_idx + 1])
        areas.append(area)
        left_rts.append(l_rt)
        right_rts.append(r_rt)

    df = pd.DataFrame({
        "排名": range(1, len(areas) + 1),
        "XIC峰面积": np.round(areas, 2),
        "左RT": np.round(left_rts, 5),
        "RT": np.round(target_rt, 5),
        "右RT": np.round(right_rts, 5)
    })
    df = df.sort_values("XIC峰面积", ascending=False).reset_index(drop=True)
    df["排名"] = range(1, len(df) + 1)
    max_area = df["XIC峰面积"].max()
    df["相对面积%"] = np.round(df["XIC峰面积"] / max_area * 100, 2) if max_area > 0 else 0.0
    return df


if __name__ == '__main__':
    mass_tolerance_ppm = int(5)
    mass_tolerance = 0.02
    for i in ["A"]:
        print(i)
        print("=" * 80)
        for j in [1]:
            BPT = 0
            XIT = 0
            raw_path = f"F:\Xcalibur\新建文件夹 (3)\{i}-{j}.raw"
            raw_file = RawFile(raw_path)

            # 1.BPC
            rt_bpc, int_bpc = raw_file.get_chromatogram(mz=0.0,
                                                        tolerance=0.0,
                                                        tolerance_units=ToleranceUnits.ppm,
                                                        trace_type=TraceType.BasePeak)
            peak_table = AreaBasePeak(rt_bpc, int_bpc)
            print("BPC 半峰高面积排序")
            print(peak_table)
            print("=" * 80)

            # 2.取RT
            # 选取BasePeak峰：强到弱
            rt_rub = peak_table.iloc[BPT]['峰顶RT']
            mz1, i1, charges1, real_rt1 = raw_file.get_scan_ms1(rt_rub)
            precursor_table = IntensityMs1(mz1, i1, charges1)
            # 3.XIC
            # 选定母离子：强到弱
            precursor_mz = precursor_table.iloc[XIT]["m/z"]
            print(f"RT1 = {real_rt1:.4f} min 母离子排名：{precursor_mz}在该时间{rt_rub}最高")
            print(precursor_table)
            print("=" * 80)
            best_precursor_mz = float(precursor_mz)
            rt_xic, intensity_xic = raw_file.get_chromatogram(best_precursor_mz,
                                                              tolerance=mass_tolerance_ppm,
                                                              tolerance_units=ToleranceUnits.ppm,
                                                              trace_type=TraceType.MassRange)
            # 外部平滑，此处仅用于画图，AreaXIC内部已经平滑
            sigma = 5
            int_win_smooth_xic = gaussian_filter1d(intensity_xic, sigma=sigma)

            rt_window = 0.3
            # 4.MS2
            try:
                mz2, i2, charges2, real_rt2 = raw_file.get_scan_ms2(rt=real_rt1, precursor_mz=best_precursor_mz)
                areaXIC_table = AreaXIC(rt_xic, intensity_xic, target_rt=real_rt2, rt_window=rt_window, sigma=sigma)
                print(f"XIC 峰面积（m/z={best_precursor_mz:.4f}）")
                print(areaXIC_table.head())  # 仅第一个有用，其他为同位素峰
                print("=" * 80)
                product_table = IntensityMs2(mz2, i2, charges2)
                print(f"RT2 = {real_rt2:.4f} min | 母离子{precursor_mz}的MS2-二级碎片")
                print(product_table.head(3))
            except Exception as e:
                print(f"未找到 MS2 图谱：{e}")

            # # ================= 绘图 =================
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

            ax1.plot(rt_bpc, int_bpc, 'r-', linewidth=1)
            ax1.set_title('BasePeak')
            ax1.grid(alpha=0.3)

            ax2.plot(mz1, i1, 'r-', linewidth=1)
            ax2.set_title(f'{real_rt1:.5f} min MS1')
            ax2.grid(alpha=0.3)

            ax3.plot(mz2, i2, 'r-', linewidth=1)
            ax3.set_title(f'{real_rt2:.5f} min MS2')
            ax3.grid(alpha=0.3)

            ax4.plot(rt_xic, int_win_smooth_xic, 'b-', linewidth=0.5)
            ax4.set_title(f'XIC m/z {best_precursor_mz:.4f}')
            ax4.grid(alpha=0.3)


            if not areaXIC_table.empty and areaXIC_table.iloc[0]["XIC峰面积"] > 0:
                left_rt = areaXIC_table.iloc[0]["左RT"]
                right_rt = areaXIC_table.iloc[0]["右RT"]
                fill_mask = (rt_xic >= left_rt) & (rt_xic <= right_rt)
                ax4.fill_between(rt_xic[fill_mask], int_win_smooth_xic[fill_mask], color='red', alpha=0.4)

            ax4.legend(['XIC', '真实积分区域'])
            plt.tight_layout()
            plt.show()
