import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from fisher_py import RawFile
from fisher_py.data.business import TraceType
import os
import time as t
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from fisher_py.data.business.mass_options import ToleranceUnits

# k = 20  # 每个文件取前 k 个最强 BasePeak, 138行使用
# l = 2  # 每个峰取前 l 个最强母离子
mass_tolerance_ppm = 5  # bpc占位参数，无用
threshold_BasePeak_area = 0.02  # 设置门限为最大BasePeak峰值5%0.02
threshold_ms1_intensity = 0.02  # 设置门限为单个ms1列表中最大强度值5%0.02
max_workers = 8
# ====================================================================

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def AreaBasePeak(rt_bpc, int_bpc, threshold: float):
    # 修复：只找峰，不提前过滤
    peaks_idx, _ = find_peaks(int_bpc, distance=5)
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
        "峰高": peak_heights,
        "峰面积": peak_areas
    })

    # 先按面积排序
    peak_table = peak_table.sort_values("峰面积", ascending=False).reset_index(drop=True)

    max_area = peak_table["峰面积"].max()
    peak_table = peak_table[peak_table["峰面积"] >= max_area * threshold].reset_index(drop=True)

    peak_table["排名"] = range(1, len(peak_table) + 1)
    return peak_table


# 修改 IntensityMs1 函数，门限已经在内部实现了，但确保参数正确使用
def IntensityMs1(mz1_values, intensities, charges=None, min_intensity_ratio=threshold_ms1_intensity):
    max_inten = np.max(intensities) if len(intensities) > 0 else 1
    intensity_threshold = max_inten * min_intensity_ratio

    filtered = []
    for i in range(len(mz1_values)):
        mz = mz1_values[i]
        inten = intensities[i]
        z = charges[i] if charges is not None else 0

        if inten < intensity_threshold:
            continue

        filtered.append((mz, inten, int(z)))

    filtered.sort(key=lambda x: x[1], reverse=True)
    if not filtered:
        return pd.DataFrame(columns=["m/z"])
    mz_out, _, _ = zip(*filtered)
    return pd.DataFrame({"m/z": np.round(mz_out, 5)})


# 获取 MS2 碎片 → 输出 mz列表;强度列表（和MassBank完全一致）
def GetMs2Pairs(raw_file, rt, precursor_mz):
    # 有的有母离子和保留时间，但是机器没有记录其二级信息，所以需要使用try语句
    try:
        mz2, i2, charges2, real_rt2 = raw_file.get_scan_ms2(rt=rt, precursor_mz=precursor_mz)
        if len(mz2) == 0:
            return None, None, None, None

        if len(mz2) < 3:
            print(f"⚠️ MS2碎片太少 ({len(mz2)} < {3})，丢弃")
            return None, None, None, None

        # 转 相对强度 (max=100)
        i2 = np.array(i2)
        max_i = i2.max()
        rel_i = (i2 / max_i * 100).round(2)

        mz_str = ";".join([f"{x:.4f}" for x in mz2])
        int_str = ";".join([f"{x}" for x in rel_i])
        return mz_str, int_str, charges2, real_rt2
    except:
        return None, None, None, None


# XIC 峰面积（取最强峰面积）, 这里的threshold起到控制寻峰作用，只是筛掉同位素峰，实际上只取一个而已
def AreaXIC(rt_xic, intensity_xic, target_rt, sigma, rt_window=0.3, threshold=0.05):
    mask = (rt_xic >= target_rt - rt_window) & (rt_xic <= target_rt + rt_window)
    rt_win = rt_xic[mask]
    int_win = intensity_xic[mask]
    if len(int_win) == 0:
        return 0.0

    int_win_smooth = gaussian_filter1d(int_win, sigma=sigma)

    peaks_idx, _ = find_peaks(int_win_smooth, height=np.max(int_win_smooth) * threshold, prominence=np.max(int_win_smooth) * 0.3)
    if len(peaks_idx) == 0:
        return 0.0
    results = peak_widths(int_win_smooth, peaks_idx, rel_height=1)
    widths, hts, l_idx_arr, r_idx_arr = results
    areas = []
    for i in range(len(peaks_idx)):
        l = int(l_idx_arr[i])
        r = int(r_idx_arr[i])
        area = simpson(int_win_smooth[l:r + 1], rt_win[l:r + 1])
        areas.append(area)
    return round(max(areas), 2) if areas else 0.0


# 并行处理（定义了每一个文件的处理方式）
def process_single_file(file_name, raw_path, mass_tolerance_ppm, threshold_BasePeak_area, threshold_ms1_intensity, sigma):
    """处理单个raw文件的核心函数"""
    print(f"  [进程{os.getpid()}] 开始处理：{file_name}")

    raw = RawFile(raw_path)
    # 1. BPC 峰（应用门限）
    rt_bpc, int_bpc = raw.get_chromatogram(mz=0.0,
                                           tolerance=mass_tolerance_ppm,  # 获取bpc时，tolerance参数无用，但我懒得修改了，可以不填
                                           tolerance_units=ToleranceUnits.ppm,
                                           trace_type=TraceType.BasePeak)
    bpc_table = AreaBasePeak(rt_bpc, int_bpc, threshold=threshold_BasePeak_area)

    # 取前 k 个BasePeak峰，加.head()，测试用，实战用门限threshold_Base控制
    topk = bpc_table
    if len(topk) == 0:
        print(f"[进程{os.getpid()}] {file_name}: 未找到有效BPC峰")
        return []
    print(
        f"[进程{os.getpid()}] {file_name}: 找到 {len(AreaBasePeak(rt_bpc, int_bpc, threshold=0.0))} 个BPC峰，门限{threshold_BasePeak_area}后剩余{len(topk)}个")

    file_results = []
    for bp_idx, bp in topk.iterrows():
        rt = bp["峰顶RT"]
        bp_num = bp_idx + 1

        # 2. MS1（应用门限）
        mz1, i1, charges1, real_rt = raw.get_scan_ms1(rt)
        ms1_table = IntensityMs1(mz1, i1, charges1, min_intensity_ratio=threshold_ms1_intensity)

        # 取前 l 个母离子，加.head()，测试用，实战用门限threshold控制
        topl = ms1_table
        if len(topl) == 0:
            print(f"[进程{os.getpid()}] BP{bp_num} RT={rt:.2f}: 无有效母离子")
            continue

        print(
            f"[进程{os.getpid()}] BP{bp_num} RT={rt:.2f}: 找到 {len(IntensityMs1(mz1, i1, charges1, min_intensity_ratio=0.0))} 个母离子，门限{threshold_ms1_intensity}取{len(topl)}个")

        for ion_idx, ion in topl.iterrows():
            ion_num = ion_idx + 1
            prec_mz = float(ion["m/z"])

            # 3. 计算XIC面积
            rt_xic, int_xic = raw.get_chromatogram(prec_mz,
                                                   tolerance=mass_tolerance_ppm,
                                                   tolerance_units=ToleranceUnits.ppm,
                                                   trace_type=TraceType.MassRange)

            # 4. MS2
            mz_str, int_str, charges2, real_rt2 = GetMs2Pairs(raw, real_rt, prec_mz)
            if mz_str is None:
                print(f"[进程{os.getpid()}] 母离子 m/z={prec_mz:.4f}: 无MS2信息，跳过")
                continue
            xic_area = AreaXIC(rt_xic, int_xic, real_rt2, sigma=sigma)
            # 整合行数据
            file_results.append({
                "文件": file_name,
                "BasePeak序号": bp_num,
                "母离子序号": ion_num,
                "id": f"{file_name}_BP{bp_num}_ION{ion_num}",
                "name": f"{file_name}_RT{real_rt:.2f}_mz{prec_mz:.4f}",
                "formula": "",
                "ion_mode": "Negative",  # 根据实际修改
                "precursor_mz": round(prec_mz, 5),
                "XIC峰面积": xic_area,
                "rt": round(real_rt, 5),  # scan_ms1返回的只是在BasePeak的精确时间吧，scan_ms2返回的是母离子XIC的精确时间
                "num_peak": len(mz_str.split(";")),
                "fragments_mz": mz_str,
                "intensities_rel": int_str,
            })
            print(
                f"      [进程{os.getpid()}] 母离子 m/z={prec_mz:.4f}: XIC面积={xic_area:.2e}, MS2碎片数={len(mz_str.split(';'))} ✓")

    print(f"  [进程{os.getpid()}] {file_name}: 完成处理，提取{len(file_results)}条数据")
    return file_results


if __name__ == '__main__':
    start_time = t.time()
    print(f"开始时间: {t.ctime()}")
    print(f"使用 {max_workers} 个进程并行处理\n")

    # 收集所有需要处理的文件
    files_to_process = []
    # 遍历样品 - 修改为你的实际文件列表
    for i in ["A", "B", "C", "D", "E"]:  # 添加更多样品1, 2, 3, 4, 5 6, 7, 8, 9, 10,  "A", "B", "C", "D", "E"
        for j in [1, 2, 3]:  # 添加更多编号, 2, 3, 4, 5, 6
            file_name = f"{i}-{j}"
            raw_path = f"F:\Xcalibur\新建文件夹 (3)\{i}-{j}.raw"
            if os.path.exists(raw_path):
                files_to_process.append((file_name, raw_path))

    if not files_to_process:
        print("未找到任何.raw文件！")
        exit()

    print(f"找到 {len(files_to_process)} 个文件待处理\n")

    # 创建偏函数，固定参数
    process_func = partial(process_single_file,
                           mass_tolerance_ppm=mass_tolerance_ppm,
                           threshold_BasePeak_area=threshold_BasePeak_area,
                           threshold_ms1_intensity=threshold_ms1_intensity,
                           sigma=5
                           )

    all_results = []
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_func, file_name, raw_path): file_name
            for file_name, raw_path in files_to_process
        }

        # 收集结果
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]

            results = future.result()
            all_results.extend(results)
            print(f"✓ 完成 {file_name}，累计提取 {len(all_results)} 条数据\n")

    # 输出 CSV
    if all_results:
        df_out = pd.DataFrame(all_results)
        # 可选：按文件、BasePeak序号、母离子序号排序
        df_out = df_out.sort_values(["文件", "BasePeak序号", "母离子序号"])
        out_path = r"F:\Xcalibur\新建文件夹 (3)\样品\positive.csv"
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        elapsed_time = t.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"全部完成！")
        print(f"处理文件数: {len(files_to_process)}")
        print(f"共提取有效数据: {len(all_results)} 条")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均每个文件耗时: {elapsed_time / len(files_to_process):.2f} 秒")
        print(f"已保存：{out_path}")
        print(f"结束时间: {t.ctime()}")
        print(f"{'=' * 60}")
