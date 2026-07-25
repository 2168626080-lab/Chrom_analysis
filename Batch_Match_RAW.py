import pandas as pd
import numpy as np
import pickle
import warnings
from matchms import Spectrum
from matchms.filtering import default_filters, normalize_intensities, add_precursor_mz, require_minimum_number_of_peaks
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, PrecursorMzMatch, ModifiedCosineGreedy, MetadataMatch
import time as t
import logging
# 屏蔽 matchms 的警告日志
logging.getLogger('matchms').setLevel(logging.ERROR)
# 或者更彻底地屏蔽所有警告
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
print(t.ctime())

# ====================== 参数设置 ======================
csv_file = r"F:\Xcalibur\新建文件夹 (2)\样品\negative3.csv"
output_csv = r"F:\Xcalibur\新建文件夹 (2)\样品\negative3配对MoNA.csv"
pickle_file = r"F:\Xcalibur\massbank_cleaned.pkl"  # ← 你的二进制文件路径
pickle_file2 = r"F:\Xcalibur\MoNAbank_cleaned.pkl"  # 另一个库

top_n = 1  # 输出前十相似母离子,在103行被注释，因为我只关心first
tolerance_da = 0.02  # 二级碎片质荷比容差（单位 Da），可改成 0.01 或 0.05
min_peaks = 5  # 过滤二级信息不足的样本，样本二级碎片少于5！
# ====================================================

print("正在加载 pickle 参考库...")
with open(pickle_file2, "rb") as f:
    references = pickle.load(f)

print(f"成功加载 {len(references)} 条参考谱图")

df = pd.read_csv(csv_file, encoding='utf-8')  # 'gb2312' 或 'utf-8'

queries = []
for idx, row in df.iterrows():
    # 处理分号分隔的峰数据
    mz_list = [float(x.strip()) for x in str(row['fragments_mz']).split(';') if str(x).strip()]
    intensity_list = [float(x.strip()) for x in str(row['intensities_rel']).split(';') if str(x).strip()]

    if len(mz_list) != len(intensity_list) or len(mz_list) < min_peaks:
        continue

    mz = np.array(mz_list, dtype=float)
    intensities = np.array(intensity_list, dtype=float)

    metadata = {
        "id": str(row.get('id')),
        "name": str(row.get('name')),  # 无用字段，可以在此处和读样本中略去,但可以用queries.metadata读出
        "formula": str(row.get('formula')),
        "precursor_mz": float(row['precursor_mz']),
        'parent_mass': float(row['precursor_mz']),
        "ionmode": str(row.get('ion_mode')),
        "adduct": str(row.get('precursor_type')),
        "rt": float(row.get('rt')),
    }

    spectrum = Spectrum(
        mz=mz,
        intensities=intensities,
        metadata=metadata,
        metadata_harmonization=True
    )

    # 清洗
    spectrum = default_filters(spectrum)
    spectrum = add_precursor_mz(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=min_peaks)

    if spectrum is not None:
        queries.append(spectrum)

print(f"成功创建 {len(queries)} 个查询谱图")
print("开始匹配...")

print("计算母离子匹配...")
mz_matcher = MetadataMatch(field="precursor_mz",
                           matching_type="difference",
                           tolerance=0.05)  # 母离子道尔顿容差
scores_mz = calculate_scores(
    references=references,
    queries=queries,
    similarity_function=mz_matcher,
    array_type="numpy")

# mz_matcher = PrecursorMzMatch(tolerance=10,  # 母离子10ppm容差
#                               tolerance_type='ppm')
# scores_mz = calculate_scores(
#     references=references,
#     queries=queries,
#     similarity_function=mz_matcher,
#     array_type="numpy")

similarity_function = CosineGreedy(tolerance=tolerance_da)
score_name = "CosineGreedy_score"

results = []

for i, query in enumerate(queries):
    query_name = query.get('compound_name') or query.get('name') or f"Query_{i + 1}"
    query_pmz = query.get('precursor_mz')

    print(f"\n{'=' * 80}")
    print(f"【查询 {i + 1}】 {query_name} | precursor_mz = {query_pmz:.4f}")
    print('=' * 80)

    # 默认全 None
    match_data = {
        "query_name": query_name,
        "query_precursor_mz": round(query_pmz, 4),
        "ref_name": None,
        "ref_inchikey": None,
        "ref_adduct": None,
        "ref_formula": None,
        "ref_smiles": None,
        "ref_ionmode": None,
        "score": None,
        "mz_delta": None,
    }

    try:
        # 1. 当前 query 的母离子匹配
        matched_refs = []
        for ref, score in scores_mz.scores_by_query(query=query):
            if score:
                matched_refs.append(ref)

        print(f"  → 母离子匹配到 {len(matched_refs)} 条参考谱图")

        if len(matched_refs) == 0:
            print("  → 无母离子匹配")
            results.append(match_data)
            continue

        # 2. 当前 query 单独计算谱图相似度
        scores_spec = calculate_scores(
            references=matched_refs,
            queries=[query],
            similarity_function=similarity_function,
            array_type="numpy"
        )

        # 3. 提取最佳匹配
        best_matches = scores_spec.scores_by_query(query, name=score_name, sort=True)

        if len(best_matches) == 0:
            print("  → 无谱图匹配结果")
            results.append(match_data)
            continue
        # best_matches=[（reference, score）,...]
        top = best_matches[0]
        ref_spectrum = top[0]
        score_value = top[1][0]
        mz_delta = abs(query_pmz - ref_spectrum.get('precursor_mz', 0))

        print(f"  → 最佳匹配: {ref_spectrum.get('name') or ref_spectrum.get('compound_name') or 'Unknown'}")
        print(f"  → 分数: {score_value:.4f}")

        # 更新匹配结果
        match_data.update({

            "ref_name": ref_spectrum.get('name') or ref_spectrum.get('compound_name'),
            "ref_inchikey": ref_spectrum.get('inchikey'),
            "ref_adduct": ref_spectrum.get('adduct'),
            "ref_formula": ref_spectrum.get('formula'),
            "ref_smiles": ref_spectrum.get('smiles'),
            "ref_ionmode": ref_spectrum.get('ionmode'),
            "score": float(score_value),
            "mz_delta": round(mz_delta, 4),

        })

    except Exception as e:
        print(f"  → 匹配过程出错: {e}")
        # 出错时保持 None 值

    results.append(match_data)

# ====================== 最终输出 ======================
df_results = pd.DataFrame(results)
print("\n" + "=" * 80)
print("所有查询最终匹配结果")
print("=" * 80)
print(df_results)

# 保存结果（取消注释即可）
df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
