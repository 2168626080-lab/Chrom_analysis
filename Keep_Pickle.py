from matchms.importing import load_from_msp
from matchms.filtering import default_filters, normalize_intensities, add_precursor_mz, require_minimum_number_of_peaks
import pickle

msp_file = r"F:\Xcalibur\MoNA-export-LC-MS_Spectra.msp"
pickle_file = r"F:\Xcalibur\MoNAbank_cleaned.pkl"   # ← 推荐保存成这个

# 读取并清洗（只执行一次）
print("正在读取并清洗 MSP...")
references = list(load_from_msp(msp_file, metadata_harmonization=True))

processed = []
for spec in references:
    spec = default_filters(spec)
    spec = add_precursor_mz(spec)
    spec = normalize_intensities(spec)
    spec = require_minimum_number_of_peaks(spec, n_required=5)
    if spec is not None:
        processed.append(spec)

print(f"清洗后保留 {len(processed)} 条谱图")

# 保存为 pickle（非常快）
with open(pickle_file, "wb") as f:
    pickle.dump(processed, f)

print(f"已保存到 {pickle_file}")
