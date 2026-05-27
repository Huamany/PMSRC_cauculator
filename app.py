import math
import numpy as np
import pandas as pd
import streamlit as st
import io

# =================== 页面配置 ===================
st.set_page_config(page_title="混合岩石流变强度计算器", layout="wide")
st.title("🪨 混合岩石流变强度计算器")
st.markdown("通过上传各矿物相的体积随温度变化的 CSV 文件，计算部分熔融混合岩的流变强度。")

# ----------------- 1. 数据读取与预处理 -----------------
def read_data_from_uploads(uploaded_files):
    """
    读取 Streamlit 上传的 CSV 文件，返回字典 data。
    """
    data = {}
    file_mapping = {
        "b.csv": "biotite",
        "a.csv": "amphibole",
        "k.csv": "K-feldspar",
        "q.csv": "quartz",
        "p.csv": "plagioclase",
        "l.csv": "liquid"
    }
    
    for uploaded_file in uploaded_files:
        fname = uploaded_file.name
        if fname not in file_mapping:
            continue
            
        phase = file_mapping[fname]
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        
        if phase == "liquid":
            df_liquid = df[df["Name"] == "liquid"]
            df_liquid_ex = df[df["Name"] == "liquid_Ex"]
            series_liquid = pd.Series(df_liquid["Vol%"].values / 100, index=df_liquid["T /K"].values).sort_index()
            series_liquid_ex = pd.Series(df_liquid_ex["Vol%"].values / 100, index=df_liquid_ex["T /K"].values).sort_index()
            data["liquid"] = series_liquid
            data["liquid_ex"] = series_liquid_ex
        else:
            series = pd.Series(df["Vol%"].values / 100, index=df["T /K"].values).sort_index()
            data[phase] = series
    return data

# ----------------- 2. 辅助函数：插值获取标量数值 -----------------
def get_scalar(series, T):
    try:
        val = series.loc[T]
        if isinstance(val, pd.Series):
            return float(val.iloc[0])
        return float(val)
    except KeyError:
        xs = series.index.astype(float)
        ys = series.values.astype(float)
        return float(np.interp(T, xs, ys))

# ----------------- 3. 计算各矿物强度 -----------------
def compute_sigma_i(T, creep_params, R, strain_rate):
    sigma_i = {}
    for phase, params in creep_params.items():
        Q = params["Q"] * 1000.0  # 转换为 J/mol
        n = params["n"]
        A = params["A"]
        sigma_i[phase] = ((strain_rate / A) * math.exp(Q / (R * T))) ** (1.0 / n)
    return sigma_i

# ----------------- 4. 固相全岩强度计算（未产生熔体阶段） -----------------
def compute_whole_rock_strength(sigma_i, vol_b, vol_a, vol_k, vol_p, vol_q):
    J_s = -0.25
    solid_phases = ["biotite", "amphibole", "K-feldspar", "quartz", "plagioclase"] 
    total_sol_vol = vol_b + vol_a + vol_k + vol_q + vol_p 
    
    sum_term = 0.0
    for ph in solid_phases:
        if ph in sigma_i:
            if ph == "biotite": vol = vol_b
            elif ph == "amphibole": vol = vol_a
            elif ph == "K-feldspar": vol = vol_k
            elif ph == "quartz": vol = vol_q
            elif ph == "plagioclase": vol = vol_p
            frac = vol / total_sol_vol if total_sol_vol > 0 else 0.0
            sum_term += frac * (sigma_i[ph] ** J_s)

    return (sum_term ** (1.0 / J_s)) if sum_term > 0 else 0.0

# ----------------- 5. 划分浅色体与暗色体体积分数 -----------------
def get_volume_partition(process_type, vol_b, vol_a, vol_k, vol_q, vol_p, vol_liq):
    if process_type == "prograde":
        vol_light = vol_liq + 0.2 * vol_a if vol_k <= 0 else vol_liq + vol_k + 0.2 * vol_a
    elif process_type == "retrograde":
        vol_light = vol_liq + vol_k + 0.2 * vol_a 
    else:
        raise ValueError("process_type 参数错误")
    
    vol_dark = vol_b + vol_q + vol_p + 0.8 * vol_a  
    return vol_light, vol_dark

# ----------------- 6. 计算暗色体强度 -----------------
def compute_dark_strength_1(sigma_i, vol_b, vol_a, vol_q, vol_p):
    J_dark = -0.25
    total_dark = vol_b + vol_q + vol_p + 0.8 * vol_a
    sum_term = 0.0
    if total_dark > 0:
        if vol_b > 0 and "biotite" in sigma_i: sum_term += (vol_b / total_dark) * (sigma_i["biotite"] ** J_dark)
        if vol_q > 0 and "quartz" in sigma_i: sum_term += (vol_q / total_dark) * (sigma_i["quartz"] ** J_dark)
        if vol_p > 0 and "plagioclase" in sigma_i: sum_term += (vol_p / total_dark) * (sigma_i["plagioclase"] ** J_dark)
        if vol_a > 0 and "amphibole" in sigma_i: sum_term += ((0.8 * vol_a) / total_dark) * (sigma_i["amphibole"] ** J_dark)
            
    return (sum_term ** (1.0 / J_dark)) if sum_term > 0 else 0.0

def compute_dark_strength_2(sigma_whole, vol_liq, strain_rate, R, T):
    if vol_liq < 0.02:
        J = 0.5
        return sigma_whole * ((1 - vol_liq) ** (1 / J))  
    else:
        d, v_s, c, Dm = 0.01, 10e-4, 10e3, 10e-15
        return (3 * strain_rate * d * d * R * T) / (8 * v_s * v_s * c * Dm * (vol_liq) * (vol_liq)) / 1e6

# ----------------- 7. 计算浅色体强度 -----------------
def compute_light_strength(T, sigma_i, vol_liq, vol_k, vol_a, strain_rate, R):
    leu_total = vol_liq + vol_k + 0.2 * vol_a
    phi_leu = vol_liq / leu_total if leu_total > 0 else 0.0
    Vs = 1.0 - phi_leu

    if phi_leu < 0.02:
        sigma_f, sigma_a = sigma_i.get("K-feldspar", 0.0), sigma_i.get("amphibole", 0.0) 
        total_matrix = vol_k + 0.2 * vol_a
        Vw = vol_k / total_matrix if total_matrix > 0 else 0.0
        J = sigma_a / sigma_f if sigma_f > 0 else 1.0

        mix_term = 0.0
        if sigma_f > 0: mix_term += sigma_f * (Vw ** (1.0 / J))
        if sigma_a > 0: mix_term += sigma_a * ((1 - Vw) ** (1.0 / J))
        return mix_term if mix_term > 0 else sigma_f
        
    elif phi_leu <= 0.40:
        d, v_s, c, Dm = 0.01, 10e-4, 10e3, 10e-15
        return (3 * strain_rate * d * d * R * T) / (8 * v_s * v_s * c * Dm * (1 - Vs) * (1 - Vs)) / 1e6
    else:
        eta_w, J_8 = 10e6, -0.4
        return eta_w * strain_rate * ((1 - (Vs / phi_leu)) ** (1 / J_8))

# ----------------- 8. 计算全岩强度（混合律） -----------------
def compute_overall_strength(sigma_dark, sigma_light, vol_dark, vol_light, vol_liq):
    J_mix = -0.05
    total = vol_light + vol_dark
    X_light = vol_light / total if total > 0 else 0.0
    mix_sum = X_light * (sigma_light ** J_mix)
    if sigma_dark > 0:
        mix_sum += (1 - X_light) * (sigma_dark ** J_mix)
    return mix_sum ** (1.0 / J_mix)

# ----------------- 9. Streamlit 前端交互与主逻辑 -----------------
st.sidebar.header("📁 1. 文件上传")
st.sidebar.markdown("请上传对应矿物相的 CSV 文件 (`b.csv`, `a.csv`, `k.csv`, `q.csv`, `p.csv`, `l.csv`)")
uploaded_files = st.sidebar.file_uploader("选择 CSV 文件", accept_multiple_files=True, type=['csv'])

st.header("⚙️ 2. 流变参数 (Creep Parameters) 调整")
st.markdown("您可以直接在下方的表格中双击修改各个矿物的 $Q$, $n$, $A$ 参数。")

# 默认流变参数
default_params = {
    "biotite":     {"Q": 51.0, "n": 18.0, "A": 10 ** -29.9},
    "quartz":      {"Q": 163.0, "n": 2.4, "A": 10 ** -5.0},
    "K-feldspar":  {"Q": 356.0, "n": 3.0, "A": 10 ** 2.6},
    "plagioclase": {"Q": 310.0, "n": 3.0, "A": 400.0},
    "amphibole":   {"Q": 154.0, "n": 3.1, "A": 10**-6.2}
}

# 使用 st.data_editor 让用户在网页上直接像 Excel 一样修改参数
param_df = pd.DataFrame(default_params).T
edited_param_df = st.data_editor(param_df, use_container_width=True)
creep_params = edited_param_df.T.to_dict()

col1, col2 = st.columns(2)
with col1:
    strain_rate = st.number_input("应变率 (Strain Rate)", value=1e-12, format="%e")
with col2:
    process_type = st.selectbox("变质过程 (Process Type)", ["prograde", "retrograde"])

R = 8.314

if st.button("🚀 开始计算强度", type="primary"):
    if not uploaded_files or len(uploaded_files) < 6:
        st.warning("⚠️ 请确保您已在左侧栏上传了全部 6 个必要的 CSV 文件！")
    else:
        with st.spinner('正在计算动力学流变强度...'):
            data = read_data_from_uploads(uploaded_files)
            
            # 基础信息打印
            solidus_T = next((T for T, vol in data.get("liquid", {}).items() if vol > 0), float('inf'))
            extraction_T = next((T for T, vol in data.get("liquid_ex", {}).items() if vol > 0), float('inf'))
            
            col_info1, col_info2 = st.columns(2)
            col_info1.info(f"**固液共存温度**: {solidus_T:.1f} K")
            col_info2.info(f"**液体第一次排出温度**: {extraction_T:.1f} K")
            
            all_temps = sorted(set(
                data["biotite"].index.tolist() + data["amphibole"].index.tolist() + 
                data["K-feldspar"].index.tolist() + data["plagioclase"].index.tolist() + 
                data["quartz"].index.tolist() + data["liquid"].index.tolist() + 
                data["liquid_ex"].index.tolist()
            ))
            
            T_vals, sigma_overall_vals, sigma_dark_vals, sigma_light_vals = [], [], [], []
            
            for T in all_temps:
                vol_b = get_scalar(data["biotite"], T)
                vol_a = get_scalar(data["amphibole"], T) 
                vol_k = get_scalar(data["K-feldspar"], T)
                vol_q = get_scalar(data["quartz"], T)
                vol_p = get_scalar(data["plagioclase"], T)
                vol_liq = get_scalar(data["liquid"], T) if T >= solidus_T else 0.0
                
                sigma_i = compute_sigma_i(T, creep_params, R, strain_rate)
                
                if T < solidus_T:
                    sigma_overall = compute_whole_rock_strength(sigma_i, vol_b, vol_a, vol_k, vol_p, vol_q) 
                    sigma_dark = float('nan')
                    sigma_light = float('nan')
                elif T < extraction_T:
                    sigma_whole = compute_whole_rock_strength(sigma_i, vol_b, vol_a, vol_k, vol_p, vol_q) 
                    sigma_overall = compute_dark_strength_2(sigma_whole, vol_liq, strain_rate, R, T)
                    sigma_dark = float('nan')
                    sigma_light = float('nan')
                else:
                    vol_light, vol_dark = get_volume_partition(process_type, vol_b, vol_a, vol_k, vol_q, vol_p, vol_liq)
                    sigma_whole = compute_whole_rock_strength(sigma_i, vol_b, vol_a, vol_k, vol_p, vol_q) 
                    sigma_dark = compute_whole_rock_strength(sigma_i, vol_b, vol_a, vol_k, vol_p, vol_q)
                    sigma_light = compute_light_strength(T, sigma_i, vol_liq, vol_k, vol_a, strain_rate, R)
                    sigma_overall = compute_overall_strength(sigma_dark, sigma_light, vol_dark, vol_light, vol_liq)
                
                T_vals.append(T)
                sigma_overall_vals.append(sigma_overall)
                sigma_dark_vals.append(sigma_dark)
                sigma_light_vals.append(sigma_light)
            
            # 整理结果
            df_results = pd.DataFrame({
                "温度 (K)": T_vals,
                "全岩强度 (MPa)": sigma_overall_vals,
                "暗色体强度 (MPa)": sigma_dark_vals,
                "浅色体强度 (MPa)": sigma_light_vals
            })
            
            st.success("✅ 计算完成！")

            # ----------------- 结果展示 -----------------
            st.header("📊 3. 计算结果")
            st.markdown("您可以直接在此图表上进行缩放、平移等交互操作。")
            
            # 使用 Streamlit 自带的交互式图表，完美避开云端中文字体丢失的问题
            chart_data = df_results.set_index("温度 (K)")
            st.line_chart(chart_data)
            
            st.markdown("### 数据表格")
            st.dataframe(df_results, use_container_width=True)
            
            # 一键下载 CSV
            csv = df_results.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 下载计算结果 (CSV)",
                data=csv,
                file_name="strength_results.csv",
                mime="text/csv",
            )