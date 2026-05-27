import math
import numpy as np
import pandas as pd
import streamlit as st
import io
import matplotlib.pyplot as plt
import matplotlib

# =================== 页面及后台绘图字体配置 ===================
st.set_page_config(page_title="混合岩石流变强度计算器", layout="wide")

# 设置 Matplotlib 后台生成的图片字体，避免中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

st.title("🪨 混合岩石流变强度计算器")
st.markdown("通过上传各矿物相的体积随温度变化的 CSV 文件，计算部分熔融混合岩的流变强度。")

# ----------------- 1. 数据读取与预处理 (加入缓存机制优化性能) -----------------
@st.cache_data(show_spinner=False)
def process_uploaded_files(uploaded_file_tuples):
    """
    处理上传的文件并提取数据。使用元组作为参数以便 Streamlit 进行哈希缓存，
    这样在用户仅仅修改流变参数时，不需要重新读取和解析 CSV，大幅提升流畅度。
    """
    data = {}
    file_mapping = {
        "b.csv": "biotite", "a.csv": "amphibole", "k.csv": "K-feldspar",
        "q.csv": "quartz", "p.csv": "plagioclase", "l.csv": "liquid"
    }
    
    for fname, file_bytes in uploaded_file_tuples:
        if fname not in file_mapping:
            continue
            
        phase = file_mapping[fname]
        df = pd.read_csv(io.BytesIO(file_bytes))
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
        return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
    except KeyError:
        return float(np.interp(T, series.index.astype(float), series.values.astype(float)))

# ----------------- 3. 计算各矿物强度 (优化：接收 log_A 并自动转换) -----------------
def compute_sigma_i(T, creep_params, R, strain_rate):
    sigma_i = {}
    for phase, params in creep_params.items():
        Q = params["Q"] * 1000.0  
        n = params["n"]
        # [优化] 将用户输入的对数直接还原为 A 值计算
        A = 10 ** params["log_A"]
        sigma_i[phase] = ((strain_rate / A) * math.exp(Q / (R * T))) ** (1.0 / n)
    return sigma_i

# ----------------- 4. 固相全岩强度计算（未产生熔体阶段） -----------------
def compute_whole_rock_strength(sigma_i, vol_b, vol_a, vol_k, vol_p, vol_q):
    J_s = -0.25
    total_sol_vol = vol_b + vol_a + vol_k + vol_q + vol_p 
    sum_term = 0.0
    
    phases_vols = {
        "biotite": vol_b, "amphibole": vol_a, "K-feldspar": vol_k, 
        "quartz": vol_q, "plagioclase": vol_p
    }
    
    for ph, vol in phases_vols.items():
        if ph in sigma_i:
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
st.sidebar.markdown("请上传全部 6 个 CSV 文件 (`b.csv`, `a.csv`, `k.csv`, `q.csv`, `p.csv`, `l.csv`)")
uploaded_files = st.sidebar.file_uploader("选择 CSV 文件", accept_multiple_files=True, type=['csv'])

st.header("⚙️ 2. 流变参数 (Creep Parameters)")
st.markdown("您可以直接在下方的表格中双击修改参数。**注意：为方便输入，$A$ 值已转换为 $\\log_{10}(A)$**（例如原 $A=10^{-29.9}$，只需输入 `-29.9`）。")

# 默认流变参数（已将 A 转换为 log_A）
default_params = {
    "biotite":     {"Q": 51.0, "n": 18.0, "log_A": -29.9},
    "quartz":      {"Q": 163.0, "n": 2.4, "log_A": -5.0},
    "K-feldspar":  {"Q": 356.0, "n": 3.0, "log_A": 2.6},
    "plagioclase": {"Q": 310.0, "n": 3.0, "log_A": 2.602}, 
    "amphibole":   {"Q": 154.0, "n": 3.1, "log_A": -6.2}
}

param_df = pd.DataFrame(default_params).T
# 使用 st.data_editor 呈现可编辑表格
edited_param_df = st.data_editor(
    param_df, 
    use_container_width=True,
    column_config={
        "Q": st.column_config.NumberColumn("Q (kJ/mol)", format="%.1f"),
        "n": st.column_config.NumberColumn("n (应力指数)", format="%.2f"),
        "log_A": st.column_config.NumberColumn("log₁₀(A)", help="指前因子 A 的 10 的指数")
    }
)
creep_params = edited_param_df.T.to_dict()

col1, col2 = st.columns(2)
with col1:
    strain_rate = st.number_input("应变率 (Strain Rate)", value=1e-12, format="%e")
with col2:
    process_type = st.selectbox("变质过程 (Process Type)", ["prograde", "retrograde"])

R = 8.314

if st.button("🚀 开始计算图表与强度", type="primary"):
    if not uploaded_files or len(uploaded_files) < 6:
        st.warning("⚠️ 请确保您已在左侧栏上传了全部 6 个必要的 CSV 文件！")
    else:
        with st.spinner('正在执行动力学流变强度计算...'):
            # 提取文件内容用于哈希缓存，提升运算流畅度
            file_tuples = tuple((f.name, f.getvalue()) for f in uploaded_files)
            data = process_uploaded_files(file_tuples)
            
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
                
                # 获取固态全岩强度，三种情况都会用到
                sigma_whole = compute_whole_rock_strength(sigma_i, vol_b, vol_a, vol_k, vol_p, vol_q)
                
                if T < solidus_T:
                    sigma_overall = sigma_whole
                    sigma_dark = float('nan')
                    sigma_light = float('nan')
                elif T < extraction_T:
                    sigma_overall = compute_dark_strength_2(sigma_whole, vol_liq, strain_rate, R, T)
                    sigma_dark = float('nan')
                    sigma_light = float('nan')
                else:
                    vol_light, vol_dark = get_volume_partition(process_type, vol_b, vol_a, vol_k, vol_q, vol_p, vol_liq)
                    sigma_dark = sigma_whole
                    sigma_light = compute_light_strength(T, sigma_i, vol_liq, vol_k, vol_a, strain_rate, R)
                    sigma_overall = compute_overall_strength(sigma_dark, sigma_light, vol_dark, vol_light, vol_liq)
                
                T_vals.append(T)
                sigma_overall_vals.append(sigma_overall)
                sigma_dark_vals.append(sigma_dark)
                sigma_light_vals.append(sigma_light)
            
            df_results = pd.DataFrame({
                "温度 (K)": T_vals,
                "全岩强度 (MPa)": sigma_overall_vals,
                "暗色体强度 (MPa)": sigma_dark_vals,
                "浅色体强度 (MPa)": sigma_light_vals
            })

            # ----------------- 结果展示区 -----------------
            st.header("📊 3. 计算结果")
            
            # 网页端交互式图表
            st.markdown("您可以直接在此图表上进行缩放、平移等交互操作。")
            chart_data = df_results.set_index("温度 (K)")
            st.line_chart(chart_data)
            
            # 导出区
            st.markdown("### 💾 导出数据与图表")
            col_down1, col_down2 = st.columns(2)
            
            # CSV 导出按钮
            with col_down1:
                csv = df_results.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 导出为 CSV 表格",
                    data=csv,
                    file_name="strength_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # 图片导出逻辑 (使用后台 Matplotlib)
            with col_down2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(T_vals, sigma_overall_vals, label="全岩强度", linewidth=2)
                ax.plot(T_vals, sigma_dark_vals, label="暗色体强度", linewidth=2)
                ax.plot(T_vals, sigma_light_vals, label="浅色体强度", linewidth=2)
                ax.set_xlabel("温度 (K)")
                ax.set_ylabel("强度 (MPa)")
                ax.set_title(f"温度变化下的强度曲线 (应变率: {strain_rate:.1e})")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", dpi=300)
                img_buf.seek(0)
                
                st.download_button(
                    label="🖼️ 导出为 PNG 图片",
                    data=img_buf,
                    file_name="strength_plot.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            plt.close(fig)
            
            # 数据表格展示
            st.markdown("### 数据详情")
            st.dataframe(df_results, use_container_width=True)
