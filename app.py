"""
Materials ML Pipeline — Streamlit 웹앱
실행: streamlit run app.py
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import os

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="Materials ML Pipeline",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── 커스텀 CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif !important;
}
.main-header {
    background: linear-gradient(135deg, #0f1f45, #1d4ed8);
    color: white; padding: 32px 40px; border-radius: 14px; margin-bottom: 24px;
}
.main-header h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 6px; }
.main-header p  { font-size: 0.95rem; opacity: 0.85; margin: 0; }
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 20px 24px; text-align: center;
}
.metric-card .val { font-size: 1.8rem; font-weight: 700; }
.metric-card .lbl { font-size: 0.78rem; color: #64748b; margin-top: 4px; }
.insight {
    border-radius: 10px; padding: 14px 18px;
    font-size: 0.88rem; line-height: 1.75; margin: 12px 0;
}
.ins-blue   { background: #eff6ff; border-left: 4px solid #2563eb; color: #1e3a8a; }
.ins-orange { background: #fffbeb; border-left: 4px solid #d97706; color: #78350f; }
.ins-green  { background: #f0fdf4; border-left: 4px solid #059669; color: #064e3b; }
.ins-red    { background: #fef2f2; border-left: 4px solid #dc2626; color: #7f1d1d; }
</style>
""", unsafe_allow_html=True)

# ── 경로 설정 ────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

# ── 모델 로드 (캐시) ─────────────────────────────────────
@st.cache_resource
def load_bandgap_model():
    m = xgb.XGBRegressor()
    m.load_model(os.path.join(DATA, "xgb_bandgap_model.json"))
    return m

@st.cache_resource
def load_dielectric_model():
    m = xgb.XGBRegressor()
    m.load_model(os.path.join(DATA, "xgb_dielectric_model.json"))
    return m

@st.cache_resource
def load_process_model():
    m = xgb.XGBRegressor()
    m.load_model(os.path.join(DATA, "xgb_process_model.json"))
    return m

@st.cache_data
def load_dual_predictions():
    return pd.read_csv(os.path.join(DATA, "dual_predictions.csv"))

@st.cache_data
def load_featurized():
    return pd.read_csv(os.path.join(DATA, "featurized_highk.csv"))

# ── 모델 로드 시도 ───────────────────────────────────────
models_ok = True
try:
    model_eg  = load_bandgap_model()
    model_k   = load_dielectric_model()
    model_gpc = load_process_model()
    dual      = load_dual_predictions()
    feat_df   = load_featurized()
    feature_cols = [c for c in feat_df.columns
                    if c not in ['material_id', 'formula', 'band_gap']]
except Exception as e:
    models_ok = False
    model_error = str(e)

# ── 헤더 ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚗️ Materials ML Pipeline</h1>
  <p>AI 기반 반도체 소재 스크리닝 · 유전율 예측 · ALD 공정 최적화</p>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error(f"⚠️ 모델 파일을 불러올 수 없습니다. 먼저 노트북 04, 06, 10을 실행해 주세요.\n\n오류: {model_error}")
    st.stop()

# ── 탭 ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 소재 검색 · 예측",
    "📊 Pareto 스크리닝",
    "⚙️ ALD 공정 예측",
    "📖 프로젝트 설명"
])

# ════════════════════════════════════════════════════════
# TAB 1 — 소재 검색 + 예측
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("소재 검색 및 물성 예측")
    st.caption("화학식을 입력하거나, 아래 목록에서 선택하세요.")

    col_input, col_result = st.columns([1, 1.4], gap="large")

    with col_input:
        # 검색 방식 선택
        search_mode = st.radio(
            "입력 방식",
            ["목록에서 선택", "화학식 직접 입력"],
            horizontal=True
        )

        if search_mode == "목록에서 선택":
            # Top 후보 목록
            top_candidates = dual.nlargest(50, 'predicted_k')['formula'].tolist()
            ref = ['HfO2', 'ZrO2', 'Al2O3', 'TiO2', 'SiO2']
            options = ref + [f for f in top_candidates if f not in ref]
            selected = st.selectbox("소재 선택", options)
            formula_input = selected
        else:
            formula_input = st.text_input(
                "화학식 입력 (예: HfO2, ZrO2, Al2O3, BaZrO3)",
                value="HfO2",
                placeholder="화학식을 입력하세요"
            )

        st.divider()
        predict_btn = st.button("🔍 예측 실행", type="primary", use_container_width=True)

    with col_result:
        if predict_btn or True:   # 기본으로 HfO2 표시
            formula_clean = formula_input.strip()

            # 데이터셋에서 찾기
            row = dual[dual['formula'] == formula_clean]

            if len(row) == 0:
                # matminer로 실시간 예측 시도
                st.markdown('<div class="insight ins-orange">⚠️ 데이터셋에 없는 소재입니다. matminer 피처화를 시도합니다.</div>', unsafe_allow_html=True)
                try:
                    from matminer.featurizers.composition import ElementProperty
                    from pymatgen.core import Composition

                    with st.spinner("피처 계산 중..."):
                        ep = ElementProperty.from_preset("magpie")
                        comp = Composition(formula_clean)
                        feat_row = ep.featurize(comp)
                        feat_names = ep.feature_labels()
                        X_new = pd.DataFrame([feat_row], columns=feat_names)

                        # feature 순서 맞추기
                        model_features = model_eg.get_booster().feature_names
                        X_aligned = X_new.reindex(columns=model_features, fill_value=0)

                        pred_eg = float(model_eg.predict(X_aligned)[0])
                        pred_k  = float(model_k.predict(X_aligned)[0])

                except ImportError:
                    st.error("matminer가 설치되어 있지 않습니다. 데이터셋에 있는 소재만 검색 가능합니다.")
                    st.stop()
                except Exception as e:
                    st.error(f"예측 실패: {e}")
                    st.stop()
            else:
                pred_eg = float(row['predicted_Eg'].values[0])
                pred_k  = float(row['predicted_k'].values[0])

            # ── 결과 표시 ──
            st.markdown(f"### {formula_clean} 예측 결과")

            m1, m2, m3 = st.columns(3)
            with m1:
                color = "#2563eb" if pred_eg > 3.0 else "#dc2626"
                st.markdown(f"""<div class="metric-card">
                    <div class="val" style="color:{color}">{pred_eg:.2f} eV</div>
                    <div class="lbl">예측 밴드갭 (Eg)</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                color = "#059669" if pred_k > 20 else "#d97706"
                st.markdown(f"""<div class="metric-card">
                    <div class="val" style="color:{color}">{pred_k:.1f}</div>
                    <div class="lbl">예측 유전율 (k)</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                dram_ok = pred_eg > 3.0 and pred_k > 20
                stars = "⭐⭐⭐" if dram_ok else ("⭐⭐" if pred_eg > 3.0 or pred_k > 20 else "⭐")
                color = "#059669" if dram_ok else "#d97706"
                st.markdown(f"""<div class="metric-card">
                    <div class="val" style="color:{color}">{stars}</div>
                    <div class="lbl">DRAM 적합도</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 판정
            issues = []
            if pred_eg < 3.0:
                issues.append(f"밴드갭 {pred_eg:.2f} eV < 3.0 eV (누설전류 위험)")
            if pred_k < 20:
                issues.append(f"유전율 {pred_k:.1f} < 20 (캐패시턴스 부족)")

            if not issues:
                st.markdown('<div class="insight ins-green">✅ <strong>DRAM high-k 후보 조건 충족!</strong><br>k > 20 이고 Eg > 3.0 eV를 동시에 만족합니다. Pareto 스크리닝 탭에서 위치를 확인하세요.</div>', unsafe_allow_html=True)
            else:
                msg = "<br>".join(f"• {i}" for i in issues)
                st.markdown(f'<div class="insight ins-orange">⚠️ <strong>DRAM 조건 미충족:</strong><br>{msg}</div>', unsafe_allow_html=True)

            # 참조값 비교
            ref_data = {
                'HfO2': (4.02, 25.0), 'ZrO2': (3.53, 25.0),
                'Al2O3': (5.87, 9.0), 'TiO2': (2.06, 80.0)
            }
            st.markdown("**레퍼런스 소재와 비교**")
            ref_rows = []
            for ref_f, (ref_eg, ref_k) in ref_data.items():
                ref_rows.append({'소재': ref_f, 'Eg 실험값': f'{ref_eg} eV', 'k 실험값': str(ref_k),
                                 '현재 소재 Eg': f'{pred_eg:.2f} eV', '현재 소재 k': f'{pred_k:.1f}'})
            cmp_df = pd.DataFrame([
                {'구분': '레퍼런스 HfO₂', 'Eg (eV)': 4.02, 'k': 25.0},
                {'구분': '레퍼런스 ZrO₂', 'Eg (eV)': 3.53, 'k': 25.0},
                {'구분': '레퍼런스 Al₂O₃', 'Eg (eV)': 5.87, 'k': 9.0},
                {'구분': f'▶ {formula_clean} (예측)', 'Eg (eV)': round(pred_eg, 2), 'k': round(pred_k, 1)},
            ])
            st.dataframe(
                cmp_df.style.apply(
                    lambda x: ['background: #dbeafe; font-weight:bold' if '▶' in str(v) else '' for v in x],
                    axis=1
                ),
                use_container_width=True, hide_index=True
            )

            # Radar / Bar chart
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=['밴드갭 Eg (eV)', '유전율 k'],
                y=[pred_eg, pred_k],
                marker_color=['#2563eb', '#059669'],
                text=[f'{pred_eg:.2f}', f'{pred_k:.1f}'],
                textposition='outside'
            ))
            fig_bar.add_hline(y=3.0, line_dash="dash", line_color="#dc2626",
                              annotation_text="Eg 최소 기준 3.0 eV", annotation_position="top right")
            fig_bar.add_hline(y=20, line_dash="dash", line_color="#d97706",
                              annotation_text="k 최소 기준 20", annotation_position="bottom right")
            fig_bar.update_layout(
                title=f"{formula_clean} 예측 물성",
                height=320, margin=dict(t=50, b=20),
                yaxis_title="값",
                showlegend=False,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 2 — Pareto 스크리닝
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("DRAM High-k 후보 Pareto 스크리닝")
    st.caption("슬라이더로 조건을 조정하면 후보 소재가 실시간으로 필터링됩니다.")

    # Oxide only filter
    try:
        from pymatgen.core import Composition

        def is_oxide_only(formula):
            try:
                comp = Composition(formula)
                excl = {'H', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Se', 'Te'}
                return len({str(e) for e in comp.elements} & excl) == 0
            except:
                return False

        use_oxide = st.checkbox("산화물만 표시 (ALD 공정 적합 소재)", value=True)
        df_screen = dual[dual['formula'].apply(is_oxide_only)] if use_oxide else dual.copy()
    except ImportError:
        df_screen = dual.copy()
        st.info("pymatgen 미설치 — 산화물 필터 비활성화")
        use_oxide = False

    col_ctrl, col_chart = st.columns([1, 2.2], gap="large")

    with col_ctrl:
        st.markdown("**필터 조건**")
        k_min = st.slider("유전율 k 최솟값", 0, 60, 20, step=1)
        eg_min = st.slider("밴드갭 Eg 최솟값 (eV)", 0.0, 6.0, 3.0, step=0.1)

        st.divider()
        filtered = df_screen[
            (df_screen['predicted_k'] > k_min) &
            (df_screen['predicted_Eg'] > eg_min)
        ]
        total = len(df_screen)
        n_filt = len(filtered)

        st.metric("전체 후보", f"{total:,}개")
        st.metric("필터 결과", f"{n_filt}개",
                  delta=f"{n_filt/total*100:.1f}%")

        if n_filt > 0:
            st.markdown(f"""
            <div class="insight ins-blue" style="margin-top:12px;">
            <strong>상위 3개</strong><br>
            {"<br>".join([
                f"• {row['formula']} (k={row['predicted_k']:.1f}, Eg={row['predicted_Eg']:.2f})"
                for _, row in filtered.nlargest(3, 'predicted_k').iterrows()
            ])}
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        # 산포도
        df_screen['구분'] = '기타 후보'
        df_screen.loc[
            (df_screen['predicted_k'] > k_min) &
            (df_screen['predicted_Eg'] > eg_min), '구분'
        ] = '조건 충족'

        fig = px.scatter(
            df_screen,
            x='predicted_k',
            y='predicted_Eg',
            color='구분',
            color_discrete_map={'기타 후보': '#cbd5e1', '조건 충족': '#d97706'},
            hover_data=['formula'],
            labels={'predicted_k': '예측 유전율 k', 'predicted_Eg': '예측 밴드갭 Eg (eV)'},
            opacity=0.7,
            size_max=8
        )

        # 레퍼런스 추가
        refs = [
            ('HfO₂', 25.0, 4.02), ('ZrO₂', 25.0, 3.53),
            ('Al₂O₃', 9.0, 5.87), ('TiO₂', 80.0, 2.06)
        ]
        for name, k_r, eg_r in refs:
            fig.add_trace(go.Scatter(
                x=[k_r], y=[eg_r], mode='markers+text',
                marker=dict(size=12, color='#dc2626', symbol='star'),
                text=[name], textposition='top right',
                textfont=dict(color='#dc2626', size=11),
                name=name, showlegend=False
            ))

        # 임계선
        fig.add_vline(x=k_min, line_dash="dash", line_color="#059669",
                      annotation_text=f"k={k_min}", annotation_position="top")
        fig.add_hline(y=eg_min, line_dash="dash", line_color="#7c3aed",
                      annotation_text=f"Eg={eg_min}", annotation_position="right")

        fig.update_layout(
            height=440, margin=dict(t=30, b=20),
            plot_bgcolor='white',
            xaxis=dict(gridcolor='#f1f5f9', title_font_size=13),
            yaxis=dict(gridcolor='#f1f5f9', title_font_size=13),
            legend=dict(orientation='h', y=-0.12)
        )
        st.plotly_chart(fig, use_container_width=True)

    # 후보 테이블
    if n_filt > 0:
        st.markdown(f"**조건 충족 후보 상위 20개** (총 {n_filt}개 중)")
        show_cols = ['formula', 'predicted_k', 'predicted_Eg']
        top20 = filtered.nlargest(20, 'predicted_k')[show_cols].copy()
        top20.columns = ['화학식', '예측 k', '예측 Eg (eV)']
        top20['예측 k']       = top20['예측 k'].round(1)
        top20['예측 Eg (eV)'] = top20['예측 Eg (eV)'].round(3)
        top20 = top20.reset_index(drop=True)
        top20.index += 1
        st.dataframe(top20, use_container_width=True)

        csv = top20.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 CSV 다운로드", csv, "candidates.csv", "text/csv")
    else:
        st.markdown('<div class="insight ins-red">조건을 만족하는 소재가 없습니다. 슬라이더를 낮춰보세요.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 3 — ALD 공정 예측
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("ALD 공정 파라미터 → GPC 예측")
    st.caption("공정 조건을 입력하면 성장률(GPC, Å/cycle)을 예측합니다.")

    # LabelEncoder (학습 시와 동일하게 하드코딩)
    material_map  = {'Al2O3': 0, 'HfO2': 1, 'ZrO2': 2}
    precursor_map = {'HfCl4': 0, 'TDMAH': 1, 'TDMAZ': 2, 'TEMAHf': 3, 'TMA': 4, 'ZrCl4': 5}
    oxidant_map   = {'H2O': 0, 'O3': 1}

    precursor_by_mat = {
        'HfO2':  ['TDMAH', 'HfCl4', 'TEMAHf'],
        'Al2O3': ['TMA'],
        'ZrO2':  ['ZrCl4', 'TDMAZ'],
    }

    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        st.markdown("**공정 조건 입력**")

        mat = st.selectbox("소재", ['HfO2', 'Al2O3', 'ZrO2'])
        prec = st.selectbox("전구체 (Precursor)", precursor_by_mat[mat])
        oxid = st.selectbox("산화제 (Oxidant)", ['H2O', 'O3'])

        # ALD 윈도우 힌트
        window_hint = {
            'HfO2': (150, 300), 'Al2O3': (100, 280), 'ZrO2': (150, 280)
        }
        lo, hi = window_hint[mat]
        st.caption(f"💡 {mat} ALD 윈도우: {lo}~{hi} °C")

        temp = st.slider("온도 (°C)", 50, 400, 250, step=10)
        pulse = st.slider("Pulse 시간 (초)", 0.05, 2.0, 0.3, step=0.05,
                          format="%.2f")
        purge = st.slider("Purge 시간 (초)", 5, 30, 10, step=5)

        predict_ald = st.button("⚙️ GPC 예측", type="primary", use_container_width=True)

    with col_right:
        if predict_ald or True:
            # 예측
            X_proc = pd.DataFrame([[
                material_map[mat],
                precursor_map.get(prec, 0),
                oxidant_map[oxid],
                temp, pulse, purge
            ]], columns=['material_enc', 'precursor_enc', 'oxidant_enc',
                         'temperature', 'pulse_time', 'purge_time'])

            gpc_pred = float(model_gpc.predict(X_proc)[0])
            gpc_pred = max(0.1, gpc_pred)   # 음수 방지

            # ALD 윈도우 판단
            in_window = lo <= temp <= hi
            pulse_ok  = pulse >= 0.1

            st.markdown(f"### {mat} GPC 예측 결과")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="val" style="color:#059669">{gpc_pred:.3f}</div>
                    <div class="lbl">예측 GPC (Å/cycle)</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                w_color = "#059669" if in_window else "#dc2626"
                w_text  = "✅ 윈도우 내" if in_window else "❌ 윈도우 외"
                st.markdown(f"""<div class="metric-card">
                    <div class="val" style="color:{w_color}; font-size:1.1rem">{w_text}</div>
                    <div class="lbl">ALD 온도 윈도우</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                p_color = "#059669" if pulse_ok else "#d97706"
                p_text  = "✅ 포화" if pulse_ok else "⚠️ 미포화"
                st.markdown(f"""<div class="metric-card">
                    <div class="val" style="color:{p_color}; font-size:1.1rem">{p_text}</div>
                    <div class="lbl">Pulse 포화 여부</div>
                </div>""", unsafe_allow_html=True)

            # 경고/안내
            msgs = []
            if not in_window:
                msgs.append(f"온도 {temp}°C가 ALD 윈도우({lo}~{hi}°C) 밖입니다.")
            if not pulse_ok:
                msgs.append(f"Pulse {pulse:.2f}s는 포화 흡착에 부족할 수 있습니다 (권장 ≥ 0.1s).")
            if msgs:
                st.markdown(f'<div class="insight ins-orange">⚠️ {"<br>".join(msgs)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight ins-green">✅ 공정 조건이 ALD 윈도우 내에 있고 pulse가 충분합니다.</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 온도 스캔 시각화
            temps_scan = np.arange(50, 420, 10)
            gpc_scan = []
            for t in temps_scan:
                X_scan = pd.DataFrame([[
                    material_map[mat], precursor_map.get(prec, 0),
                    oxidant_map[oxid], t, pulse, purge
                ]], columns=X_proc.columns)
                gpc_scan.append(float(model_gpc.predict(X_scan)[0]))

            fig_temp = go.Figure()

            # ALD window 배경
            fig_temp.add_vrect(x0=lo, x1=hi, fillcolor="#d1fae5", opacity=0.3,
                               line_width=0, annotation_text="ALD Window",
                               annotation_position="top left",
                               annotation_font_color="#059669")

            fig_temp.add_trace(go.Scatter(
                x=temps_scan, y=gpc_scan,
                mode='lines', name='예측 GPC',
                line=dict(color='#059669', width=2.5)
            ))
            fig_temp.add_vline(
                x=temp, line_dash="dash", line_color="#2563eb",
                annotation_text=f"현재 {temp}°C", annotation_position="top"
            )
            fig_temp.update_layout(
                title=f"{mat} 온도별 GPC 예측 (pulse={pulse:.2f}s, purge={purge}s)",
                xaxis_title="온도 (°C)",
                yaxis_title="GPC (Å/cycle)",
                height=340, margin=dict(t=50, b=20),
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#f1f5f9'),
                yaxis=dict(gridcolor='#f1f5f9')
            )
            st.plotly_chart(fig_temp, use_container_width=True)

            # pulse 스캔
            pulses_scan = np.arange(0.02, 2.1, 0.05)
            gpc_pulse = []
            for p in pulses_scan:
                X_p = pd.DataFrame([[
                    material_map[mat], precursor_map.get(prec, 0),
                    oxidant_map[oxid], temp, p, purge
                ]], columns=X_proc.columns)
                gpc_pulse.append(float(model_gpc.predict(X_p)[0]))

            fig_pulse = go.Figure()
            fig_pulse.add_vrect(x0=0.1, x1=2.1, fillcolor="#dbeafe", opacity=0.2,
                                line_width=0, annotation_text="포화 구간",
                                annotation_position="top left",
                                annotation_font_color="#2563eb")
            fig_pulse.add_trace(go.Scatter(
                x=pulses_scan, y=gpc_pulse,
                mode='lines', name='예측 GPC',
                line=dict(color='#2563eb', width=2.5)
            ))
            fig_pulse.add_vline(
                x=pulse, line_dash="dash", line_color="#d97706",
                annotation_text=f"현재 {pulse:.2f}s", annotation_position="top"
            )
            fig_pulse.update_layout(
                title=f"{mat} Pulse 시간별 GPC (온도={temp}°C)",
                xaxis_title="Pulse 시간 (s)",
                yaxis_title="GPC (Å/cycle)",
                height=300, margin=dict(t=50, b=20),
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#f1f5f9'),
                yaxis=dict(gridcolor='#f1f5f9')
            )
            st.plotly_chart(fig_pulse, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 4 — 프로젝트 설명
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("프로젝트 개요")

    st.markdown("""
    ### 3단계 파이프라인

    | 단계 | 목표 | 모델 | 성능 |
    |------|------|------|------|
    | Phase 1 | 밴드갭 (Eg) 예측 | XGBoost | R²=0.749, MAE=0.287 eV |
    | Phase 2 | 유전율 (k) 예측 + Pareto 스크리닝 | XGBoost | R²=0.714 |
    | Phase 3 | ALD 공정 GPC 예측 | XGBoost | R²=0.918 (더미 데이터) |

    ### 핵심 인사이트: 같은 파이프라인, 세 가지 다른 물리
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **Phase 1 · 밴드갭**
        - 1위 Feature: `avg_dev NdValence`
        - d⁰ 혼합 정도 → 전하이동경로
        - **결정장 이론**으로 설명
        """)
    with c2:
        st.markdown("""
        **Phase 2 · 유전율**
        - 1위 Feature: `avg_dev NdUnfilled`
        - 빈 d 오비탈 → 이온 분극
        - **Clausius-Mossotti**로 설명
        """)
    with c3:
        st.markdown("""
        **Phase 3 · ALD GPC**
        - 1위 Feature: `pulse_time`
        - 포화 흡착 완성 → GPC 안정
        - **자기제한 반응**으로 설명
        """)

    st.markdown("""
    ---
    ### 데이터 출처
    - **Materials Project API** — 153,000개 무기 화합물 DFT 계산값
    - **matminer Magpie** — 132개 조성 기반 특성 자동 생성
    - **ALD 공정 데이터** — 현재 더미 데이터 (실제 문헌 데이터 교체 가능)

    ### 스크리닝 조건 (DRAM 기준)
    - 유전율 **k > 20** (HfO₂ ≈ 25 기준)
    - 밴드갭 **Eg > 3.0 eV** (누설전류 억제)
    - 산화물만 (ALD/CVD 공정 가능)
    """)

    st.info("📁 프로젝트 폴더: 노트북 01~11 순서대로 실행하면 모델이 생성됩니다.")
