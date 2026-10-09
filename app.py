import streamlit as st
import pandas as pd
import requests
import os
import plotly.express as px
import plotly.graph_objects as go
import datetime

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="HCW Surveillance Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------------
# 2. SECURE CREDENTIALS LOADING
# Checks st.secrets first, then falls back to os.environ
# ---------------------------------------------------------
def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key)

ODK_URL = get_secret("ODK_URL")
ODK_USERNAME = get_secret("ODK_USERNAME")
ODK_PASSWORD = get_secret("ODK_PASSWORD")
PROJECT_ID = str(get_secret("PROJECT_ID")) if get_secret("PROJECT_ID") is not None else None

FORM_ID_SURVEILLANCE = get_secret("FORM_ID_SURVEILLANCE")
FORM_ID_SAMPLING = get_secret("FORM_ID_SAMPLING")
FORM_ID_REPORT = get_secret("FORM_ID_REPORT")
FORM_ID_OUTCOME = get_secret("FORM_ID_OUTCOME")
FORM_ID_EXIT = get_secret("FORM_ID_EXIT")
ENTITY_ID_ILI = get_secret("ENTITY_ID_ILI")
ENTITY_ID_ELIGIBLE = get_secret("ENTITY_ID_ELIGIBLE")
ENTITY_ID_PARTICIPANTS = get_secret("ENTITY_ID_PARTICIPANTS") or "participant_ids"

# ---------------------------------------------------------
# 3. GLOBAL OPTIMIZED DATA HELPERS
# ---------------------------------------------------------
def get_nested_col(df, key):
    if key in df.columns: 
        return df[key]
    for col in df.columns:
        valid_series = df[col].dropna()
        if not valid_series.empty:
            first_val = valid_series.iloc[0]
            if isinstance(first_val, dict):
                has_key = any(key in d for d in valid_series.head(100) if isinstance(d, dict))
                if has_key:
                    return df[col].apply(lambda x: x.get(key) if isinstance(x, dict) else None)
    matches = [c for c in df.columns if key in str(c).lower()]
    if matches: 
        return df[matches[0]]
    return pd.Series([None] * len(df), index=df.index)

def get_any_col(df, key, group_name):
    clean_key = key.replace(f'{group_name}-', '')
    if group_name in df.columns:
        valid_series = df[group_name].dropna()
        if not valid_series.empty:
            first_val = valid_series.iloc[0]
            if isinstance(first_val, dict):
                has_key = any(clean_key in d for d in valid_series.head(100) if isinstance(d, dict))
                if has_key:
                    return df[group_name].apply(lambda x: x.get(clean_key) if isinstance(x, dict) else None)
    if key in df.columns: 
        return df[key]
    if clean_key in df.columns: 
        return df[clean_key]
    matches = [c for c in df.columns if clean_key in str(c).lower()]
    if matches: 
        return df[matches[0]]
    return pd.Series([None] * len(df), index=df.index)

def clean_phone_series(s):
    if s is None:
        return pd.Series(dtype=str)
    
    def clean_single(val):
        if pd.isna(val):
            return ""
        if isinstance(val, dict):
            val = str(val.get('phone_number') or val.get('phone') or val.get('number') or '')
        val_str = str(val)
        if val_str.endswith('.0'):
            val_str = val_str[:-2]
        digits = "".join([c for c in val_str if c.isdigit()])
        return digits[-10:] if len(digits) >= 10 else digits

    return s.map(clean_single)

def clean_episode_series(s):
    return s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True).replace({'nan': '', 'None': ''})

def clean_barcode_series(s):
    return s.astype(str).str.strip().str.upper().replace({'NAN': '', 'NONE': ''})

def format_age_sex_row(row, age_col, sex_col):
    age = str(row[age_col]).split('.')[0] if pd.notna(row[age_col]) else '?'
    if age in ('nan', 'None'): 
        age = '?'
    sex = str(row[sex_col]).strip() if pd.notna(row[sex_col]) else '?'
    if sex in ('nan', 'None'): 
        sex = '?'
    sex = sex[0].upper() if sex != '?' and len(sex) > 0 else '?'
    return f"{age} / {sex}"

CITY_MAP = {'NC': 'NCT Delhi', 'PU': 'Pune', 'CH': 'Chennai', 'JO': 'Jodhpur', 'KO': 'Kolkata', 'GU': 'Guwahati'}
ABBR_MAP = {v: k for k, v in CITY_MAP.items()}
for k in CITY_MAP.keys(): 
    ABBR_MAP[k] = k 

def get_phase_global(city, hosp, current_study_phase):
    if pd.isna(city): 
        return 'Unknown'
    abbr = ABBR_MAP.get(city, city)
    if abbr == 'NC':
        if current_study_phase == "Main Study":
            return 'NC'
        hosp_str = str(hosp).upper()
        if 'NC03' in hosp_str: return 'NC (Phase 1)'
        if 'NC01' in hosp_str: return 'NC (Phase 2)'
        return 'NC'
    return abbr

# ---------------------------------------------------------
# 4. DATA FETCHING FUNCTIONS (CACHED)
# ---------------------------------------------------------
@st.cache_data(ttl=43200, show_spinner=False)
def fetch_odk_data(endpoint_suffix):
    """Fetches data from ODK Central using the provided OData endpoint suffix."""
    if not all([ODK_URL, ODK_USERNAME, ODK_PASSWORD, PROJECT_ID]):
        return pd.DataFrame()
        
    endpoint = f"{ODK_URL.rstrip('/')}/v1/projects/{PROJECT_ID}/{endpoint_suffix}"
    try:
        response = requests.get(endpoint, auth=(ODK_USERNAME, ODK_PASSWORD))
        response.raise_for_status()
        data = response.json()
        if "value" in data:
            df = pd.DataFrame(data["value"])
            # Unpack nested entity data or currentVersion properties if present
            for nested_col in ["data", "currentVersion"]:
                if nested_col in df.columns:
                    try:
                        nested_df = pd.json_normalize(df[nested_col].dropna())
                        for col in nested_df.columns:
                            if col not in df.columns:
                                df[col] = nested_df[col]
                    except Exception:
                        pass
            return df
    except requests.exceptions.HTTPError as errh:
        st.error(f"HTTP Error for {endpoint_suffix}: {errh}")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    return pd.DataFrame()

def load_all_data():
    """Loads data from all form submissions and the entity list."""
    data = {}
    if FORM_ID_SURVEILLANCE: data['Surveillance'] = fetch_odk_data(f"forms/{FORM_ID_SURVEILLANCE}.svc/Submissions")
    if FORM_ID_SAMPLING: data['Sampling'] = fetch_odk_data(f"forms/{FORM_ID_SAMPLING}.svc/Submissions")
    if FORM_ID_REPORT: data['Report'] = fetch_odk_data(f"forms/{FORM_ID_REPORT}.svc/Submissions")
    if FORM_ID_OUTCOME: data['Outcome'] = fetch_odk_data(f"forms/{FORM_ID_OUTCOME}.svc/Submissions")
    if FORM_ID_EXIT: data['Exit'] = fetch_odk_data(f"forms/{FORM_ID_EXIT}.svc/Submissions")
    if ENTITY_ID_ILI: data['ILI Entities'] = fetch_odk_data(f"datasets/{ENTITY_ID_ILI}.svc/Entities")
    if ENTITY_ID_ELIGIBLE: data['Eligible Entities'] = fetch_odk_data(f"datasets/{ENTITY_ID_ELIGIBLE}.svc/Entities")
    if ENTITY_ID_PARTICIPANTS: data['Participant IDs'] = fetch_odk_data(f"datasets/{ENTITY_ID_PARTICIPANTS}.svc/Entities")
    return data

# ---------------------------------------------------------
# 5. DASHBOARD UI & TABS
# ---------------------------------------------------------
st.title("🏥 Health Care Worker (HCW) Surveillance")
st.markdown("Real-time monitoring across multiple forms and entity lists.")

with st.spinner("Connecting to ODK Central and downloading datasets..."):
    datasets = load_all_data()

if datasets:
    # Universal Sidebar Filters
    st.sidebar.header("🔄 Data Sync")
    st.sidebar.caption("Data is cached locally for 12 hours, click Manual Sync to update the dashboard")
    if st.sidebar.button("Manual Sync", type="primary", use_container_width=True):
        fetch_odk_data.clear()
        st.rerun()
        
    st.sidebar.divider()
    st.sidebar.header("🔍 Data Filters")
    
    study_phase = st.sidebar.radio("Study Phase", ["Pilot Study", "Main Study"], index=1)
    
    today = datetime.date.today()
    if study_phase == "Pilot Study":
        default_start = datetime.date(2026, 1, 1)
        default_end = datetime.date(2026, 6, 5)
    else:
        default_start = datetime.date(2026, 6, 15)
        default_end = today
    
    if 'last_study_phase' not in st.session_state:
        st.session_state.last_study_phase = study_phase
    if study_phase != st.session_state.last_study_phase:
        st.session_state.date_range_val = (default_start, default_end)
        st.session_state.last_study_phase = study_phase

    if 'date_range_val' not in st.session_state:
        st.session_state.date_range_val = (default_start, default_end)
        
    st.sidebar.markdown("**Quick Date Filters**")
    st.sidebar.caption("💡Tip: Use last 4 wks button to get optimized charts for last 4 weeks of surveillance")
    btn_col1, btn_col2, btn_col3 = st.sidebar.columns(3)
        
    if btn_col1.button("Last Wk", help="Monday to Sunday of last week", use_container_width=True):
        last_sunday = today - datetime.timedelta(days=today.weekday() + 1)
        last_monday = last_sunday - datetime.timedelta(days=6)
        st.session_state.date_range_val = (last_monday, last_sunday)
        st.session_state.date_mode = "Date Range"

    if btn_col2.button("Last 4Wks", help="Monday of 4 weeks ago to Today", use_container_width=True):
        this_monday = today - datetime.timedelta(days=today.weekday())
        start_4w = this_monday - datetime.timedelta(days=28)
        st.session_state.date_range_val = (start_4w, today)
        st.session_state.date_mode = "Date Range"
        
    if btn_col3.button("Reset", help="Reset to Phase Defaults", use_container_width=True):
        st.session_state.date_range_val = (default_start, default_end)
        st.session_state.date_mode = "Date Range"
    
    if 'date_mode' not in st.session_state:
        st.session_state.date_mode = "Date Range"
        
    date_mode = st.sidebar.radio("Date Selection", ["Date Range", "Single Date"], key="date_mode")
    
    if date_mode == "Single Date":
        selected_date = st.sidebar.date_input("Select Date", value=st.session_state.date_range_val[1])
        start_date = selected_date
        end_date = selected_date
    else:
        date_range = st.sidebar.date_input("Select Date Range", value=st.session_state.date_range_val)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            st.session_state.date_range_val = date_range
        elif isinstance(date_range, tuple) and len(date_range) == 1:
            start_date = date_range[0]
            end_date = date_range[0]
        else:
            start_date = default_start
            end_date = default_end
            
    all_sites = ['NCT Delhi', 'Pune', 'Chennai', 'Jodhpur', 'Kolkata', 'Guwahati']
    selected_sites = st.sidebar.multiselect("Select Site(s)", options=all_sites, default=all_sites)
    
    all_cadres = ['Doctor', 'Nurse', 'Allied']
    selected_cadres = st.sidebar.multiselect("Select Cadre(s)", options=all_cadres, default=all_cadres)

    # Tabs
    tab1, tab2a, tab2, tab3, tab4, tab6, tab7 = st.tabs([
        "🛗 Cohort summary", 
        "ℹ️ Daily surveillance",
        "💬 Weekly response rate", 
        "📈 ILI Rate", 
        "🦠 COVID / Flu Positivity", 
        "📝 Outcome of Illness",
        "⚠️ Non Respondents"
    ])
    
    def style_table(df):
        return df.style.apply(
            lambda x: ['background-color: rgba(150, 150, 150, 0.1)' if i % 2 == 0 else '' for i in range(len(x))], 
            axis=0
        ).set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}])

    # ==========================================
    # TAB 1: COHORT SUMMARY
    # ==========================================
    with tab1:
        if 'ILI Entities' in datasets and not datasets['ILI Entities'].empty:
            df_all = datasets['ILI Entities'].copy()
            
            if 'city' in df_all.columns:
                df_all['Site_Name'] = df_all['city'].map(CITY_MAP).fillna(df_all['city'])
            else:
                df_all['Site_Name'] = 'Unknown Site'
                
            cadre_col = 'pat_cadre_status' if 'pat_cadre_status' in df_all.columns else None
            
            df_all['Status'] = 'Active'
            if 'excluded' in df_all.columns:
                is_excluded = df_all['excluded'].astype(str).str.strip() == '1'
                df_all.loc[is_excluded, 'Status'] = 'Dropped Out'
            
            if selected_sites:
                df_all = df_all[df_all['Site_Name'].isin(selected_sites)]
                
            if cadre_col and selected_cadres:
                pattern = '|'.join(selected_cadres)
                df_all = df_all[df_all[cadre_col].str.contains(pattern, case=False, na=False)]
            
            df_entities = df_all[df_all['Status'] == 'Active'].copy()
            df_excluded = df_all[df_all['Status'] == 'Dropped Out'].copy()
            
            if cadre_col:
                st.subheader("Enrolments by Site and Cadre")
                
                fig_cadre_site = go.Figure()
                sites_ordered = sorted(df_all['Site_Name'].unique())
                cadres_to_plot = [c for c in ['Doctor', 'Nurse', 'Allied'] if c in df_all[cadre_col].unique()]
                if not cadres_to_plot: 
                    cadres_to_plot = ['Doctor', 'Nurse', 'Allied']
                
                cadre_colors = {'Doctor': '#2E4B71', 'Nurse': '#85B65A', 'Allied': '#C87550'}
                dropout_color = '#E0E0E0'
                
                for cadre in cadres_to_plot:
                    active_counts = []
                    dropped_counts = []
                    for site in sites_ordered:
                        subset = df_all[(df_all['Site_Name'] == site) & (df_all[cadre_col] == cadre)]
                        active_counts.append(len(subset[subset['Status'] == 'Active']))
                        dropped_counts.append(len(subset[subset['Status'] == 'Dropped Out']))
                        
                    fig_cadre_site.add_trace(go.Bar(
                        name=f"{cadre} (Active)", x=sites_ordered, y=active_counts,
                        offsetgroup=cadre, marker_color=cadre_colors.get(cadre, 'gray'), legendgroup=cadre
                    ))
                    fig_cadre_site.add_trace(go.Bar(
                        name=f"{cadre} (Dropped)", x=sites_ordered, y=dropped_counts,
                        offsetgroup=cadre, base=active_counts, marker_color=dropout_color,
                        showlegend=False, hoverinfo='x+y+name'
                    ))
                
                fig_cadre_site.add_trace(go.Bar(name='Dropped Out', x=[None], y=[None], marker_color=dropout_color))

                fig_cadre_site.update_layout(
                    barmode='group', yaxis_title="Number of Enrolments", xaxis_title="",
                    legend_title_text="Cadre & Status", margin=dict(t=20, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_cadre_site, use_container_width=True, key="enrol_chart")
                
                table1_data = []
                for site in sites_ordered:
                    row = {'Study Site': site}
                    for cadre in cadres_to_plot:
                        subset = df_all[(df_all['Site_Name'] == site) & (df_all[cadre_col] == cadre)]
                        row[f'{cadre}s Enrolled'] = len(subset)
                        row[f'{cadre} Drop out'] = len(subset[subset['Status'] == 'Dropped Out'])
                    table1_data.append(row)
                    
                df_table1 = pd.DataFrame(table1_data)
                
                if not df_table1.empty:
                    total_row = {'Study Site': 'Grand Total'}
                    for col in df_table1.columns[1:]: 
                        total_row[col] = df_table1[col].sum()
                    df_table1 = pd.concat([df_table1, pd.DataFrame([total_row])], ignore_index=True)
                
                t1_col1, t1_col2 = st.columns([2.5, 1])
                with t1_col1:
                    st.dataframe(style_table(df_table1), use_container_width=True, hide_index=True)
                with t1_col2:
                    enrolled_site_totals = df_all.groupby('Site_Name').size()
                    grand_total_enrolled = enrolled_site_totals.sum()
                    card_html_enrolled = f"""
                    <div style='background-color:rgba(150, 150, 150, 0.1);padding:20px;border-radius:10px;'>
                        <h4 style='margin-top:0;text-align:center;'>Total Enrolled</h4>
                        <h1 style='color:#C87550;margin:0;text-align:center;font-size:3rem;'>{grand_total_enrolled}</h1>
                        <hr style='margin:15px 0; border-color:rgba(0,0,0,0.1);'>
                    """
                    for site, count in enrolled_site_totals.items():
                        card_html_enrolled += f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span><b>{site}</b></span><span style='color:#C87550;font-weight:bold;'>{count}</span></div>"
                    card_html_enrolled += "</div>"
                    st.markdown(card_html_enrolled, unsafe_allow_html=True)
                
                st.divider()

                st.subheader("Cohort under surveillance currently")
                t2_col1, t2_col2 = st.columns([2.5, 1])
                
                with t2_col1:
                    crosstab_active = pd.crosstab(df_entities['Site_Name'], df_entities[cadre_col])
                    ordered_cols = [col for col in ['Doctor', 'Nurse', 'Allied'] if col in crosstab_active.columns]
                    other_cols = [col for col in crosstab_active.columns if col not in ordered_cols]
                    crosstab_active = crosstab_active[ordered_cols + other_cols]
                    
                    if not crosstab_active.empty:
                        crosstab_active.loc['Grand Total'] = crosstab_active.sum(axis=0)
                        crosstab_active = crosstab_active.reset_index().rename(columns={'Site_Name': 'Study Site'})
                        st.dataframe(style_table(crosstab_active), use_container_width=True, hide_index=True)
                    else:
                        st.info("No active participants under surveillance for the selected filters.")
                        
                with t2_col2:
                    site_totals = df_entities.groupby('Site_Name').size()
                    grand_total = site_totals.sum()
                    card_html = f"""
                    <div style='background-color:rgba(150, 150, 150, 0.1);padding:20px;border-radius:10px;'>
                        <h4 style='margin-top:0;text-align:center;'>Total Active</h4>
                        <h1 style='color:#2E4B71;margin:0;text-align:center;font-size:3rem;'>{grand_total}</h1>
                        <hr style='margin:15px 0; border-color:rgba(0,0,0,0.1);'>
                    """
                    for site, count in site_totals.items():
                        card_html += f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span><b>{site}</b></span><span style='color:#2E4B71;font-weight:bold;'>{count}</span></div>"
                    card_html += "</div>"
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("Cadre variable ('pat_cadre_status') not found.")

            st.divider()

            age_col = None
            if 'info-pat_age' in df_entities.columns:
                age_col = 'info-pat_age'
            else:
                matches = [c for c in df_entities.columns if 'age' in str(c).lower() and 'cadre' not in str(c).lower()]
                if matches: 
                    age_col = matches[0]
            
            if age_col:
                df_entities['age_numeric'] = pd.to_numeric(df_entities[age_col], errors='coerce')

            col1, col2, col3 = st.columns([1.2, 1.5, 1.3])
            
            with col1:
                st.subheader("Sex Distribution")
                sex_col = None
                if 'info-pat_gender' in df_entities.columns:
                    sex_col = 'info-pat_gender'
                else:
                    matches = [c for c in df_entities.columns if 'gender' in str(c).lower()]
                    if matches: 
                        sex_col = matches[0]
                
                if sex_col:
                    sex_colors = {'Male': 'darkviolet', 'Female': 'pink', 'male': 'darkviolet', 'female': 'pink'}
                    fig_sex = px.pie(df_entities, names=sex_col, color=sex_col, hole=0.4, color_discrete_map=sex_colors)
                    fig_sex.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_sex, use_container_width=True)
                else:
                    st.info("Sex/Gender variable not found.")

            with col2:
                st.subheader("Age Breakdown")
                if age_col:
                    st.markdown("**By Site**")
                    age_by_site = df_entities.groupby('Site_Name')['age_numeric'].agg(['mean', 'std']).reset_index()
                    age_by_site.columns = ['Site Name', 'Mean Age', 'Standard Deviation']
                    st.dataframe(
                        style_table(age_by_site), use_container_width=True, hide_index=True,
                        column_config={
                            "Site Name": st.column_config.TextColumn("Study Site"),
                            "Mean Age": st.column_config.NumberColumn("Mean Age", format="%.1f"),
                            "Standard Deviation": st.column_config.NumberColumn("Standard Deviation", format="%.2f")
                        }
                    )
                    
                    if cadre_col:
                        st.markdown("**By Cadre**")
                        age_by_cadre = df_entities.groupby(cadre_col)['age_numeric'].agg(['mean', 'std']).reset_index()
                        age_by_cadre.columns = ['HCW Cadre', 'Mean Age', 'Standard Deviation']
                        st.dataframe(
                            style_table(age_by_cadre), use_container_width=True, hide_index=True,
                            column_config={
                                "HCW Cadre": st.column_config.TextColumn("HCW Cadre"),
                                "Mean Age": st.column_config.NumberColumn("Mean Age", format="%.1f"),
                                "Standard Deviation": st.column_config.NumberColumn("Standard Deviation", format="%.2f")
                            }
                        )
                else:
                    st.info("Age variable not found.")
            
            with col3:
                st.subheader("Age Statistics")
                if age_col and not df_entities['age_numeric'].dropna().empty:
                    overall_mean = df_entities['age_numeric'].mean()
                    overall_sd = df_entities['age_numeric'].std()
                    st.markdown("Mean Age")
                    st.markdown(f"<h1 style='margin: 0; padding: 0;'>{overall_mean:.1f} yrs</h1>", unsafe_allow_html=True)
                    st.markdown(f"<span style='background-color: #f0f2f6; padding: 2px 8px; border-radius: 12px; color: #555;'>↑ ± {overall_sd:.1f} SD</span>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Age Distribution**")
                    fig_age_dist = px.histogram(df_entities, x='age_numeric', color_discrete_sequence=['#3273c4'])
                    fig_age_dist.update_traces(xbins=dict(size=1))
                    fig_age_dist.update_layout(yaxis_title="Count", xaxis_title="Age", margin=dict(t=10, b=0, l=0, r=0), bargap=0.1)
                    st.plotly_chart(fig_age_dist, use_container_width=True)
                else:
                    st.info("Age data not available for statistics.")
            
            st.divider()
            
            st.subheader(f"Excluded Participants ({len(df_excluded)})")
            if not df_excluded.empty:
                st.info("The following participants were excluded from the metrics above.")
                exit_df = datasets.get('Exit', pd.DataFrame())
                
                df_excluded['Name'] = get_nested_col(df_excluded, 'pat_name')
                df_excluded['Age'] = get_nested_col(df_excluded, 'pat_age')
                df_excluded['Sex'] = get_nested_col(df_excluded, 'pat_gender')
                df_excluded['Department'] = get_nested_col(df_excluded, 'pat_department')
                df_excluded['Designation'] = get_nested_col(df_excluded, 'pat_designation')
                df_excluded['Age & Sex'] = df_excluded.apply(lambda r: format_age_sex_row(r, 'Age', 'Sex'), axis=1)
                df_excluded['Serial No.'] = range(1, len(df_excluded) + 1)
                df_excluded['exclusion_reason'] = get_nested_col(df_excluded, 'exclusion_reason')
                
                if not exit_df.empty and 'exclusion_reason' in df_excluded.columns:
                    exit_df['exclusion_id_match'] = get_nested_col(exit_df, 'exclusion_id')
                    exit_df['Enrollment Date'] = get_nested_col(exit_df, 'pat_enrollment_date')
                    exit_df['Last Survey Date'] = get_nested_col(exit_df, 'display_last_survey_date')
                    exit_df['ILI Episodes'] = get_nested_col(exit_df, 'pat_ili_episode')
                    exit_df['exit_reason_raw'] = get_nested_col(exit_df, 'q2_1')
                    exit_df['withdrawal_reason_raw'] = get_nested_col(exit_df, 'q3b_2')
                    exit_df['withdrawal_other'] = get_nested_col(exit_df, 'q3b_2_1')

                    q2_map = {'1': 'Participant Death', '2': 'Refusal / Withdrawal of Consent'}
                    q3_map = {'1': 'Too busy / time constraint', '2': 'Dislike of weekly tracking', '3': 'Resigned from current post / terminated',
                              '4': 'Transferred to another city / hospital', '5': 'No longer interested in the study', '9': 'Other'}
                    
                    def map_multiple(val, mapping):
                        if pd.isna(val): 
                            return ''
                        parts = str(val).split()
                        return ', '.join([mapping.get(p, p) for p in parts])
                        
                    exit_df['Exit Reason'] = exit_df['exit_reason_raw'].astype(str).str.strip().map(q2_map).fillna('')
                    exit_df['Withdrawal Reason Base'] = exit_df['withdrawal_reason_raw'].apply(lambda x: map_multiple(x, q3_map))
                    
                    def format_withdrawal(row):
                        reason = row['Withdrawal Reason Base']
                        other = row['withdrawal_other']
                        if pd.notna(other) and str(other).strip() not in ('', 'None'):
                            return f"{reason} ({str(other).strip()})"
                        return reason
                        
                    exit_df['Withdrawal Reason'] = exit_df.apply(format_withdrawal, axis=1)
                    exit_df['Enrollment Date'] = pd.to_datetime(exit_df['Enrollment Date'], errors='coerce').dt.strftime('%d-%b-%Y').fillna(exit_df['Enrollment Date'])
                    exit_df['Last Survey Date'] = pd.to_datetime(exit_df['Last Survey Date'], errors='coerce').dt.strftime('%d-%b-%Y').fillna(exit_df['Last Survey Date'])
                    
                    merged_ex = pd.merge(
                        df_excluded, 
                        exit_df[['exclusion_id_match', 'Enrollment Date', 'Last Survey Date', 'ILI Episodes', 'Exit Reason', 'Withdrawal Reason']],
                        left_on='exclusion_reason', right_on='exclusion_id_match', how='left'
                    )
                    
                    merged_ex['Cadre'] = merged_ex[cadre_col] if cadre_col else 'Unknown'
                    merged_ex['Site'] = merged_ex['Site_Name']
                    
                    display_cols = ['Serial No.', 'Name', 'Age & Sex', 'exclusion_reason', 'Department', 'Designation', 'Cadre', 'Site',
                                    'Enrollment Date', 'Last Survey Date', 'ILI Episodes', 'Exit Reason', 'Withdrawal Reason']
                    
                    show_df = merged_ex[[c for c in display_cols if c in merged_ex.columns]].fillna('')
                    st.dataframe(
                        style_table(show_df), use_container_width=True, hide_index=True,
                        column_config={"Serial No.": st.column_config.NumberColumn("S.No.", width="small"), "exclusion_reason": st.column_config.TextColumn("Exclusion ID", width="small")}
                    )
                else:
                    df_excluded['Cadre'] = df_excluded[cadre_col] if cadre_col else 'Unknown'
                    df_excluded['Site'] = df_excluded['Site_Name']
                    display_cols = ['Serial No.', 'Name', 'Age & Sex', 'exclusion_reason', 'Department', 'Designation', 'Cadre', 'Site']
                    show_df = df_excluded[[c for c in display_cols if c in df_excluded.columns]].fillna('')
                    st.dataframe(
                        style_table(show_df), use_container_width=True, hide_index=True,
                        column_config={"Serial No.": st.column_config.NumberColumn("S.No.", width="small"), "exclusion_reason": st.column_config.TextColumn("Exclusion ID", width="small")}
                    )
                    
                st.divider()
                st.subheader("Exclusion Trends")
                ex_col1, ex_col2 = st.columns(2)
                
                with ex_col1:
                    st.markdown("**Site-wise participants exited**")
                    site_ex_counts = df_excluded['Site_Name'].value_counts().reset_index()
                    site_ex_counts.columns = ['Site', 'Count']
                    
                    fig_ex_site = px.bar(
                        site_ex_counts, x='Site', y='Count', text='Count',
                        color_discrete_sequence=['#C87550']
                    )
                    fig_ex_site.update_traces(textposition='outside')
                    y_max_site = site_ex_counts['Count'].max() * 1.2 if not site_ex_counts.empty and site_ex_counts['Count'].max() > 0 else 10
                    fig_ex_site.update_layout(
                        yaxis_title="Participants Exited (n)", 
                        xaxis_title="", 
                        yaxis=dict(range=[0, y_max_site]), 
                        margin=dict(t=10, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig_ex_site, use_container_width=True)
                    
                with ex_col2:
                    st.markdown("**Month-wise participants exited**")
                    exit_df_chart = datasets.get('Exit', pd.DataFrame())
                    
                    if not exit_df_chart.empty and 'exclusion_reason' in df_excluded.columns:
                        exit_df_chart['exclusion_id_match'] = get_nested_col(exit_df_chart, 'exclusion_id')
                        exit_df_chart['Exit Date'] = pd.to_datetime(get_nested_col(exit_df_chart, 'today'), errors='coerce')
                        
                        df_ex_dates = pd.merge(
                            df_excluded[['exclusion_reason']], 
                            exit_df_chart[['exclusion_id_match', 'Exit Date']], 
                            left_on='exclusion_reason', 
                            right_on='exclusion_id_match', 
                            how='inner'
                        )
                        df_ex_dates = df_ex_dates.dropna(subset=['Exit Date'])
                        
                        if not df_ex_dates.empty:
                            df_ex_dates['Sort_YM'] = df_ex_dates['Exit Date'].dt.to_period('M')
                            df_ex_dates['Month & Year'] = df_ex_dates['Exit Date'].dt.strftime('%b %Y')
                            
                            month_ex_counts = df_ex_dates.groupby(['Sort_YM', 'Month & Year']).size().reset_index(name='Count')
                            month_ex_counts = month_ex_counts.sort_values('Sort_YM')
                            
                            fig_ex_month = px.bar(
                                month_ex_counts, x='Month & Year', y='Count', text='Count',
                                color_discrete_sequence=['#2E4B71']
                            )
                            fig_ex_month.update_traces(textposition='outside')
                            y_max_month = month_ex_counts['Count'].max() * 1.2 if month_ex_counts['Count'].max() > 0 else 10
                            fig_ex_month.update_layout(
                                yaxis_title="Participants Exited (n)", 
                                xaxis_title="", 
                                yaxis=dict(range=[0, y_max_month]), 
                                margin=dict(t=10, b=0, l=0, r=0)
                            )
                            st.plotly_chart(fig_ex_month, use_container_width=True)
                        else:
                            st.info("Valid exit dates not available to generate the month-wise chart.")
                    else:
                        st.info("Exit form data not available to generate the month-wise chart.")
            else:
                st.success("No excluded participants found for the selected filters.")
        else:
            st.warning("No Entity data found. Check if ENTITY_ID_ILI is correct and data exists.")

    # ==========================================
    # TAB 2a: DAILY SURVEILLANCE DATA
    # ==========================================
    with tab2a:
        if 'ILI Entities' in datasets and 'Surveillance' in datasets and not datasets['Surveillance'].empty and not datasets['ILI Entities'].empty:
            surv_df2a = datasets['Surveillance'].copy()
            ent_df2a = datasets['ILI Entities'].copy()

            if 'excluded' in ent_df2a.columns:
                is_excluded = ent_df2a['excluded'].astype(str).str.strip() == '1'
                ent_df2a = ent_df2a[~is_excluded]

            hosp_col_ent = next((c for c in ent_df2a.columns if 'hospital' in c.lower() or 'facility' in c.lower()), None)
            ent_city_col = 'city' if 'city' in ent_df2a.columns else next((c for c in ent_df2a.columns if 'city' in c.lower()), None)
            
            cadre_col_ent = 'pat_cadre_status'
            if cadre_col_ent not in ent_df2a.columns:
                ent_df2a[cadre_col_ent] = get_nested_col(ent_df2a, 'pat_cadre_status').fillna('Unknown')
            
            if ent_city_col:
                ent_df2a['Site_Chart'] = ent_df2a.apply(lambda row: get_phase_global(row.get(ent_city_col), row.get(hosp_col_ent), study_phase), axis=1)
                ent_df2a['Site_Name'] = ent_df2a[ent_city_col].map(CITY_MAP).fillna(ent_df2a[ent_city_col])
            else:
                ent_df2a['Site_Chart'] = 'Unknown'
                ent_df2a['Site_Name'] = 'Unknown'

            ent_df2a['clean_phone'] = clean_phone_series(get_nested_col(ent_df2a, 'phone_number'))

            surv_df2a['extracted_city'] = get_nested_col(surv_df2a, 'pat_city')
            surv_df2a['extracted_phone'] = get_nested_col(surv_df2a, 'pat_phone')
            surv_df2a['extracted_hosp'] = get_nested_col(surv_df2a, 'pat_hospital')
            if surv_df2a['extracted_hosp'].isna().all(): 
                surv_df2a['extracted_hosp'] = get_nested_col(surv_df2a, 'hospital')
            if surv_df2a['extracted_hosp'].isna().all(): 
                surv_df2a['extracted_hosp'] = get_nested_col(surv_df2a, 'facility')

            surv_col = 'extracted_phone' if surv_df2a['extracted_phone'].notna().any() else ('phone_no' if 'phone_no' in surv_df2a.columns else next((c for c in surv_df2a.columns if 'phone' in c.lower()), None))
            surv_df2a['clean_phone'] = clean_phone_series(surv_df2a[surv_col]) if surv_col else ''
                
            surv_city_col = 'extracted_city' if surv_df2a['extracted_city'].notna().any() else next((c for c in surv_df2a.columns if 'city' in c.lower()), None)
            if surv_city_col:
                surv_df2a['surv_native_Site_Chart'] = surv_df2a.apply(lambda row: get_phase_global(row.get(surv_city_col), row.get('extracted_hosp'), study_phase), axis=1)
                surv_df2a['surv_native_Site_Name'] = surv_df2a[surv_city_col].map(CITY_MAP).fillna(surv_df2a[surv_city_col])
            else:
                surv_df2a['surv_native_Site_Chart'] = 'Unknown'
                surv_df2a['surv_native_Site_Name'] = 'Unknown'
                
            if 'clean_phone' in ent_df2a.columns and 'clean_phone' in surv_df2a.columns:
                ent_subset = ent_df2a[ent_df2a['clean_phone'] != ''][['clean_phone', cadre_col_ent, 'Site_Chart', 'Site_Name']].drop_duplicates(subset=['clean_phone'])
                surv_df2a = surv_df2a.merge(ent_subset, on='clean_phone', how='left')
                surv_df2a['Site_Chart'] = surv_df2a['Site_Chart'].fillna(surv_df2a['surv_native_Site_Chart'])
                surv_df2a['Site_Name'] = surv_df2a['Site_Name'].fillna(surv_df2a['surv_native_Site_Name'])
                surv_df2a[cadre_col_ent] = surv_df2a[cadre_col_ent].fillna('Unknown')
            else:
                surv_df2a['Site_Chart'] = surv_df2a['surv_native_Site_Chart']
                surv_df2a['Site_Name'] = surv_df2a['surv_native_Site_Name']
                surv_df2a[cadre_col_ent] = 'Unknown'

            # Apply Global Filters
            if selected_sites:
                surv_df2a = surv_df2a[surv_df2a['Site_Name'].isin(selected_sites)]
                ent_df2a = ent_df2a[ent_df2a['Site_Name'].isin(selected_sites)]
                
            # Date filter logic
            date_series = get_nested_col(surv_df2a, 'today')
            if date_series.isna().all() and '__system' in surv_df2a.columns:
                date_series = surv_df2a['__system'].apply(lambda x: x.get('submissionDate') if isinstance(x, dict) else None)

            if date_series is not None and not date_series.isna().all():
                surv_df2a['today_dt'] = pd.to_datetime(date_series, errors='coerce').dt.date
                surv_df2a = surv_df2a[(surv_df2a['today_dt'] >= start_date) & (surv_df2a['today_dt'] <= end_date)]
                
            if selected_cadres:
                pattern = '|'.join(selected_cadres)
                surv_df2a = surv_df2a[surv_df2a[cadre_col_ent].str.contains(pattern, case=False, na=False)]
                ent_df2a = ent_df2a[ent_df2a[cadre_col_ent].str.contains(pattern, case=False, na=False)]

            delta = end_date - start_date
            total_days = delta.days + 1
            num_weeks = max(1, round(total_days / 7))
            
            contact_day_col = 'pat_contact_day' if 'pat_contact_day' in ent_df2a.columns else next((c for c in ent_df2a.columns if 'contact' in str(c).lower() or 'day' in str(c).lower()), None)
            if not contact_day_col:
                ent_df2a['pat_contact_day'] = get_nested_col(ent_df2a, 'pat_contact_day')
                if ent_df2a['pat_contact_day'].notna().any():
                    contact_day_col = 'pat_contact_day'
            
            if contact_day_col:
                weekday_counts = {i: 0 for i in range(7)}
                for d in [start_date + datetime.timedelta(days=i) for i in range(total_days)]: 
                    weekday_counts[d.weekday()] += 1
                day_map = {'1': 0, 'monday': 0, 'mon': 0, '2': 1, 'tuesday': 1, 'tue': 1,
                           '3': 2, 'wednesday': 2, 'wed': 2, '4': 3, 'thursday': 3, 'thu': 3,
                           '5': 4, 'friday': 4, 'fri': 4}
                ent_df2a['contact_wd'] = ent_df2a[contact_day_col].astype(str).str.lower().str.strip().map(day_map)
                ent_df2a['forms_due'] = ent_df2a['contact_wd'].map(weekday_counts).fillna(num_weeks)
            else:
                ent_df2a['forms_due'] = num_weeks
                
            total_due = int(ent_df2a['forms_due'].sum())
                
            total_filled = len(surv_df2a)
            if isinstance(total_due, (int, float)) and total_due > 0:
                filled_pct = (total_filled / total_due) * 100
                filled_display = f"{total_filled} ({filled_pct:.1f}%)"
            else:
                filled_display = str(total_filled)
            
            st.subheader("Form Response Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;'><h4>Total Forms Due in selected date range</h4><h1 style='color:#C87550;margin:0;'>{total_due}</h1></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;'><h4>Total Forms Filled</h4><h1 style='color:#85B65A;margin:0;'>{filled_display}</h1></div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Submissions Summary (Selected Date Range)")
            
            week_col_raw = 'weekofyear'
            if week_col_raw not in surv_df2a.columns:
                surv_df2a[week_col_raw] = pd.to_datetime(surv_df2a.get('today_dt', pd.Series(dtype='object')), errors='coerce').dt.strftime('%U')

            surv_df2a['week_num'] = pd.to_numeric(surv_df2a[week_col_raw], errors='coerce').fillna(0).astype(int)
            surv_df2a['Week'] = 'Week ' + surv_df2a['week_num'].astype(str).str.zfill(2)
            ordered_weeks2a = sorted(surv_df2a[surv_df2a['week_num'] > 0]['Week'].unique())
            
            def get_submitter_raw(row):
                if 'SubmitterName' in row and pd.notna(row['SubmitterName']) and str(row['SubmitterName']).strip() != '': 
                    return str(row['SubmitterName']).strip()
                if '__system' in row and isinstance(row['__system'], dict):
                    sys_name = row['__system'].get('submitterName')
                    if sys_name: 
                        return str(sys_name).strip()
                return "Unknown/Other"
            
            def map_sub_summary(x):
                x = str(x).strip()
                if study_phase == "Main Study":
                    if "Participant" in x: return "Forms filled by Participant"
                    if "Data Collector" in x: return "Forms Filled by Data Collector"
                else:
                    if "Participant" in x and "Main" not in x: return "Forms filled by Participant"
                    if "Data Collector" in x and "Main" not in x: return "Forms Filled by Data Collector"
                    if "Participant" in x: return "Forms filled by Participant"
                    if "Data Collector" in x: return "Forms Filled by Data Collector"
                return "Forms Filled by Data Collector"
                
            surv_df2a['Raw_Submitter'] = surv_df2a.apply(get_submitter_raw, axis=1)
            surv_df2a['Sub_Category'] = surv_df2a['Raw_Submitter'].apply(map_sub_summary)

            site_due_dict = ent_df2a.groupby('Site_Chart')['forms_due'].sum().to_dict()
            chart_data = []
            ordered_y_axis = []
            
            for site, den in site_due_dict.items():
                den = int(den)
                if den == 0: 
                    continue
                site_ent_subset = ent_df2a[ent_df2a['Site_Chart'] == site]
                valid_phones = set(site_ent_subset['clean_phone'])
                site_surv = surv_df2a[surv_df2a['Site_Chart'] == site]
                phones_part = site_surv[site_surv['Sub_Category'] == 'Forms filled by Participant']['clean_phone'].tolist()
                phones_dc = site_surv[site_surv['Sub_Category'] == 'Forms Filled by Data Collector']['clean_phone'].tolist()
                part_count = len([p for p in phones_part if p in valid_phones])
                dc_count = len([p for p in phones_dc if p in valid_phones])
                not_filled = max(0, den - part_count - dc_count)
                
                y_label = site
                if y_label not in ordered_y_axis: 
                    ordered_y_axis.append(y_label)
                
                chart_data.extend([
                    {'Site_Chart': y_label, 'Category': 'Forms filled by Participant', 'Percentage': (part_count / den) * 100 if den > 0 else 0, 'Count': part_count},
                    {'Site_Chart': y_label, 'Category': 'Forms Filled by Data Collector', 'Percentage': (dc_count / den) * 100 if den > 0 else 0, 'Count': dc_count},
                    {'Site_Chart': y_label, 'Category': 'Forms not filled', 'Percentage': (not_filled / den) * 100 if den > 0 else 0, 'Count': not_filled}
                ])
                
            df_chart = pd.DataFrame(chart_data)
            
            if not df_chart.empty:
                fig_summary = px.bar(
                    df_chart, y='Site_Chart', x='Percentage', color='Category', orientation='h',
                    color_discrete_map={'Forms filled by Participant': '#85B65A', 'Forms Filled by Data Collector': '#2E4B71', 'Forms not filled': '#E0E0E0'},
                    category_orders={"Site_Chart": list(reversed(sorted(ordered_y_axis)))},
                    hover_data={'Count': True, 'Percentage': ':.1f'}
                )
                fig_summary.update_layout(barmode='stack', xaxis_title="% of forms filled", yaxis_title="", margin=dict(t=20, b=0, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, title=""))
                fig_summary.update_traces(texttemplate='%{x:.0f}%', textposition='inside')
                fig_summary.update_layout(height=max(300, len(ordered_y_axis) * 60))
                st.plotly_chart(fig_summary, use_container_width=True, key="submissions_summary_stacked_col")
            else:
                st.info("No data available to generate submissions summary chart.")
                
            st.divider()

            surv_df2a['Submitter'] = surv_df2a['Raw_Submitter']
            st.subheader("Forms Filled by Site & Submitter")
            if not surv_df2a.empty:
                submitter_agg = surv_df2a.groupby(['Site_Chart', 'Submitter']).size().reset_index(name='Count')
                fig_sub = px.bar(
                    submitter_agg, x='Site_Chart', y='Count', color='Submitter', barmode='stack', text='Count',
                    category_orders={'Site_Chart': sorted(surv_df2a['Site_Chart'].unique())},
                    labels={'Site_Chart': '', 'Count': 'Forms Filled (n)', 'Submitter': 'Filled By'},
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_sub.update_traces(textposition='inside')
                fig_sub.update_layout(margin=dict(t=20, b=0, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
                st.plotly_chart(fig_sub, use_container_width=True)
                
                st.markdown("**Submitter Breakdown (%)**")
                sub_totals = surv_df2a['Submitter'].value_counts().reset_index()
                sub_totals.columns = ['Submitter', 'Forms Filled']
                sub_totals['Percentage'] = (sub_totals['Forms Filled'] / total_filled * 100).round(1).astype(str) + "%"
                st.dataframe(style_table(sub_totals), use_container_width=True, hide_index=True)
                
                st.divider()
                st.subheader("Surveillance Submissions Line List")
                
                surv_ll = surv_df2a.copy()
                surv_ll['Serial No.'] = range(1, len(surv_ll) + 1)
                
                surv_ll['Name'] = get_nested_col(surv_ll, 'pat_name')
                surv_ll['Age'] = get_nested_col(surv_ll, 'pat_age')
                surv_ll['Sex'] = get_nested_col(surv_ll, 'pat_gender')
                surv_ll['Age & Sex'] = surv_ll.apply(lambda r: format_age_sex_row(r, 'Age', 'Sex'), axis=1)
                surv_ll['Designation'] = get_nested_col(surv_ll, 'pat_designation')
                surv_ll['Department'] = get_nested_col(surv_ll, 'pat_department')
                surv_ll['Cadre'] = surv_ll[cadre_col_ent]
                surv_ll['Site'] = surv_ll['Site_Name']
                surv_ll['Form Filled By'] = surv_ll['Submitter']
                
                if 'clean_phone' in surv_ll.columns and 'clean_phone' in ent_df2a.columns:
                    if contact_day_col:
                        ent_days = ent_df2a[['clean_phone', contact_day_col]].drop_duplicates('clean_phone')
                        surv_ll = surv_ll.merge(ent_days, on='clean_phone', how='left')
                        surv_ll['Day of Surveillance'] = surv_ll[contact_day_col].fillna('Unknown').astype(str).str.title()
                    else:
                        surv_ll['Day of Surveillance'] = 'Unknown'
                else:
                    surv_ll['Day of Surveillance'] = 'Unknown'

                display_cols_2a = ['Serial No.', 'Name', 'Age & Sex', 'Designation', 'Cadre', 'Site', 'Form Filled By', 'Day of Surveillance']
                surv_ll_disp = surv_ll[[c for c in display_cols_2a if c in surv_ll.columns]].fillna('')
                
                st.dataframe(
                    style_table(surv_ll_disp), use_container_width=True, hide_index=True,
                    column_config={"Serial No.": st.column_config.NumberColumn("S.No.", width="small")}
                )
            else:
                st.info("No surveillance forms filled in the selected date range.")
        else:
            st.info("Awaiting data from both 'ILI Entities' and 'Surveillance' forms.")

    # ==========================================
    # TAB 3: WEEKLY RESPONSE RATE
    # ==========================================
    with tab2:
        if 'ILI Entities' in datasets and 'Surveillance' in datasets and not datasets['Surveillance'].empty and not datasets['ILI Entities'].empty:
            surv_df = datasets['Surveillance'].copy()
            ent_df = datasets['ILI Entities'].copy()

            if 'excluded' in ent_df.columns:
                is_excluded = ent_df['excluded'].astype(str).str.strip() == '1'
                ent_df = ent_df[~is_excluded]

            hosp_col_ent = next((c for c in ent_df.columns if 'hospital' in c.lower() or 'facility' in c.lower()), None)
            ent_city_col = 'city' if 'city' in ent_df.columns else next((c for c in ent_df.columns if 'city' in c.lower()), None)
            cadre_col_ent = 'pat_cadre_status'
            if cadre_col_ent not in ent_df.columns: 
                ent_df[cadre_col_ent] = 'Unknown'
            
            if ent_city_col:
                ent_df['Site_Chart'] = ent_df.apply(lambda row: get_phase_global(row.get(ent_city_col), row.get(hosp_col_ent), study_phase), axis=1)
                ent_df['Site_Name'] = ent_df[ent_city_col].map(CITY_MAP).fillna(ent_df[ent_city_col])
            else:
                ent_df['Site_Chart'] = 'Unknown'
                ent_df['Site_Name'] = 'Unknown'

            ent_df['clean_phone'] = clean_phone_series(get_nested_col(ent_df, 'phone_number'))

            surv_df['extracted_city'] = get_nested_col(surv_df, 'pat_city')
            surv_df['extracted_phone'] = get_nested_col(surv_df, 'pat_phone')
            surv_df['extracted_hosp'] = get_nested_col(surv_df, 'pat_hospital')
            if surv_df['extracted_hosp'].isna().all(): 
                surv_df['extracted_hosp'] = get_nested_col(surv_df, 'hospital')
            if surv_df['extracted_hosp'].isna().all(): 
                surv_df['extracted_hosp'] = get_nested_col(surv_df, 'facility')

            surv_city_col = 'extracted_city' if surv_df['extracted_city'].notna().any() else next((c for c in surv_df.columns if 'city' in c.lower()), None)
            surv_col = 'extracted_phone' if surv_df['extracted_phone'].notna().any() else ('phone_no' if 'phone_no' in surv_df.columns else next((c for c in surv_df.columns if 'phone' in c.lower()), None))
            
            surv_df['clean_phone'] = clean_phone_series(surv_df[surv_col]) if surv_col else ''
                
            if surv_city_col:
                surv_df['surv_native_Site_Chart'] = surv_df.apply(lambda row: get_phase_global(row.get(surv_city_col), row.get('extracted_hosp'), study_phase), axis=1)
                surv_df['surv_native_Site_Name'] = surv_df[surv_city_col].map(CITY_MAP).fillna(surv_df[surv_city_col])
            else:
                surv_df['surv_native_Site_Chart'] = 'Unknown'
                surv_df['surv_native_Site_Name'] = 'Unknown'
                
            if 'clean_phone' in ent_df.columns and 'clean_phone' in surv_df.columns:
                ent_subset = ent_df[ent_df['clean_phone'] != ''][['clean_phone', cadre_col_ent, 'Site_Chart', 'Site_Name']].drop_duplicates(subset=['clean_phone'])
                surv_df = surv_df.merge(ent_subset, on='clean_phone', how='left')
                surv_df['Site_Chart'] = surv_df['Site_Chart'].fillna(surv_df['surv_native_Site_Chart'])
                surv_df['Site_Name'] = surv_df['Site_Name'].fillna(surv_df['surv_native_Site_Name'])
                surv_df[cadre_col_ent] = surv_df[cadre_col_ent].fillna('Unknown')
            else:
                surv_df['Site_Chart'] = surv_df['surv_native_Site_Chart']
                surv_df['Site_Name'] = surv_df['surv_native_Site_Name']
                surv_df[cadre_col_ent] = 'Unknown'

            if selected_sites: 
                surv_df = surv_df[surv_df['Site_Name'].isin(selected_sites)]
                
            date_series = get_nested_col(surv_df, 'today')
            if date_series.isna().all() and '__system' in surv_df.columns:
                date_series = surv_df['__system'].apply(lambda x: x.get('submissionDate') if isinstance(x, dict) else None)

            if date_series is not None and not date_series.isna().all():
                surv_df['today_dt'] = pd.to_datetime(date_series, errors='coerce').dt.date
                surv_df = surv_df[(surv_df['today_dt'] >= start_date) & (surv_df['today_dt'] <= end_date)]
                
            if selected_cadres:
                pattern = '|'.join(selected_cadres)
                surv_df = surv_df[surv_df[cadre_col_ent].str.contains(pattern, case=False, na=False)]

            week_col_raw = 'weekofyear'
            if week_col_raw not in surv_df.columns:
                surv_df[week_col_raw] = pd.to_datetime(surv_df.get('today_dt', pd.Series(dtype='object')), errors='coerce').dt.strftime('%U')

            surv_df['week_num'] = pd.to_numeric(surv_df[week_col_raw], errors='coerce').fillna(0).astype(int)
            surv_df['Week'] = 'Week ' + surv_df['week_num'].astype(str).str.zfill(2)
            ordered_weeks = sorted(surv_df[surv_df['week_num'] > 0]['Week'].unique())
            
            custom_colors = ['#C87550', '#C2982B', '#EAC13E', '#6495ED', '#85B65A', '#2E4B71', '#8B4513', '#2F4F4F', '#9b59b6', '#34495e']

            st.subheader("Response Rates (%)")
            if selected_sites: 
                ent_df = ent_df[ent_df['Site_Name'].isin(selected_sites)]
            if selected_cadres: 
                ent_df = ent_df[ent_df[cadre_col_ent].str.contains('|'.join(selected_cadres), case=False, na=False)]

            denom_site_all = ent_df.groupby('Site_Chart')['clean_phone'].nunique().to_dict() if 'clean_phone' in ent_df.columns else {}
            denom_cadre_all = ent_df.groupby(cadre_col_ent)['clean_phone'].nunique().to_dict() if 'clean_phone' in ent_df.columns else {}

            if 'pilot' in ent_df.columns and 'clean_phone' in ent_df.columns:
                pilot_mask = ent_df['pilot'].astype(str).str.strip() == '1'
                denom_site_pilot = ent_df[pilot_mask].groupby('Site_Chart')['clean_phone'].nunique().to_dict()
                denom_cadre_pilot = ent_df[pilot_mask].groupby(cadre_col_ent)['clean_phone'].nunique().to_dict()
            else:
                denom_site_pilot = denom_site_all
                denom_cadre_pilot = denom_cadre_all
                
            def get_rate(num, den): 
                return min((num / den) * 100, 100.0) if den > 0 else 0.0

            if 'clean_phone' in surv_df.columns and not surv_df.empty:
                num_site = surv_df[surv_df['week_num'] > 0].groupby(['Site_Chart', 'Week', 'week_num'])['clean_phone'].nunique().reset_index(name='Responded')
                num_site['Denominator'] = num_site.apply(
                    lambda r: denom_site_all.get(r['Site_Chart'], 0) if study_phase == "Main Study" else (denom_site_pilot.get(r['Site_Chart'], 0) if r['week_num'] <= 22 else denom_site_all.get(r['Site_Chart'], 0)), axis=1
                )
                num_site['Response Rate (%)'] = num_site.apply(lambda r: get_rate(r['Responded'], r['Denominator']), axis=1)
                
                pivot_site = num_site.pivot(index='Site_Chart', columns='Week', values='Response Rate (%)').round(1).fillna(0)
                pivot_site = pivot_site[[w for w in ordered_weeks if w in pivot_site.columns]].reset_index().rename(columns={'Site_Chart': 'Site / Phase'})
                
                st.markdown("**Overall Site-wise Response Rates**")
                st.dataframe(pivot_site.style.format(formatter={col: "{:.1f}%" for col in pivot_site.columns if col != 'Site / Phase'}).apply(lambda x: ['background-color: rgba(150, 150, 150, 0.1)' if i % 2 == 0 else '' for i in range(len(x))], axis=0).set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]), use_container_width=True, hide_index=True)
                
                fig_site_rr = px.bar(num_site, x='Site_Chart', y='Response Rate (%)', color='Week', barmode='group', text='Response Rate (%)', category_orders={'Week': ordered_weeks, 'Site_Chart': sorted(surv_df['Site_Chart'].unique())}, labels={'Site_Chart': '', 'Week': ''}, color_discrete_sequence=custom_colors)
                fig_site_rr.update_traces(textposition='outside', texttemplate='%{text:.0f}%')
                fig_site_rr.update_layout(yaxis=dict(range=[0, 120], title="Response Rate (%)"), xaxis_title="", margin=dict(t=20, b=0, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
                st.plotly_chart(fig_site_rr, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                num_cadre = surv_df[surv_df['week_num'] > 0].groupby([cadre_col_ent, 'Week', 'week_num'])['clean_phone'].nunique().reset_index(name='Responded')
                num_cadre['Denominator'] = num_cadre.apply(lambda r: denom_cadre_all.get(r[cadre_col_ent], 0) if study_phase == "Main Study" else (denom_cadre_pilot.get(r[cadre_col_ent], 0) if r['week_num'] <= 22 else denom_cadre_all.get(r[cadre_col_ent], 0)), axis=1)
                num_cadre['Response Rate (%)'] = num_cadre.apply(lambda r: get_rate(r['Responded'], r['Denominator']), axis=1)
                
                pivot_cadre = num_cadre.pivot(index=cadre_col_ent, columns='Week', values='Response Rate (%)').round(1).fillna(0)
                pivot_cadre = pivot_cadre[[w for w in ordered_weeks if w in pivot_cadre.columns]].reset_index().rename(columns={cadre_col_ent: 'HCW Cadre'})
                cadre_order = ['Doctor', 'Nurse', 'Allied']
                pivot_cadre['sort_idx'] = pivot_cadre['HCW Cadre'].apply(lambda x: cadre_order.index(x) if x in cadre_order else 99)
                pivot_cadre = pivot_cadre.sort_values('sort_idx').drop(columns=['sort_idx'])

                st.markdown("**Overall Cadre-wise Response Rates**")
                st.dataframe(pivot_cadre.style.format(formatter={col: "{:.1f}%" for col in pivot_cadre.columns if col != 'HCW Cadre'}).apply(lambda x: ['background-color: rgba(150, 150, 150, 0.1)' if i % 2 == 0 else '' for i in range(len(x))], axis=0).set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]), use_container_width=True, hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Site-wise Response Rates by Cadre**")
                col_d, col_n, col_a = st.columns(3)
                for cadre_name, col in [('Doctor', col_d), ('Nurse', col_n), ('Allied', col_a)]:
                    with col:
                        st.markdown(f"**{cadre_name}**")
                        cadre_ent = ent_df[ent_df[cadre_col_ent].str.contains(cadre_name, case=False, na=False)]
                        denom_all_sc = cadre_ent.groupby('Site_Chart')['clean_phone'].nunique().to_dict()
                        if 'pilot' in cadre_ent.columns:
                            denom_pilot_sc = cadre_ent[cadre_ent['pilot'].astype(str).str.strip() == '1'].groupby('Site_Chart')['clean_phone'].nunique().to_dict()
                        else: 
                            denom_pilot_sc = denom_all_sc
                            
                        cadre_surv = surv_df[surv_df[cadre_col_ent].str.contains(cadre_name, case=False, na=False)]
                        if not cadre_surv.empty and cadre_surv['week_num'].max() > 0:
                            num_sc = cadre_surv[cadre_surv['week_num'] > 0].groupby(['Site_Chart', 'Week', 'week_num'])['clean_phone'].nunique().reset_index(name='Responded')
                            num_sc['Denominator'] = num_sc.apply(lambda r: denom_all_sc.get(r['Site_Chart'], 0) if study_phase == "Main Study" else (denom_pilot_sc.get(r['Site_Chart'], 0) if r['week_num'] <= 22 else denom_all_sc.get(r['Site_Chart'], 0)), axis=1)
                            num_sc['Response Rate (%)'] = num_sc.apply(lambda r: get_rate(r['Responded'], r['Denominator']), axis=1)
                            
                            fig_cadre = px.bar(num_sc, x='Site_Chart', y='Response Rate (%)', color='Week', barmode='group', text='Response Rate (%)', category_orders={'Week': ordered_weeks, 'Site_Chart': sorted(surv_df['Site_Chart'].unique())}, labels={'Site_Chart': '', 'Week': ''}, color_discrete_sequence=custom_colors)
                            fig_cadre.update_traces(textposition='outside', texttemplate='%{text:.0f}%')
                            fig_cadre.update_layout(yaxis=dict(range=[0, 120], title="Response Rate (%)" if cadre_name == 'Doctor' else ""), xaxis_title="", margin=dict(t=10, b=0, l=0, r=0), showlegend=False)
                            st.plotly_chart(fig_cadre, use_container_width=True, key=f"rr_chart_{cadre_name}")
                        else:
                            st.info(f"No data for {cadre_name}.")
                            
                st.divider()
                st.subheader("Total Form Submissions by Site & Week")
                fig_total = px.histogram(surv_df[surv_df['week_num'] > 0], x='Site_Chart', color='Week', barmode='group', category_orders={'Week': ordered_weeks, 'Site_Chart': sorted(surv_df['Site_Chart'].unique())}, labels={'Site_Chart': '', 'Week': ''}, color_discrete_sequence=custom_colors)
                fig_total.update_layout(yaxis_title="Forms Submitted (n)", xaxis_title="", margin=dict(t=20, b=0, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
                st.plotly_chart(fig_total, use_container_width=True)
            else:
                st.info("No submission records available to compute response rates.")
        else:
            st.info("Awaiting data from both 'ILI Entities' and 'Surveillance' forms. Please ensure both are successfully fetching.")

    # ==========================================
    # TAB 4: ILI RATE
    # ==========================================
    with tab3:
        if 'Surveillance' in datasets and not datasets['Surveillance'].empty:
            surv_df3 = datasets['Surveillance'].copy()
            
            surv_df3['ext_city'] = get_nested_col(surv_df3, 'pat_city')
            surv_df3['ext_hosp'] = get_nested_col(surv_df3, 'pat_hospital')
            if surv_df3['ext_hosp'].isna().all(): 
                surv_df3['ext_hosp'] = get_nested_col(surv_df3, 'hospital')
                
            surv_df3['ext_name'] = get_nested_col(surv_df3, 'pat_name')
            surv_df3['ext_age'] = get_nested_col(surv_df3, 'pat_age')
            surv_df3['ext_gender'] = get_nested_col(surv_df3, 'pat_gender')
            surv_df3['ext_cadre'] = get_nested_col(surv_df3, 'pat_cadre')
            surv_df3['ext_desig'] = get_nested_col(surv_df3, 'pat_designation')
            surv_df3['ext_dept'] = get_nested_col(surv_df3, 'pat_department')
            surv_df3['ext_ep'] = get_nested_col(surv_df3, 'episode_number')
            surv_df3['ext_eligible'] = get_nested_col(surv_df3, 'eligible_survey')
            
            surv_df3['Site_Chart'] = surv_df3.apply(lambda row: get_phase_global(row['ext_city'], row['ext_hosp'], study_phase), axis=1)
            surv_df3['Site_Name'] = surv_df3['ext_city'].map(CITY_MAP).fillna(surv_df3['ext_city'])
            surv_df3['is_positive'] = surv_df3['ext_eligible'].astype(str).str.strip().str.upper().isin(['TRUE', '1', 'YES'])
            
            date_series = get_nested_col(surv_df3, 'today')
            if date_series.isna().all() and '__system' in surv_df3.columns:
                date_series = surv_df3['__system'].apply(lambda x: x.get('submissionDate') if isinstance(x, dict) else None)

            if date_series is not None and not date_series.isna().all():
                surv_df3['today_dt'] = pd.to_datetime(date_series, errors='coerce').dt.date
                surv_df3 = surv_df3[(surv_df3['today_dt'] >= start_date) & (surv_df3['today_dt'] <= end_date)]
                
                week_col_raw = 'weekofyear'
                if week_col_raw in surv_df3.columns:
                    surv_df3['week_num'] = pd.to_numeric(surv_df3[week_col_raw], errors='coerce').fillna(0).astype(int)
                else:
                    surv_df3['week_num'] = pd.to_numeric(pd.to_datetime(surv_df3['today_dt']).dt.strftime('%U'), errors='coerce').fillna(0).astype(int)
                surv_df3['Week'] = 'Week ' + surv_df3['week_num'].astype(str).str.zfill(2)
            else:
                surv_df3['today_dt'] = None
                surv_df3['week_num'] = 0
                surv_df3['Week'] = 'Unknown'
                
            if selected_sites: 
                surv_df3 = surv_df3[surv_df3['Site_Name'].isin(selected_sites)]
            if selected_cadres: 
                surv_df3 = surv_df3[surv_df3['ext_cadre'].fillna('').str.contains('|'.join(selected_cadres), case=False, na=False)]
                
            st.subheader("Week-wise Influenza Like Illness (ILI)")
            st.caption("Absolute number of submitted forms reporting ILI symptoms per week.")
            
            if not surv_df3.empty and surv_df3['week_num'].max() > 0:
                pos_agg = surv_df3[surv_df3['week_num'] > 0].groupby(['Week', 'week_num', 'Site_Chart']).agg(
                    Total_Responses=('__id', 'count') if '__id' in surv_df3.columns else ('today_dt', 'count'),
                    Positive_Cases=('is_positive', 'sum')
                ).reset_index()
                
                pos_agg['Positivity Rate (%)'] = (pos_agg['Positive_Cases'] / pos_agg['Total_Responses'] * 100).round(1).fillna(0)
                ordered_weeks = sorted(pos_agg['Week'].unique())
                custom_colors = ['#C87550', '#C2982B', '#EAC13E', '#6495ED', '#85B65A', '#2E4B71', '#8B4513', '#2F4F4F']

                fig_pos = px.bar(
                    pos_agg, x='Site_Chart', y='Positive_Cases', color='Week', barmode='group', text='Positive_Cases',
                    category_orders={'Week': ordered_weeks, 'Site_Chart': sorted(surv_df3['Site_Chart'].unique())},
                    labels={'Site_Chart': '', 'Week': ''}, color_discrete_sequence=custom_colors
                )
                y_max = pos_agg['Positive_Cases'].max() * 1.2 if pos_agg['Positive_Cases'].max() > 0 else 10
                fig_pos.update_traces(textposition='outside', texttemplate='%{text:.0f}')
                fig_pos.update_layout(yaxis=dict(range=[0, y_max], title="ILI Positive Cases (n)"), xaxis_title="", margin=dict(t=20, b=0, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
                st.plotly_chart(fig_pos, use_container_width=True)
                
                st.markdown("**ILI Rate Data**")
                pivot_pos = pos_agg.pivot(index='Site_Chart', columns='Week', values='Positivity Rate (%)').fillna(0).round(1)
                pivot_pos = pivot_pos[[w for w in ordered_weeks if w in pivot_pos.columns]].reset_index().rename(columns={'Site_Chart': 'Site / Phase'})
                st.dataframe(pivot_pos.style.format(formatter={col: "{:.1f}%" for col in pivot_pos.columns if col != 'Site / Phase'}).apply(lambda x: ['background-color: rgba(150, 150, 150, 0.1)' if i % 2 == 0 else '' for i in range(len(x))], axis=0).set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]), use_container_width=True, hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Site Wise reported ILI (Incidence)**")
                
                inc_df = surv_df3.groupby('Site_Chart').agg(
                    PERSON_WEEKS=('__id', 'count') if '__id' in surv_df3.columns else ('today_dt', 'count'),
                    ILI_POSITIVE=('is_positive', 'sum')
                ).reset_index()
                
                inc_df['NOT_POSITIVE'] = inc_df['PERSON_WEEKS'] - inc_df['ILI_POSITIVE']
                inc_df['PERSON_YEARS'] = inc_df['PERSON_WEEKS'] / 52.0
                inc_df['INCIDENCE_RATE'] = (inc_df['ILI_POSITIVE'] / inc_df['PERSON_YEARS']) * 1000.0
                
                total_pw = inc_df['PERSON_WEEKS'].sum()
                total_pos = inc_df['ILI_POSITIVE'].sum()
                total_not_pos = inc_df['NOT_POSITIVE'].sum()
                total_py = total_pw / 52.0
                total_ir = (total_pos / total_py) * 1000.0 if total_py > 0 else 0.0
                
                total_row = pd.DataFrame([{'Site_Chart': 'Grand Total', 'NOT_POSITIVE': total_not_pos, 'ILI_POSITIVE': total_pos, 'PERSON_WEEKS': total_pw, 'PERSON_YEARS': total_py, 'INCIDENCE_RATE': total_ir}])
                inc_df = pd.concat([inc_df, total_row], ignore_index=True)
                inc_df = inc_df[['Site_Chart', 'NOT_POSITIVE', 'ILI_POSITIVE', 'PERSON_WEEKS', 'PERSON_YEARS', 'INCIDENCE_RATE']].rename(columns={'Site_Chart': 'Site', 'NOT_POSITIVE': 'NOT POSITIVE', 'ILI_POSITIVE': 'ILI POSITIVE', 'PERSON_WEEKS': 'Person Weeks', 'PERSON_YEARS': 'Person Years', 'INCIDENCE_RATE': 'Incidence Rate (per 1000 person years)'})
                
                def style_inc(df):
                    styled = df.style.format({'Person Years': '{:.2f}', 'Incidence Rate (per 1000 person years)': '{:.2f}'}).apply(lambda x: ['background-color: rgba(150, 150, 150, 0.1)' if i % 2 == 0 else '' for i in range(len(x))], axis=0).set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}])
                    def bold_grand_total(row): 
                        return ['font-weight: bold'] * len(row) if row['Site'] == 'Grand Total' else [''] * len(row)
                    return styled.apply(bold_grand_total, axis=1)

                st.dataframe(style_inc(inc_df), use_container_width=True, hide_index=True)
            else:
                st.info("No data available in the selected date range.")
                
            st.divider()
            st.subheader("Line List: ILI Cases")
            pos_cases = surv_df3[surv_df3['is_positive'] == True].copy()
            
            if not pos_cases.empty:
                pos_cases['Serial No.'] = range(1, len(pos_cases) + 1)
                pos_cases['Age & Sex'] = pos_cases.apply(lambda r: format_age_sex_row(r, 'ext_age', 'ext_gender'), axis=1)
                pos_cases['Eligible on'] = pd.to_datetime(pos_cases['today_dt']).dt.strftime('%d-%b-%Y')
                
                display_df = pos_cases[['Serial No.', 'Site_Chart', 'ext_name', 'Age & Sex', 'ext_cadre', 'ext_desig', 'ext_dept', 'Eligible on', 'ext_ep']].copy()
                display_df.columns = ['Serial No.', 'Site', 'Name', 'Age & Sex', 'Cadre', 'Designation', 'Department', 'Eligible on', 'Episode no.']
                display_df = display_df.fillna('')
                
                st.dataframe(
                    display_df.style.apply(lambda x: ['background-color: rgba(150, 150, 150, 0.1)' if i % 2 == 0 else '' for i in range(len(x))], axis=0).set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]),
                    use_container_width=True, hide_index=True,
                    column_config={"Serial No.": st.column_config.NumberColumn("S.No.", width="small")}
                )
            else:
                st.success("No ILI positive cases reported in the selected date range and filters.")
        else:
            st.info("Awaiting data from 'Surveillance' form. Please ensure it is successfully fetching.")

    # ==========================================
    # TAB 5: SAMPLING & Flu/COVID Positivity
    # ==========================================
    with tab4:
        if 'Sampling' in datasets and not datasets['Sampling'].empty:
            if 'pos_cases' in locals() and not pos_cases.empty:
                samp_df = datasets['Sampling'].copy()
                pos_cases_samp = pos_cases.copy()

                pos_cases_samp['surv_phone'] = get_nested_col(pos_cases_samp, 'pat_phone')
                if pos_cases_samp['surv_phone'].isna().all(): 
                    pos_cases_samp['surv_phone'] = get_nested_col(pos_cases_samp, 'phone_no')
                pos_cases_samp['clean_phone'] = clean_phone_series(pos_cases_samp['surv_phone'])
                pos_cases_samp['clean_episode'] = clean_episode_series(pos_cases_samp['ext_ep'])

                missing_ep_count = (pos_cases_samp['clean_episode'] == '').sum()
                if missing_ep_count > 0:
                    st.warning(f"⚠️ {missing_ep_count} ILI-positive record(s) are missing an episode number in the Surveillance form and cannot be linked to a Sampling form. They will show as 'Not Filled' below.")

                samp_df['samp_phone'] = get_any_col(samp_df, 'phone_no', 'sampling')
                samp_df['clean_phone'] = clean_phone_series(samp_df['samp_phone'])
                samp_df['samp_episode_raw'] = get_nested_col(samp_df, 'pat_episode')
                samp_df['clean_episode'] = clean_episode_series(samp_df['samp_episode_raw'])
                samp_df['samp_barcode_raw'] = get_any_col(samp_df, 'sampling_barcode', 'sampling')
                samp_df['clean_barcode'] = clean_barcode_series(samp_df['samp_barcode_raw'])

                samp_df['fever_date'] = pd.to_datetime(get_any_col(samp_df, 'sampling-q2', 'sampling'), errors='coerce').dt.normalize()
                samp_df['cough_date'] = pd.to_datetime(get_any_col(samp_df, 'sampling-q3', 'sampling'), errors='coerce').dt.normalize()
                samp_df['fill_date'] = pd.to_datetime(get_any_col(samp_df, 'sampling-q4_1', 'sampling'), errors='coerce').combine_first(
                                       pd.to_datetime(get_any_col(samp_df, 'today', 'sampling'), errors='coerce')).dt.normalize()
                samp_df['q4_status'] = get_any_col(samp_df, 'sampling-q4', 'sampling')

                samp_df['q5_raw'] = get_any_col(samp_df, 'sampling-q5', 'sampling').astype(str).str.replace(r'\.0$', '', regex=True)
                
                def get_outside_res(val, target_code):
                    if pd.isna(val) or val in ['nan', 'None', '']: 
                        return pd.NA
                    vals = str(val).split()
                    if target_code in vals: 
                        return 'POSITIVE'
                    if '5' in vals: 
                        return 'NEGATIVE'
                    if any(c in vals for c in ['1','2','3','4']): 
                        return 'NEGATIVE'
                    return pd.NA
                    
                samp_df['Outside_Inf_A'] = samp_df['q5_raw'].apply(lambda x: get_outside_res(x, '1'))
                samp_df['Outside_Inf_B'] = samp_df['q5_raw'].apply(lambda x: get_outside_res(x, '2'))
                samp_df['Outside_SARS'] = samp_df['q5_raw'].apply(lambda x: get_outside_res(x, '3'))

                samp_df['fever_diff'] = (samp_df['fill_date'] - samp_df['fever_date']).dt.days
                samp_df['cough_diff'] = (samp_df['fill_date'] - samp_df['cough_date']).dt.days
                fever_valid = samp_df['fever_diff'].isna() | (samp_df['fever_diff'] <= 10)
                cough_valid = samp_df['cough_diff'].isna() | (samp_df['cough_diff'] <= 10)
                samp_df['is_samp_eligible'] = fever_valid & cough_valid & (samp_df['fever_diff'].notna() | samp_df['cough_diff'].notna())

                q4_map = {'1': 'Yes, sampling done', '2': 'RTPCR done outside the project', '3': 'Refused to provide the sample', '4': 'Healthcare worker on leave for next 7 days or more', '5': 'HCW not available due to other reasons', '1.0': 'Yes, sampling done', '2.0': 'RTPCR done outside the project', '3.0': 'Refused to provide the sample', '4.0': 'Healthcare worker on leave for next 7 days or more', '5.0': 'HCW not available due to other reasons'}
                samp_df['Sampling Status'] = samp_df['q4_status'].astype(str).str.strip().map(q4_map).fillna('Status Unknown/Other')

                samp_df_valid = samp_df[(samp_df['clean_phone'] != '') & (samp_df['clean_episode'] != '')].copy()
                samp_df_unique = samp_df_valid.sort_values('fill_date', ascending=False).drop_duplicates(['clean_phone', 'clean_episode'])

                merged_samp = pd.merge(
                    pos_cases_samp,
                    samp_df_unique[['clean_phone', 'clean_episode', 'Sampling Status', 'is_samp_eligible', 'fill_date', 'clean_barcode', 'Outside_Inf_A', 'Outside_Inf_B', 'Outside_SARS']],
                    on=['clean_phone', 'clean_episode'],
                    how='left'
                )

                if 'Report' in datasets and not datasets['Report'].empty:
                    rep_df = datasets['Report'].copy()
                    rep_df['rep_barcode_raw'] = get_any_col(rep_df, 'reporting_barcode', 'barcode')
                    rep_df['clean_barcode'] = clean_barcode_series(rep_df['rep_barcode_raw'])

                    res_map = {'1': 'POSITIVE', '2': 'NEGATIVE', '1.0': 'POSITIVE', '2.0': 'NEGATIVE'}
                    sample_type_map = {'1': 'Nasal Swab', '2': 'Throat swab', '3': 'BAL', '4': 'Endotracheal aspirate', '5': 'Combined NP & Throat', '1.0': 'Nasal Swab', '2.0': 'Throat swab', '3.0': 'BAL', '4.0': 'Endotracheal aspirate', '5.0': 'Combined NP & Throat'}

                    rep_df['Sample Type'] = get_any_col(rep_df, 'q3', 'reporting').astype(str).str.strip().map(sample_type_map).fillna('Not Recorded')
                    rep_df['Date of Sampling'] = pd.to_datetime(get_any_col(rep_df, 'q4', 'reporting'), errors='coerce').dt.strftime('%d-%b-%Y')
                    rep_df['Influenza A'] = get_any_col(rep_df, 'q5', 'reporting').astype(str).str.strip().map(res_map).fillna('Pending/Unknown')
                    rep_df['Influenza B'] = get_any_col(rep_df, 'q6', 'reporting').astype(str).str.strip().map(res_map).fillna('Pending/Unknown')
                    rep_df['SARS CoV2'] = get_any_col(rep_df, 'q7', 'reporting').astype(str).str.strip().map(res_map).fillna('Pending/Unknown')

                    sort_col = 'today' if 'today' in rep_df.columns else 'Date of Sampling'
                    rep_df_valid = rep_df[(rep_df['clean_barcode'] != '')].copy()
                    rep_df_unique = rep_df_valid.sort_values(sort_col, ascending=False).drop_duplicates('clean_barcode') if sort_col in rep_df_valid.columns else rep_df_valid.drop_duplicates('clean_barcode')

                    merged_samp = pd.merge(merged_samp, rep_df_unique[['clean_barcode', 'Sample Type', 'Date of Sampling', 'Influenza A', 'Influenza B', 'SARS CoV2']], on='clean_barcode', how='left')
                else:
                    if FORM_ID_REPORT:
                        st.warning(f"Attempted to fetch RTPCR data using Form ID '{FORM_ID_REPORT}', but no records were found or the fetch failed.")
                    else:
                        st.warning("FORM_ID_REPORT secret is missing or empty.")

                for col in ['Sample Type', 'Date of Sampling', 'Influenza A', 'Influenza B', 'SARS CoV2']:
                    if col not in merged_samp.columns:
                        merged_samp[col] = pd.NA

                def get_final_status(row):
                    if pd.isna(row['Sampling Status']):
                        return 'Not Filled'
                    if row['is_samp_eligible'] == False:
                        return 'Not sampled within 10 days of onset'
                    return row['Sampling Status']

                merged_samp['Final Status'] = merged_samp.apply(get_final_status, axis=1)

                def resolve_res(row, virus_col, outside_col):
                    if row['Final Status'] == 'RTPCR done outside the project':
                        return row[outside_col] if pd.notna(row[outside_col]) else 'No Report'
                    return row[virus_col] if pd.notna(row[virus_col]) else 'No Report'

                merged_samp['Influenza A'] = merged_samp.apply(lambda r: resolve_res(r, 'Influenza A', 'Outside_Inf_A'), axis=1)
                merged_samp['Influenza B'] = merged_samp.apply(lambda r: resolve_res(r, 'Influenza B', 'Outside_Inf_B'), axis=1)
                merged_samp['SARS CoV2'] = merged_samp.apply(lambda r: resolve_res(r, 'SARS CoV2', 'Outside_SARS'), axis=1)

                def resolve_sample_type(row):
                    if row['Final Status'] == 'RTPCR done outside the project':
                        return 'Outside Report' if pd.notna(row['Outside_Inf_A']) else 'No Report'
                    return row['Sample Type'] if pd.notna(row['Sample Type']) else 'No Report'

                def resolve_sample_date(row):
                    if row['Final Status'] == 'RTPCR done outside the project':
                        return row['fill_date'].strftime('%d-%b-%Y') if pd.notna(row['fill_date']) else '-'
                    return row['Date of Sampling'] if pd.notna(row['Date of Sampling']) else '-'

                merged_samp['Sample Type'] = merged_samp.apply(resolve_sample_type, axis=1)
                merged_samp['Date of Sampling'] = merged_samp.apply(resolve_sample_date, axis=1)

                total_ili = len(merged_samp)
                forms_filled = merged_samp['Sampling Status'].notna().sum()
                sampling_done_count = merged_samp['Final Status'].isin(['Yes, sampling done', 'RTPCR done outside the project']).sum()
                reports_entered_count = (merged_samp['Sample Type'] != 'No Report').sum()

                def get_pct_str(num, den):
                    if den > 0:
                        return f"{num} ({(num/den)*100:.1f}%)"
                    return str(num)

                st.subheader("Sampling Form Overview")
                col1, col2 = st.columns(2)
                with col1: 
                    st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;'><h4>Total ILI Positive Cases</h4><h1 style='color:#2E4B71;margin:0;'>{total_ili}</h1></div>", unsafe_allow_html=True)
                with col2: 
                    st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;'><h4>Sampling Forms Filled</h4><h1 style='color:#85B65A;margin:0;'>{get_pct_str(forms_filled, total_ili)}</h1></div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                col4, col5 = st.columns(2)
                with col4: 
                    st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;'><h4>Sampling Done</h4><h1 style='color:#6495ED;margin:0;'>{get_pct_str(sampling_done_count, forms_filled)}</h1></div>", unsafe_allow_html=True)
                with col5: 
                    st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;'><h4>Reports Entered</h4><h1 style='color:#9b59b6;margin:0;'>{get_pct_str(reports_entered_count, sampling_done_count)}</h1></div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Sampling Status Distribution**")
                status_counts = merged_samp['Final Status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']

                fig_status = px.bar(status_counts, x='Count', y='Status', orientation='h', text='Count', color='Status', color_discrete_sequence=px.colors.qualitative.Safe)
                fig_status.update_traces(textposition='outside', texttemplate='%{text}')
                fig_status.update_layout(showlegend=False, xaxis_title="Number of Cases", yaxis_title="", margin=dict(t=10, b=0, l=0, r=0), height=300)
                st.plotly_chart(fig_status, use_container_width=True)
                
                st.divider()
                st.subheader("RTPCR Report Overview")

                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1: 
                    st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #EAC13E;'><h4>Influenza A Positive</h4><h1 style='color:#EAC13E;margin:0;'>{(merged_samp['Influenza A'] == 'POSITIVE').sum()}</h1></div>", unsafe_allow_html=True)
                with r_col2: 
                    st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #C87550;'><h4>Influenza B Positive</h4><h1 style='color:#C87550;margin:0;'>{(merged_samp['Influenza B'] == 'POSITIVE').sum()}</h1></div>", unsafe_allow_html=True)
                with r_col3: 
                    st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #2E4B71;'><h4>SARS CoV2 Positive</h4><h1 style='color:#2E4B71;margin:0;'>{(merged_samp['SARS CoV2'] == 'POSITIVE').sum()}</h1></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### Weekly Viral Positivity by Site")
                
                if 'Week' not in merged_samp.columns: 
                    merged_samp['Week'] = 'Unknown'
                virus_cols = ['Influenza A', 'Influenza B', 'SARS CoV2']
                df_melted = merged_samp[['Week', 'Site_Chart'] + virus_cols].copy().melt(id_vars=['Week', 'Site_Chart'], value_vars=virus_cols, var_name='Virus', value_name='Result')
                df_positives = df_melted[df_melted['Result'] == 'POSITIVE']
                
                if not df_positives.empty:
                    agg_virus = df_positives.groupby(['Week', 'Site_Chart', 'Virus']).size().reset_index(name='Count')
                    idx = pd.MultiIndex.from_product([sorted(df_positives['Site_Chart'].unique()), sorted(df_positives['Week'].unique()), virus_cols], names=['Site_Chart', 'Week', 'Virus'])
                    agg_virus_complete = agg_virus.set_index(['Site_Chart', 'Week', 'Virus']).reindex(idx, fill_value=0).reset_index()
                    
                    fig_viral = go.Figure()
                    virus_colors = {'Influenza A': '#EAC13E', 'Influenza B': '#C87550', 'SARS CoV2': '#2E4B71'}
                    for virus in virus_cols:
                        v_data = agg_virus_complete[agg_virus_complete['Virus'] == virus]
                        fig_viral.add_trace(go.Bar(name=virus, x=[v_data['Site_Chart'], v_data['Week']], y=v_data['Count'], marker_color=virus_colors[virus], hovertemplate="Site: %{x[0]}<br>Week: %{x[1]}<br>%{fullData.name}: %{y}<extra></extra>"))
                    fig_viral.update_layout(barmode='stack', yaxis_title="Flu Positives (n)", xaxis_title="", yaxis=dict(tickformat="d", dtick=1), margin=dict(t=20, b=0, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, title=""))
                    st.plotly_chart(fig_viral, use_container_width=True)
                else:
                    st.info("No positive RTPCR reports available in the selected date range to generate the chart.")

                st.divider()
                st.subheader("Sampling & Reports Line List")
                merged_samp['Sampling Form Filled'] = merged_samp['Sampling Status'].notna().map({True: 'Yes', False: 'No'})
                display_samp = merged_samp[['Serial No.', 'Site_Chart', 'ext_name', 'Age & Sex', 'ext_cadre', 'ext_desig', 'ext_dept', 'Eligible on', 'ext_ep', 'Sampling Form Filled', 'Final Status', 'clean_barcode', 'Sample Type', 'Date of Sampling', 'Influenza A', 'Influenza B', 'SARS CoV2']].copy()
                display_samp.columns = ['Serial No.', 'Site', 'Name', 'Age & Sex', 'Cadre', 'Designation', 'Department', 'Eligible on', 'Episode no.', 'Form Filled?', 'Sampling Status', 'Barcode', 'Sample Type', 'Sampling Date', 'Inf A', 'Inf B', 'SARS-CoV-2']
                
                st.dataframe(style_table(display_samp.fillna('')), use_container_width=True, hide_index=True, column_config={"Serial No.": st.column_config.NumberColumn("S.No.", width="small")})
            else:
                st.info("No ILI positive cases reported in the selected date range. No sampling required.")
        else:
            st.info("Awaiting data from 'Sampling' form. Please ensure it is successfully fetching.")

    # ==========================================
    # TAB 6: OUTCOME OF ILLNESS
    # ==========================================
    with tab6:
        st.subheader("Outcome of Illness Overview")
        if 'Eligible Entities' in datasets and not datasets['Eligible Entities'].empty:
            elig_df = datasets['Eligible Entities'].copy()

            def format_mean_sd(mean_val, sd_val):
                return f"{mean_val:.1f} ± {sd_val:.1f}" if pd.notna(mean_val) and pd.notna(sd_val) else "0.0 ± 0.0"

            elig_df['ext_phone'] = get_nested_col(elig_df, 'phone_no')
            if elig_df['ext_phone'].isna().all(): 
                elig_df['ext_phone'] = get_nested_col(elig_df, 'phone_number')
            if elig_df['ext_phone'].isna().all(): 
                elig_df['ext_phone'] = get_nested_col(elig_df, 'name')

            elig_df['clean_phone'] = clean_phone_series(elig_df['ext_phone'])
            elig_df['ext_episode_raw'] = get_nested_col(elig_df, 'episode_no')
            elig_df['clean_episode'] = clean_episode_series(elig_df['ext_episode_raw'])

            elig_city = get_nested_col(elig_df, 'city')
            elig_df['Site'] = elig_city.map(CITY_MAP).fillna(elig_city).fillna('Unknown Site')

            if selected_sites: 
                elig_df = elig_df[elig_df['Site'].isin(selected_sites)]

            cadre_col_elig_val = get_nested_col(elig_df, 'pat_cadre_status')
            if cadre_col_elig_val.isna().all(): 
                cadre_col_elig_val = get_nested_col(elig_df, 'cadre')
            elig_df['Cadre_Filter'] = cadre_col_elig_val.fillna('Unknown')

            if selected_cadres: 
                elig_df = elig_df[elig_df['Cadre_Filter'].str.contains('|'.join(selected_cadres), case=False, na=False)]

            elig_date_col_val = get_nested_col(elig_df, 'eligible_on')
            if not elig_date_col_val.isna().all():
                elig_df['eligible_on_dt'] = pd.to_datetime(elig_date_col_val, dayfirst=True, errors='coerce')
                elig_df['outcome_due_dt'] = elig_df['eligible_on_dt'] + pd.Timedelta(days=14)
                elig_df = elig_df[elig_df['eligible_on_dt'].notna() & (elig_df['eligible_on_dt'].dt.date >= start_date) & (elig_df['eligible_on_dt'].dt.date <= end_date)]
            else:
                st.warning("Variable 'eligible_on' not found in the Eligible Entities list. Showing all available records.")
                elig_df['outcome_due_dt'] = pd.NaT

            current_date = datetime.date.today()
            total_outcomes_due = len(elig_df[elig_df['outcome_due_dt'].dt.date <= current_date])

            missing_ep_count = (elig_df['clean_episode'] == '').sum()
            if missing_ep_count > 0:
                st.warning(f"⚠️ {missing_ep_count} due outcome record(s) are missing an episode number ('episode_no').")

            out_df = datasets.get('Outcome', pd.DataFrame()).copy()
            total_outcomes_filled = 0

            if not out_df.empty:
                out_df['out_phone'] = get_nested_col(out_df, 'phone_no')
                if out_df['out_phone'].isna().all(): 
                    out_df['out_phone'] = get_nested_col(out_df, 'pat_phone')
                out_df['clean_phone'] = clean_phone_series(out_df['out_phone'])

                out_df['out_episode_raw'] = get_nested_col(out_df, 'pat_episode')
                if out_df['out_episode_raw'].isna().all(): 
                    out_df['out_episode_raw'] = get_nested_col(out_df, 'episode_no')
                if out_df['out_episode_raw'].isna().all(): 
                    out_df['out_episode_raw'] = get_nested_col(out_df, 'episode_number')
                out_df['clean_episode'] = clean_episode_series(out_df['out_episode_raw'])

                q1_2_map = {'1': 'Alive & recovered', '2': 'Still recovering', '3': 'Died', '4': 'Refused to share', '5': 'Could not be contacted', '1.0': 'Alive & recovered', '2.0': 'Still recovering', '3.0': 'Died', '4.0': 'Refused to share', '5.0': 'Could not be contacted'}
                q2_map = {'1': 'Resolved', '2': 'Ongoing', '1.0': 'Resolved', '2.0': 'Ongoing'}
                yn_map = {'1': 'Yes', '2': 'No', '1.0': 'Yes', '2.0': 'No'}

                out_df['Status (q1_2)'] = get_nested_col(out_df, 'q1_2').astype(str).str.strip().map(q1_2_map).fillna('Not Recorded')
                out_df['Fever (q2_1)'] = get_nested_col(out_df, 'q2_1').astype(str).str.strip().map(q2_map).fillna('Not Recorded')
                out_df['Cough (q2_2)'] = get_nested_col(out_df, 'q2_2').astype(str).str.strip().map(q2_map).fillna('Not Recorded')
                out_df['Leave Taken (q3_1)'] = get_nested_col(out_df, 'q3_1').astype(str).str.strip().map(yn_map).fillna('Not Recorded')
                out_df['Worked Ill (q3_2)'] = get_nested_col(out_df, 'q3_2').astype(str).str.strip().map(yn_map).fillna('Not Recorded')
                out_df['Hosp (q4_1)'] = get_nested_col(out_df, 'q4_1').astype(str).str.strip().map(yn_map).fillna('Not Recorded')
                out_df['ICU (q4_1_b)'] = get_nested_col(out_df, 'q4_1_b').astype(str).str.strip().map(yn_map).fillna('Not Recorded')
                out_df['Antibiotic (q4_1_d)'] = get_nested_col(out_df, 'q4_1_d').astype(str).str.strip().map(yn_map).fillna('Not Recorded')
                out_df['Antiviral (q4_1_e)'] = get_nested_col(out_df, 'q4_1_e').astype(str).str.strip().map(yn_map).fillna('Not Recorded')

                out_df['q3_1_a_raw'] = pd.to_numeric(get_nested_col(out_df, 'q3_1_a'), errors='coerce')
                out_df['q3_2_a_raw'] = pd.to_numeric(get_nested_col(out_df, 'q3_2_a'), errors='coerce')
                out_df['q3_2_b_raw'] = get_nested_col(out_df, 'q3_2_b')
                out_df['q4_1_a_raw'] = pd.to_numeric(get_nested_col(out_df, 'q4_1_a'), errors='coerce')
                out_df['q4_1_c_raw'] = pd.to_numeric(get_nested_col(out_df, 'q4_1_c'), errors='coerce')

                sort_col_out = 'today' if 'today' in out_df.columns else None
                out_df_sorted = out_df.sort_values(sort_col_out, ascending=False) if sort_col_out else out_df
                out_df_unique = out_df_sorted[(out_df_sorted['clean_phone'] != '') & (out_df_sorted['clean_episode'] != '')].drop_duplicates(['clean_phone', 'clean_episode'], keep='first')

                merge_cols = ['clean_phone', 'clean_episode', 'Status (q1_2)', 'Fever (q2_1)', 'Cough (q2_2)', 'Leave Taken (q3_1)', 'q3_1_a_raw', 'Worked Ill (q3_2)', 'q3_2_a_raw', 'q3_2_b_raw', 'Hosp (q4_1)', 'q4_1_a_raw', 'ICU (q4_1_b)', 'q4_1_c_raw', 'Antibiotic (q4_1_d)', 'Antiviral (q4_1_e)']
                merged_outcome = pd.merge(elig_df, out_df_unique[[c for c in merge_cols if c in out_df_unique.columns]], on=['clean_phone', 'clean_episode'], how='left')
                merged_outcome = merged_outcome[merged_outcome['outcome_due_dt'].dt.date <= datetime.date.today()]
                total_outcomes_filled = merged_outcome['Status (q1_2)'].notna().sum()
            else:
                merged_outcome = elig_df.copy()
                merged_outcome['Status (q1_2)'] = pd.NA
                merged_outcome['Fever (q2_1)'] = pd.NA
                merged_outcome['Cough (q2_2)'] = pd.NA
                st.info("No 'Outcome of Illness' forms have been synced yet.")

            col1, col2 = st.columns(2)
            with col1: 
                st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #C87550;'><h4>Total Outcomes Due</h4><h1 style='color:#C87550;margin:0;'>{total_outcomes_due}</h1></div>", unsafe_allow_html=True)
            with col2: 
                st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #85B65A;'><h4>Total Outcomes Filled</h4><h1 style='color:#85B65A;margin:0;'>{total_outcomes_filled} ({(total_outcomes_filled/total_outcomes_due)*100 if total_outcomes_due>0 else 0:.1f}%)</h1></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Outcome of Illness Line List")
            if not merged_outcome.empty:
                merged_outcome['Name'] = get_nested_col(merged_outcome, 'pat_name')
                merged_outcome['Age & Sex'] = merged_outcome.apply(lambda r: format_age_sex_row(r, get_nested_col(merged_outcome, 'age').name if get_nested_col(merged_outcome, 'age').name else 'age', get_nested_col(merged_outcome, 'gender').name if get_nested_col(merged_outcome, 'gender').name else 'gender'), axis=1)
                merged_outcome['Cadre'] = merged_outcome['Cadre_Filter']
                merged_outcome['Designation'] = get_nested_col(merged_outcome, 'designation')
                merged_outcome['Episode No.'] = get_nested_col(merged_outcome, 'episode_no')

                if 'eligible_on_dt' in merged_outcome.columns:
                    merged_outcome['Eligible On'] = pd.to_datetime(merged_outcome['eligible_on_dt']).dt.strftime('%d-%b-%Y').fillna('Unknown')
                    merged_outcome['Outcome Due'] = pd.to_datetime(merged_outcome['outcome_due_dt']).dt.strftime('%d-%b-%Y').fillna('Unknown')
                else:
                    merged_outcome['Eligible On'] = 'Unknown'
                    merged_outcome['Outcome Due'] = 'Unknown'

                merged_outcome['Status (q1_2)'] = merged_outcome['Status (q1_2)'].fillna('Form Not Filled')
                for c in ['Fever (q2_1)', 'Cough (q2_2)', 'Leave Taken (q3_1)', 'Worked Ill (q3_2)']: 
                    merged_outcome[c] = merged_outcome.get(c, pd.Series()).fillna('')

                display_cols = ['Name', 'Age & Sex', 'Designation', 'Cadre', 'Site', 'Episode No.', 'Eligible On', 'Outcome Due', 'Status (q1_2)', 'Fever (q2_1)', 'Cough (q2_2)', 'Leave Taken (q3_1)', 'Worked Ill (q3_2)']
                show_df = merged_outcome[[c for c in display_cols if c in merged_outcome.columns]].copy()
                show_df.insert(0, 'S.No.', range(1, len(show_df) + 1))
                
                st.dataframe(style_table(show_df.fillna('')), use_container_width=True, hide_index=True)

                st.divider()
                if total_outcomes_filled > 0:
                    df_filled = merged_outcome[merged_outcome['Status (q1_2)'] != 'Form Not Filled'].copy()
                    yn_color_map = {'Yes': '#2E4B71', 'No': '#E0E0E0', 'Not Recorded': '#f0f2f6'}

                    st.markdown("### Absenteeism")
                    colA, colB, colC = st.columns(3)
                    
                    abs_counts = df_filled['Leave Taken (q3_1)'].value_counts().reset_index()
                    abs_counts.columns = ['Response', 'Count']
                    fig_abs = px.pie(abs_counts, names='Response', values='Count', hole=0.5, title="Leave Taken while Ill", color='Response', color_discrete_map=yn_color_map)
                    fig_abs.update_traces(sort=False)
                    fig_abs.update_layout(margin=dict(t=30, b=10, l=10, r=10))
                    colA.plotly_chart(fig_abs, use_container_width=True)

                    total_abs_days = df_filled['q3_1_a_raw'].sum()
                    with colB: 
                        st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #EAC13E;height:100%;'><h4>Cumulative Days of Leave</h4><h1 style='color:#EAC13E;margin:0;'>{total_abs_days:.0f}</h1></div>", unsafe_allow_html=True)
                    with colC: 
                        st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #C87550;height:100%;'><h4>Mean ± SD Days of Leave (If on leave)</h4><h1 style='color:#C87550;margin:0;'>{format_mean_sd(df_filled['q3_1_a_raw'].mean(), df_filled['q3_1_a_raw'].std())}</h1></div>", unsafe_allow_html=True)

                    st.markdown("**Absenteeism Details**")
                    abs_df = df_filled[['Site', 'Name', 'Age & Sex', 'Designation', 'Cadre', 'Leave Taken (q3_1)', 'q3_1_a_raw']].copy()
                    abs_df.rename(columns={'q3_1_a_raw': 'Days of Leave'}, inplace=True)
                    abs_df['Days of Leave'] = pd.to_numeric(abs_df['Days of Leave'], errors='coerce').astype('Int64').astype(str).replace('<NA>', '')
                    st.dataframe(style_table(abs_df.fillna('')), use_container_width=True, hide_index=True)
                    
                    st.divider()

                    st.markdown("### Presenteeism")
                    colA, colB, colC = st.columns(3)
                    pres_counts = df_filled['Worked Ill (q3_2)'].value_counts().reset_index()
                    pres_counts.columns = ['Response', 'Count']
                    fig_pres = px.pie(pres_counts, names='Response', values='Count', hole=0.5, title="Worked while Ill", color='Response', color_discrete_map=yn_color_map)
                    fig_pres.update_traces(sort=False)
                    fig_pres.update_layout(margin=dict(t=30, b=10, l=10, r=10))
                    colA.plotly_chart(fig_pres, use_container_width=True)

                    total_pres_days = df_filled['q3_2_a_raw'].sum()
                    with colB: 
                        st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #228B22;height:100%;'><h4>Cumulative days worked while ill</h4><h1 style='color:#228B22;margin:0;'>{total_pres_days:.0f}</h1></div>", unsafe_allow_html=True)
                    with colC: 
                        st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #228B22;height:100%;'><h4>Days worked with illness: Mean ± SD Days</h4><h1 style='color:#228B22;margin:0;'>{format_mean_sd(df_filled['q3_2_a_raw'].mean(), df_filled['q3_2_a_raw'].std())}</h1></div>", unsafe_allow_html=True)

                    st.markdown("**Presenteeism Details**")
                    pres_df = df_filled[['Site', 'Name', 'Age & Sex', 'Designation', 'Cadre', 'Worked Ill (q3_2)', 'q3_2_a_raw']].copy()
                    pres_df.rename(columns={'q3_2_a_raw': 'Days Worked Ill'}, inplace=True)
                    pres_df['Days Worked Ill'] = pd.to_numeric(pres_df['Days Worked Ill'], errors='coerce').astype('Int64').astype(str).replace('<NA>', '')
                    st.dataframe(style_table(pres_df.fillna('')), use_container_width=True, hide_index=True)

                    st.markdown("#### PPE use while working with illness")
                    df_ppe = df_filled[df_filled['Worked Ill (q3_2)'] == 'Yes']
                    denom_ppe = len(df_ppe)
                    if denom_ppe > 0:
                        ppe_map = {'1': 'Mask', '2': 'Gloves', '3': 'Eye protection', '4': 'Sanitizer', '5': 'Frequent hand wash', '9': 'None'}
                        ppe_cols = st.columns(6)
                        for idx, (key, label) in enumerate(ppe_map.items()):
                            used_count = df_ppe['q3_2_b_raw'].apply(lambda x: key in str(x).split()).sum()
                            fig_ppe = px.pie(names=['Used', 'Not Used'], values=[used_count, denom_ppe - used_count], hole=0.7, color=['Used', 'Not Used'], color_discrete_map={'Used': '#2E4B71', 'Not Used': '#E0E0E0'})
                            fig_ppe.update_traces(textinfo='none', hoverinfo='label+value', sort=False)
                            fig_ppe.update_layout(showlegend=False, margin=dict(t=20, b=20, l=10, r=10), height=150, annotations=[dict(text=f"<b>{used_count}/{denom_ppe}</b>", x=0.5, y=0.5, font_size=16, showarrow=False)])
                            with ppe_cols[idx]:
                                st.markdown(f"<div style='text-align:center;font-weight:bold;font-size:0.9rem;'>{label}</div>", unsafe_allow_html=True)
                                st.plotly_chart(fig_ppe, use_container_width=True)
                    else:
                        st.info("No participants reported working while ill.")
                    
                    st.divider()
                    st.markdown("### Clinical Course & Treatment")
                    c_colA, c_colB, c_colC, c_colD = st.columns(4)
                    
                    fig_hosp = px.pie(df_filled['Hosp (q4_1)'].value_counts().reset_index(name='Count').rename(columns={'Hosp (q4_1)':'Response'}), names='Response', values='Count', hole=0.5, title="Hospitalized?", color='Response', color_discrete_map=yn_color_map)
                    fig_hosp.update_traces(sort=False)
                    fig_hosp.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=300)
                    c_colA.plotly_chart(fig_hosp, use_container_width=True)
                    
                    fig_icu = px.pie(df_filled['ICU (q4_1_b)'].value_counts().reset_index(name='Count').rename(columns={'ICU (q4_1_b)':'Response'}), names='Response', values='Count', hole=0.5, title="ICU Admission?", color='Response', color_discrete_map=yn_color_map)
                    fig_icu.update_traces(sort=False)
                    fig_icu.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=300)
                    c_colB.plotly_chart(fig_icu, use_container_width=True)

                    with c_colC: 
                        st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #C87550;height:100%;'><h4>Cumulative Days of</h4><hr style='margin:10px 0;'><p style='margin:0;font-size:0.9rem'>Hospitalization</p><h2 style='color:#C87550;margin:0;'>{df_filled['q4_1_a_raw'].sum():.0f}</h2><p style='margin:10px 0 0 0;font-size:0.9rem'>ICU Admission</p><h2 style='color:#C87550;margin:0;'>{df_filled['q4_1_c_raw'].sum():.0f}</h2></div>", unsafe_allow_html=True)
                    with c_colD: 
                        st.markdown(f"<div style='background-color:rgba(150, 150, 150, 0.1);padding:15px;border-radius:10px;text-align:center;border-top: 4px solid #2E4B71;height:100%;'><h4>Mean ± SD Days</h4><hr style='margin:10px 0;'><p style='margin:0;font-size:0.9rem'>Hospitalization</p><h2 style='color:#2E4B71;margin:0;'>{format_mean_sd(df_filled['q4_1_a_raw'].mean(), df_filled['q4_1_a_raw'].std())}</h2><p style='margin:10px 0 0 0;font-size:0.9rem'>ICU Admission</p><h2 style='color:#2E4B71;margin:0;'>{format_mean_sd(df_filled['q4_1_c_raw'].mean(), df_filled['q4_1_c_raw'].std())}</h2></div>", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    med_colA, med_colB = st.columns(2)
                    
                    fig_anti_v = px.pie(df_filled['Antiviral (q4_1_e)'].value_counts().reset_index(name='Count').rename(columns={'Antiviral (q4_1_e)':'Response'}), names='Response', values='Count', hole=0.5, title="Antiviral Use", color='Response', color_discrete_map=yn_color_map)
                    fig_anti_v.update_traces(sort=False)
                    fig_anti_v.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=400)
                    med_colA.plotly_chart(fig_anti_v, use_container_width=True)
                    
                    fig_anti_b = px.pie(df_filled['Antibiotic (q4_1_d)'].value_counts().reset_index(name='Count').rename(columns={'Antibiotic (q4_1_d)':'Response'}), names='Response', values='Count', hole=0.5, title="Antibiotic Use", color='Response', color_discrete_map=yn_color_map)
                    fig_anti_b.update_traces(sort=False)
                    fig_anti_b.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=400)
                    med_colB.plotly_chart(fig_anti_b, use_container_width=True)

            else:
                st.success("No participants found who were ILI Positive and are due for Outcome in the selected date range")

    # ==========================================
    # TAB 7: NON RESPONDENTS
    # ==========================================
    with tab7:
        st.subheader("Non-Respondents Overview")
        st.caption("Active participants who were due for surveillance but whose forms are missing for the selected weeks.")
        st.warning("📅**Please select Last 4 weeks in sidebar to get the list of the most recent non-responders**")
        
        if 'ILI Entities' in datasets and 'Surveillance' in datasets and not datasets['Surveillance'].empty and not datasets['ILI Entities'].empty:
            surv_df7 = datasets['Surveillance'].copy()
            ent_df7 = datasets['ILI Entities'].copy()

            phone_to_pid = {}
            pid_to_all_phones = {}
            if 'Participant IDs' in datasets and not datasets['Participant IDs'].empty:
                part_id_df = datasets['Participant IDs'].copy()
                part_id_df['pid_label'] = get_nested_col(part_id_df, 'label')
                part_id_df['pid_phone'] = clean_phone_series(get_nested_col(part_id_df, 'phone_number'))
                valid_pids = part_id_df[(part_id_df['pid_phone'] != '') & (part_id_df['pid_label'].notna())]
                phone_to_pid = dict(zip(valid_pids['pid_phone'], valid_pids['pid_label']))
                pid_to_all_phones = valid_pids.groupby('pid_label')['pid_phone'].apply(lambda x: ', '.join(x.unique())).to_dict()
            else:
                st.warning("⚠️ 'participant_ids' entity list is missing or empty. The dashboard will fall back to using phone numbers as the unique ID.")

            if 'excluded' in ent_df7.columns:
                ent_df7 = ent_df7[ent_df7['excluded'].astype(str).str.strip() != '1']
                
            ent_df7['ext_phone'] = get_nested_col(ent_df7, 'phone_no')
            if ent_df7['ext_phone'].isna().all(): 
                ent_df7['ext_phone'] = get_nested_col(ent_df7, 'phone_number')
            if ent_df7['ext_phone'].isna().all(): 
                ent_df7['ext_phone'] = get_nested_col(ent_df7, 'name')
            ent_df7['clean_phone'] = clean_phone_series(ent_df7['ext_phone'])
            ent_df7 = ent_df7[ent_df7['clean_phone'] != '']
            ent_df7['Participant ID'] = ent_df7['clean_phone'].map(phone_to_pid).fillna(ent_df7['clean_phone'])
            
            ent_city = get_nested_col(ent_df7, 'city')
            ent_df7['Site'] = ent_city.map(CITY_MAP).fillna(ent_city).fillna('Unknown Site')
            ent_df7['Cadre'] = get_nested_col(ent_df7, 'pat_cadre_status').combine_first(get_nested_col(ent_df7, 'cadre')).fillna('Unknown')
            
            if selected_sites: 
                ent_df7 = ent_df7[ent_df7['Site'].isin(selected_sites)]
            if selected_cadres: 
                ent_df7 = ent_df7[ent_df7['Cadre'].str.contains('|'.join(selected_cadres), case=False, na=False)]
                
            surv_phone = get_nested_col(surv_df7, 'pat_phone')
            if surv_phone.isna().all(): 
                surv_phone = get_nested_col(surv_df7, 'phone_no')
            surv_df7['clean_phone'] = clean_phone_series(surv_phone)
            surv_df7 = surv_df7[surv_df7['clean_phone'] != '']
            surv_df7['Participant ID'] = surv_df7['clean_phone'].map(phone_to_pid).fillna(surv_df7['clean_phone'])
            
            delta = end_date - start_date
            date_list = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]
            weeks_in_range = sorted(list(set([f"Week {int(d.strftime('%U')):02d}" for d in date_list])))
            
            date_series = get_nested_col(surv_df7, 'today')
            if date_series.isna().all() and '__system' in surv_df7.columns:
                date_series = surv_df7['__system'].apply(lambda x: x.get('submissionDate') if isinstance(x, dict) else None)

            if date_series is not None and not date_series.isna().all():
                surv_df7['today_dt'] = pd.to_datetime(date_series, errors='coerce').dt.date
                surv_df7 = surv_df7[(surv_df7['today_dt'] >= start_date) & (surv_df7['today_dt'] <= end_date)]
                week_col_raw = 'weekofyear'
                if week_col_raw in surv_df7.columns:
                    surv_df7['week_num'] = pd.to_numeric(surv_df7[week_col_raw], errors='coerce').fillna(0).astype(int)
                else:
                    surv_df7['week_num'] = pd.to_numeric(pd.to_datetime(surv_df7['today_dt']).dt.strftime('%U'), errors='coerce').fillna(0).astype(int)
                surv_df7['Week'] = 'Week ' + surv_df7['week_num'].astype(str).str.zfill(2)
            else:
                surv_df7['Week'] = 'Unknown'
                
            submitted_dict = surv_df7.groupby('Participant ID')['Week'].apply(set).to_dict()
            ent_df7['enrolled_dt'] = pd.to_datetime(get_nested_col(ent_df7, 'today'), errors='coerce').dt.date
            
            missed_records, consec_2_pids, consec_4_pids = [], set(), set()
            unique_participants = ent_df7.drop_duplicates(subset=['Participant ID']).copy()
            
            for _, row in unique_participants.iterrows():
                pid, site, enrolled_d = row['Participant ID'], row['Site'], row['enrolled_dt']
                subs = submitted_dict.get(pid, set())
                missed_indices = []
                
                for i, w in enumerate(weeks_in_range):
                    is_due = True
                    if pd.notna(enrolled_d):
                        if w < f"Week {int(enrolled_d.strftime('%U')):02d}" and enrolled_d.year >= start_date.year: 
                            is_due = False
                    if is_due and w not in subs:
                        missed_records.append({'Participant ID': pid, 'Site': site, 'Week': w})
                        missed_indices.append(i)
                
                has_2, has_4 = False, False
                for i in range(len(missed_indices)):
                    if not has_2 and i+1 < len(missed_indices) and missed_indices[i+1] == missed_indices[i]+1: 
                        has_2 = True
                    if not has_4 and i+3 < len(missed_indices) and missed_indices[i+3] == missed_indices[i]+3 and missed_indices[i+2] == missed_indices[i]+2 and missed_indices[i+1] == missed_indices[i]+1: 
                        has_4 = True
                
                if has_2: consec_2_pids.add(pid)
                if has_4: consec_4_pids.add(pid)
                
            st.markdown("**Weekly Non-Respondents**")
            df_missed = pd.DataFrame(missed_records)
            if not df_missed.empty:
                agg_missed = df_missed.groupby(['Site', 'Week']).size().reset_index(name='Count')
                fig_missed = px.bar(agg_missed, x='Site', y='Count', color='Week', barmode='group', text='Count', color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_missed.update_traces(textposition='outside')
                fig_missed.update_layout(yaxis_title="Non-Respondents (n)", xaxis_title="", margin=dict(t=20, b=0, l=0, r=0))
                st.plotly_chart(fig_missed, use_container_width=True)
            else:
                st.success("No missing responses in the selected period for active participants!")
                
            st.divider()
            unique_participants['Serial No.'] = range(1, len(unique_participants) + 1)
            unique_participants['Name'] = get_nested_col(unique_participants, 'pat_name')
            unique_participants['Age&Sex'] = unique_participants.apply(lambda r: format_age_sex_row(r, get_nested_col(unique_participants, 'pat_age').name if get_nested_col(unique_participants, 'pat_age').name else 'pat_age', get_nested_col(unique_participants, 'pat_gender').name if get_nested_col(unique_participants, 'pat_gender').name else 'pat_gender'), axis=1)
            unique_participants['Department'] = get_nested_col(unique_participants, 'pat_department')
            unique_participants['Designation'] = get_nested_col(unique_participants, 'pat_designation')
            unique_participants['Phone number'] = unique_participants['Participant ID'].map(pid_to_all_phones).fillna(unique_participants['clean_phone'])
            
            disp_cols = ['Serial No.', 'Site', 'Participant ID', 'Name', 'Age&Sex', 'Department', 'Designation', 'Phone number']

            st.subheader("Non-Responders for 2 consecutive weeks")
            if len(weeks_in_range) < 2: 
                st.info("Please select a date range covering at least 2 weeks in the sidebar to view this metric.")
            else:
                df_2 = unique_participants[unique_participants['Participant ID'].isin(consec_2_pids)].copy()
                if not df_2.empty:
                    col1, col2 = st.columns([1, 2.5])
                    with col1:
                        fig_2 = px.bar(df_2.groupby('Site').size().reset_index(name='Count'), x='Site', y='Count', text='Count', color_discrete_sequence=['#EAC13E'])
                        fig_2.update_traces(textposition='outside')
                        fig_2.update_layout(yaxis_title="Count", xaxis_title="", margin=dict(t=20, b=0, l=0, r=0), height=350)
                        st.plotly_chart(fig_2, use_container_width=True)
                    with col2:
                        st.warning("⚠️ **Please contact these participants in person and try to obtain their responses in the coming weeks**")
                        show_2 = df_2[[c for c in disp_cols if c in df_2.columns]].fillna('')
                        show_2['Serial No.'] = range(1, len(show_2) + 1)
                        st.dataframe(style_table(show_2), use_container_width=True, hide_index=True, height=300)
                else: 
                    st.success("No active participants found with 2 consecutive missed weeks!")
                    
            st.divider()
            st.subheader("Non-responders for 4 consecutive weeks")
            if len(weeks_in_range) < 4: 
                st.info("Please select a date range covering at least 4 weeks in the sidebar to view this metric.")
            else:
                df_4 = unique_participants[unique_participants['Participant ID'].isin(consec_4_pids)].copy()
                if not df_4.empty:
                    col1, col2 = st.columns([1, 2.5])
                    with col1:
                        fig_4 = px.bar(df_4.groupby('Site').size().reset_index(name='Count'), x='Site', y='Count', text='Count', color_discrete_sequence=['#C87550'])
                        fig_4.update_traces(textposition='outside')
                        fig_4.update_layout(yaxis_title="Count", xaxis_title="", margin=dict(t=20, b=0, l=0, r=0), height=350)
                        st.plotly_chart(fig_4, use_container_width=True)
                    with col2:
                        st.error("🚨 **Please ascertain the reason for Non-response for 4 consecutive weeks and consider exiting the participant using the participant exit form**")
                        show_4 = df_4[[c for c in disp_cols if c in df_4.columns]].fillna('')
                        show_4['Serial No.'] = range(1, len(show_4) + 1)
                        st.dataframe(style_table(show_4), use_container_width=True, hide_index=True, height=300)
                else: 
                    st.success("No active participants found with 4 consecutive missed weeks!")

        else:
            st.info("Awaiting data from both 'ILI Entities' and 'Surveillance' forms. Please ensure both are successfully fetching.")

else:
    st.info("Awaiting data connection. Please configure your secrets in the Streamlit App Settings.")
