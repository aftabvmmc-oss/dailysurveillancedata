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
    page_title="HCW Daily Surveillance",
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
ENTITY_ID_ILI = get_secret("ENTITY_ID_ILI")

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
    data = {}
    if FORM_ID_SURVEILLANCE: 
        data['Surveillance'] = fetch_odk_data(f"forms/{FORM_ID_SURVEILLANCE}.svc/Submissions")
    if ENTITY_ID_ILI: 
        data['ILI Entities'] = fetch_odk_data(f"datasets/{ENTITY_ID_ILI}.svc/Entities")
    return data

# ---------------------------------------------------------
# 5. DASHBOARD UI
# ---------------------------------------------------------
st.title("🏥 HCW Daily Surveillance")

with st.spinner("Connecting to ODK Central and downloading datasets..."):
    datasets = load_all_data()

if datasets:
    st.sidebar.header("🔄 Data Sync")
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
    btn_col1, btn_col2, btn_col3 = st.sidebar.columns(3)
        
    if btn_col1.button("Last Wk", use_container_width=True):
        last_sunday = today - datetime.timedelta(days=today.weekday() + 1)
        last_monday = last_sunday - datetime.timedelta(days=6)
        st.session_state.date_range_val = (last_monday, last_sunday)
        st.session_state.date_mode = "Date Range"

    if btn_col2.button("Last 4Wks", use_container_width=True):
        this_monday = today - datetime.timedelta(days=today.weekday())
        start_4w = this_monday - datetime.timedelta(days=28)
        st.session_state.date_range_val = (start_4w, today)
        st.session_state.date_mode = "Date Range"
        
    if btn_col3.button("Reset", use_container_width=True):
        st.session_state.date_range_val = (default_start, default_end)
        st.session_state.date_mode = "Date Range"
    
    if 'date_mode' not in st.session_state:
        st.session_state.date_mode = "Date Range"
        
    date_mode = st.sidebar.radio("Date Selection", ["Date Range", "Single Date"], key="date_mode")
    
    if date_mode == "Single Date":
        selected_date = st.sidebar.date_input("Select Date", value=st.session_state.date_range_val[1])
        start_date = end_date = selected_date
    else:
        date_range = st.sidebar.date_input("Select Date Range", value=st.session_state.date_range_val)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            st.session_state.date_range_val = date_range
        elif isinstance(date_range, tuple) and len(date_range) == 1:
            start_date = end_date = date_range[0]
        else:
            start_date = default_start
            end_date = default_end
            
    all_sites = ['NCT Delhi', 'Pune', 'Chennai', 'Jodhpur', 'Kolkata', 'Guwahati']
    selected_sites = st.sidebar.multiselect("Select Site(s)", options=all_sites, default=all_sites)
    
    all_cadres = ['Doctor', 'Nurse', 'Allied']
    selected_cadres = st.sidebar.multiselect("Select Cadre(s)", options=all_cadres, default=all_cadres)

    # ==========================================
    # DAILY SURVEILLANCE DATA
    # ==========================================
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

        # Apply Filters
        if selected_sites:
            surv_df2a = surv_df2a[surv_df2a['Site_Name'].isin(selected_sites)]
            ent_df2a = ent_df2a[ent_df2a['Site_Name'].isin(selected_sites)]
            
        # Extract and filter by submission date safely
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
        date_list = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]
        weekday_counts = {i: 0 for i in range(7)}
        for d in date_list: 
            weekday_counts[d.weekday()] += 1
            
        day_map = {'1': 0, 'monday': 0, 'mon': 0, '2': 1, 'tuesday': 1, 'tue': 1,
                   '3': 2, 'wednesday': 2, 'wed': 2, '4': 3, 'thursday': 3, 'thu': 3,
                   '5': 4, 'friday': 4, 'fri': 4}
                   
        contact_day_col = 'pat_contact_day' if 'pat_contact_day' in ent_df2a.columns else next((c for c in ent_df2a.columns if 'contact_day' in str(c).lower()), None)
        if not contact_day_col:
            ent_df2a['pat_contact_day'] = get_nested_col(ent_df2a, 'pat_contact_day')
            if ent_df2a['pat_contact_day'].notna().any():
                contact_day_col = 'pat_contact_day'
        
        if contact_day_col:
            ent_df2a['contact_wd'] = ent_df2a[contact_day_col].astype(str).str.lower().str.strip().map(day_map)
            ent_df2a['forms_due'] = ent_df2a['contact_wd'].map(weekday_counts).fillna(0)
            total_due = int(ent_df2a['forms_due'].sum())
        else:
            total_due = "Unknown"
            
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

        if 'forms_due' not in ent_df2a.columns:
            ent_df2a['forms_due'] = 1

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
            def style_sub(df): 
                return df.style.apply(lambda x: ['background-color: rgba(150, 150, 150, 0.1)' if i % 2 == 0 else '' for i in range(len(x))], axis=0).set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}])
            st.dataframe(style_sub(sub_totals), use_container_width=True, hide_index=True)
            
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
                style_sub(surv_ll_disp), use_container_width=True, hide_index=True,
                column_config={"Serial No.": st.column_config.NumberColumn("S.No.", width="small")}
            )
        else:
            st.info("No surveillance forms filled in the selected date range.")
    else:
        st.info("Awaiting data from both 'ILI Entities' and 'Surveillance' forms.")
else:
    st.info("Awaiting data connection. Please configure your secrets in the Streamlit App Settings.")
