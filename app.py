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
# Streamlit Community Cloud maps TOML secrets to environment variables.
# ---------------------------------------------------------
ODK_URL = os.environ.get("ODK_URL")
ODK_USERNAME = os.environ.get("ODK_USERNAME")
ODK_PASSWORD = os.environ.get("ODK_PASSWORD")
PROJECT_ID = os.environ.get("PROJECT_ID")

FORM_ID_SURVEILLANCE = os.environ.get("FORM_ID_SURVEILLANCE")
ENTITY_ID_ILI = os.environ.get("ENTITY_ID_ILI")

# ---------------------------------------------------------
# 3. GLOBAL OPTIMIZED DATA HELPERS
# ---------------------------------------------------------
def get_nested_col(df, key):
    if key in df.columns: return df[key]
    for col in df.columns:
        valid_series = df[col].dropna()
        if not valid_series.empty:
            first_val = valid_series.iloc[0]
            if isinstance(first_val, dict):
                has_key = any(key in d for d in valid_series.head(100) if isinstance(d, dict))
                if has_key:
                    return df[col].apply(lambda x: x.get(key) if isinstance(x, dict) else None)
    matches = [c for c in df.columns if key in str(c).lower()]
    if matches: return df[matches[0]]
    return pd.Series([None]*len(df), index=df.index)

def clean_phone_series(s):
    # Ensure series elements are handled safely even if they contain None, dicts, or mixed types
    if s is None:
        return pd.Series(dtype=str)
    
    # Convert series to string, filling missing values safely
    s_str = s.fillna("").astype(str)
    
    # Remove decimal points (common when reading IDs from Excel/CSV) and non-digits
    s_cleaned = s_str.str.replace(r'\.0$', '', regex=True).str.replace(r'\D', '', regex=True)
    
    # Safely slice the last 10 characters only if the length is at least 10
    return s_cleaned.apply(lambda x: x[-10:] if isinstance(x, str) and len(x) >= 10 else x)
    
def format_age_sex_row(row, age_col, sex_col):
    age = str(row[age_col]).split('.')[0] if pd.notna(row[age_col]) else '?'
    if age == 'nan' or age == 'None': age = '?'
    sex = str(row[sex_col]).strip() if pd.notna(row[sex_col]) else '?'
    if sex == 'nan' or sex == 'None': sex = '?'
    sex = sex[0].upper() if sex != '?' and len(sex) > 0 else '?'
    return f"{age} / {sex}"

CITY_MAP = {'NC': 'NCT Delhi', 'PU': 'Pune', 'CH': 'Chennai', 'JO': 'Jodhpur', 'KO': 'Kolkata', 'GU': 'Guwahati'}
ABBR_MAP = {v: k for k, v in CITY_MAP.items()}
for k in CITY_MAP.keys(): ABBR_MAP[k] = k 

def get_phase_global(city, hosp, current_study_phase):
    if pd.isna(city): return 'Unknown'
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
@st.cache_data(ttl=43200, persist="disk", show_spinner=False)
def fetch_odk_data(endpoint_suffix):
    if not all([ODK_URL, ODK_USERNAME, ODK_PASSWORD, PROJECT_ID]):
        return pd.DataFrame()
        
    endpoint = f"{ODK_URL.rstrip('/')}/v1/projects/{PROJECT_ID}/{endpoint_suffix}"
    try:
        response = requests.get(endpoint, auth=(ODK_USERNAME, ODK_PASSWORD))
        response.raise_for_status()
        data = response.json()
        if "value" in data:
            return pd.DataFrame(data["value"])
    except requests.exceptions.HTTPError as errh:
        st.error(f"HTTP Error for {endpoint_suffix}: {errh}")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    return pd.DataFrame()

def load_all_data():
    data = {}
    if FORM_ID_SURVEILLANCE: data['Surveillance'] = fetch_odk_data(f"forms/{FORM_ID_SURVEILLANCE}.svc/Submissions")
    if ENTITY_ID_ILI: data['ILI Entities'] = fetch_odk_data(f"datasets/{ENTITY_ID_ILI}.svc/Entities")
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
        if cadre_col_ent not in ent_df2a.columns: ent_df2a[cadre_col_ent] = 'Unknown'
        
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
            surv_df2a['extracted_hosp'] = get_nested_col(surv_df2a, 'hosp_name')
