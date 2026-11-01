
"""
Employee Burnout Risk Prediction Dashboard.

Reads data/burnout_predictions.csv (produced by src/pipeline.py) and displays
a polished HR analytics dashboard with KPI cards, a risk distribution donut
chart, a feature-driver bar chart, a filterable employee table, and a tab
for HR to add a new employee and get an instant live prediction.

Access is restricted to registered HR users.
Passwords are stored using salted PBKDF2 hashing.
"""

import os
import json
import hashlib
import hmac
import secrets
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

USERS_PATH = "data/hr_users.json"
DATA_PATH = "data/burnout_predictions.csv"
RISK_COLORS = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}


# ---------------------------------------------------------------------------
# PASSWORD SECURITY
# ---------------------------------------------------------------------------

def hash_password(password):
    """Create a salted PBKDF2 password hash."""

    iterations = 600000
    salt = secrets.token_hex(16)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        iterations
    ).hex()

    return f"pbkdf2${iterations}${salt}${hashed}"


def load_users():
    if not os.path.exists(USERS_PATH):
        return {}

    with open(USERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def verify_login(username, password):
    """Verify old SHA-256 and new PBKDF2 password hashes."""

    users = load_users()

    if username not in users:
        return False

    stored_password = users[username]

    # Verify new PBKDF2 password.
    if stored_password.startswith("pbkdf2$"):
        try:
            _, iterations, salt, stored_hash = stored_password.split("$")

            calculated_hash = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt),
                int(iterations)
            ).hex()

            return hmac.compare_digest(
                calculated_hash,
                stored_hash
            )

        except (ValueError, TypeError):
            return False

    # Verify old SHA-256 password.
    old_hash = hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()

    if hmac.compare_digest(stored_password, old_hash):

        # Upgrade the old password hash after successful login.
        users[username] = hash_password(password)

        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)

        return True

    return False


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Employee Burnout Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #F8FAFC;
}

.dashboard-header {
    background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.dashboard-header h1 {
    color: #FFFFFF;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
}

.kpi-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    border-left: 5px solid #6366F1;
}

.kpi-card.high {
    border-left-color: #EF4444;
}

.kpi-card.medium {
    border-left-color: #F59E0B;
}

.kpi-card.low {
    border-left-color: #10B981;
}

.kpi-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
}

.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1E293B;
    line-height: 1;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 0.3rem;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1E293B;
    margin: 1.6rem 0 0.8rem 0;
}

.risk-badge {
    padding: 4px 12px;
    border-radius: 999px;
    color: white;
    font-weight: 600;
    font-size: 0.78rem;
    display: inline-block;
}

table.styled-table {
    width: 100%;
    border-collapse: collapse;
    background: #FFFFFF;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

table.styled-table th {
    background: #1E293B;
    color: #F1F5F9;
    text-align: left;
    padding: 12px 14px;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

table.styled-table td {
    padding: 11px 14px;
    font-size: 0.87rem;
    color: #334155;
    border-bottom: 1px solid #F1F5F9;
}

table.styled-table tr:hover td {
    background: #F8FAFC;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 20px;
    font-weight: 600;
}

.login-box {
    max-width: 420px;
    margin: 100px auto;
    background: #FFFFFF;
    padding: 2.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HR LOGIN
# ---------------------------------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.hr_username = None

if not st.session_state.authenticated:

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)

    st.markdown("### 🔒 HR Login")

    st.caption(
        "This dashboard is restricted to authorized HR personnel only."
    )

    entered_username = st.text_input("Username")

    entered_password = st.text_input(
        "Password",
        type="password"
    )

    login_clicked = st.button("Login")

    if login_clicked:

        if verify_login(entered_username, entered_password):

            st.session_state.authenticated = True
            st.session_state.hr_username = entered_username

            st.rerun()

        else:
            st.error(
                "Incorrect username or password. Please try again."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ---------------------------------------------------------------------------
# LOAD EMPLOYEE DATA
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


try:
    df = load_data(DATA_PATH)

except FileNotFoundError:

    st.error(
        f"Could not find {DATA_PATH}. "
        "Run `python src/pipeline.py` first to generate predictions."
    )

    st.stop()


risk_counts = df["risk_level"].value_counts()
total = len(df)


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def risk_badge(risk_level):

    color = RISK_COLORS.get(
        risk_level,
        "#999999"
    )

    return (
        f'<span class="risk-badge" '
        f'style="background-color:{color};">'
        f'{risk_level}</span>'
    )


def kpi_card(col, label, value, css_class, sub=""):

    with col:

        st.markdown(f"""
        <div class="kpi-card {css_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DASHBOARD HEADER
# ---------------------------------------------------------------------------

st.markdown("""
<div class="dashboard-header">
    <h1>Employee Burnout Risk Dashboard</h1>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------------------------

tab_dashboard, tab_add = st.tabs([
    "📊  Dashboard",
    "➕  Add New Employee"
])


# ============================ TAB 1: DASHBOARD ============================

with tab_dashboard:

    k1, k2, k3, k4 = st.columns(4)

    kpi_card(
        k1,
        "Total Employees",
        total,
        "",
        "Across the organization"
    )

    kpi_card(
        k2,
        "High Risk",
        int(risk_counts.get("High", 0)),
        "high",
        f"{risk_counts.get('High', 0) / total:.0%} of workforce"
    )

    kpi_card(
        k3,
        "Medium Risk",
        int(risk_counts.get("Medium", 0)),
        "medium",
        f"{risk_counts.get('Medium', 0) / total:.0%} of workforce"
    )

    kpi_card(
        k4,
        "Low Risk",
        int(risk_counts.get("Low", 0)),
        "low",
        f"{risk_counts.get('Low', 0) / total:.0%} of workforce"
    )

    st.markdown(
        '<div class="section-title">Risk Overview</div>',
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns([1, 1.4])

    # Risk distribution donut chart.
    with chart_col1:

        fig_donut = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.62,
            marker=dict(
                colors=[
                    RISK_COLORS.get(lbl, "#999")
                    for lbl in risk_counts.index
                ]
            ),
            textinfo="label+percent",
            textfont=dict(size=13, family="Inter"),
        )])

        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
            annotations=[dict(
                text=f"{total}<br>employees",
                x=0.5,
                y=0.5,
                font_size=15,
                showarrow=False,
                font_family="Inter"
            )],
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig_donut,
            use_container_width=True
        )

    # Top drivers of high burnout risk.
    with chart_col2:

        high_risk_df = df[df["risk_level"] == "High"]

        if not high_risk_df.empty:

            feature_counts = (
                high_risk_df["top_feature"]
                .value_counts()
                .sort_values()
            )

            fig_bar = go.Figure(go.Bar(
                x=feature_counts.values,
                y=feature_counts.index,
                orientation="h",
                marker=dict(color="#EF4444"),
                text=feature_counts.values,
                textposition="outside",
            ))

            fig_bar.update_layout(
                title=dict(
                    text="Top Drivers of High Risk",
                    font=dict(size=14, family="Inter")
                ),
                margin=dict(t=40, b=10, l=10, r=30),
                height=320,
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(title=""),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

        else:
            st.info(
                "No High-risk employees in the current dataset."
            )

    st.markdown(
        '<div class="section-title">Employee Detail</div>',
        unsafe_allow_html=True
    )

    # Sidebar filters and logout.
    with st.sidebar:

        st.markdown(
            f"**Logged in as:** {st.session_state.hr_username}"
        )

        if st.button("Logout"):

            st.session_state.authenticated = False
            st.session_state.hr_username = None

            st.rerun()

        st.divider()

        st.markdown("### Filters")

        selected_risk = st.multiselect(
            "Risk level",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

        search_id = st.text_input(
            "Search Employee ID",
            ""
        )

        sort_col = st.selectbox(
            "Sort by",
            options=[
                "risk_score",
                "employee_id",
                "risk_level"
            ],
            index=0
        )

        sort_ascending = st.checkbox(
            "Ascending order",
            value=False
        )

    filtered_df = df[
        df["risk_level"].isin(selected_risk)
    ]

    if search_id:

        filtered_df = filtered_df[
            filtered_df["employee_id"]
            .astype(str)
            .str.contains(
                search_id,
                case=False,
                na=False
            )
        ]

    filtered_df = filtered_df.sort_values(
        sort_col,
        ascending=sort_ascending
    )

    st.caption(
        f"Showing {len(filtered_df)} of {total} employees"
    )

    display_df = filtered_df.copy()

    display_df["risk_level"] = (
        display_df["risk_level"].apply(risk_badge)
    )

    html_table = (
        display_df[
            [
                "employee_id",
                "risk_score",
                "risk_level",
                "top_feature",
                "reason",
                "action"
            ]
        ]
        .rename(columns={
            "employee_id": "Employee ID",
            "risk_score": "Risk Score",
            "risk_level": "Risk Level",
            "top_feature": "Top Driving Factor",
            "reason": "Reason",
            "action": "Recommended Action",
        })
        .to_html(
            escape=False,
            index=False,
            classes="styled-table"
        )
    )

    st.markdown(
        html_table,
        unsafe_allow_html=True
    )


# ========================== TAB 2: ADD EMPLOYEE ==========================

with tab_add:

    st.caption(
        "Enter a new employee's details to get an instant "
        "burnout risk prediction — no retraining needed."
    )

    with st.form("new_employee_form"):

        c1, c2, c3 = st.columns(3)

        with c1:

            new_id = st.text_input(
                "Employee ID",
                placeholder="e.g. emp4327"
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            company_type = st.selectbox(
                "Company Type",
                ["Product", "Service"]
            )

        with c2:

            wfh = st.selectbox(
                "WFH Setup Available",
                ["Yes", "No"]
            )

            designation = st.slider(
                "Designation Level",
                0,
                5,
                2
            )

            joining_date = st.date_input(
                "Date of Joining"
            )

        with c3:

            resource_alloc = st.number_input(
                "Resource Allocation",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )

            fatigue_score = st.number_input(
                "Mental Fatigue Score",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Predict Risk"
        )

    if submitted:

        if not new_id:

            st.warning(
                "Please enter an Employee ID."
            )

        else:

            from inference import predict_new_employee

            with st.spinner(
                "Running model + SHAP + generating explanation..."
            ):

                raw_input = {
                    "Gender": gender,
                    "Company Type": company_type,
                    "WFH Setup Available": wfh,
                    "Designation": designation,
                    "Resource Allocation": resource_alloc,
                    "Mental Fatigue Score": fatigue_score,
                    "Date of Joining": str(joining_date),
                }

                result = predict_new_employee(raw_input)

            badge = risk_badge(
                result["risk_level"]
            )

            st.markdown(f"""
            <div class="kpi-card"
                 style="border-left-color:{RISK_COLORS.get(result['risk_level'], '#999')};
                 margin-top:1rem;">

                <div class="kpi-label">
                    Prediction for {new_id}
                </div>

                <div class="kpi-value" style="font-size:1.4rem;">
                    {badge} &nbsp; Score: {result['risk_score']}
                </div>

                <div class="kpi-sub"
                     style="margin-top:0.6rem; line-height:1.5;">

                    <b>Top driving factor:</b>
                    {result['top_feature']}<br>

                    <b>Reason:</b>
                    {result['reason']}<br>

                    <b>Recommended action:</b>
                    {result['action']}

                </div>
            </div>
            """, unsafe_allow_html=True)

            new_row = {
                "employee_id": new_id,
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "top_feature": result["top_feature"],
                "reason": result["reason"],
                "action": result["action"],
            }

            pd.DataFrame([new_row]).to_csv(
                DATA_PATH,
                mode="a",
                header=False,
                index=False
            )

            st.success(
                f"{new_id} added. Switch to the Dashboard tab "
                "and refresh to see them in the table."
            )

            st.cache_data.clear()
