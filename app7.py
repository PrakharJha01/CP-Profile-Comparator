import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  # To load environment variables from .env file

load_dotenv()  # Load environment variables from .env file


# Using the official Codeforces API to fetch profile data.
# (Scraping the profile HTML page directly is unreliable on cloud hosts like
# Streamlit Community Cloud: Codeforces often rejects requests without a
# browser-like User-Agent, and datacenter IPs get rate-limited/blocked more
# aggressively than home IPs. The public API is meant for programmatic use
# and returns clean JSON instead.)
CF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def get_profile(handle):
    # --- Basic user info (rank, rating, max rating, avatar) ---
    try:
        info_resp = requests.get(
            "https://codeforces.com/api/user.info",
            params={"handles": handle},
            headers=CF_HEADERS,
            timeout=10,
        )
        info_resp.raise_for_status()
        info_json = info_resp.json()
    except requests.exceptions.RequestException as e:
        st.session_state["last_fetch_error"] = f"{handle}: network error ({e})"
        return None

    if info_json.get("status") != "OK":
        st.session_state["last_fetch_error"] = (
            f"{handle}: {info_json.get('comment', 'user not found')}"
        )
        return None

    user = info_json["result"][0]

    data = {"Handle": handle}

    avatar = user.get("titlePhoto") or user.get("avatar")
    if avatar and avatar.startswith("//"):
        avatar = "https:" + avatar
    data["Avatar"] = avatar or None

    rank = user.get("rank", "unrated")
    data["Tag"] = rank.title() if rank else "Unrated"

    data["Rating"] = str(user.get("rating", "N/A"))
    data["Max Rating"] = str(user.get("maxRating", "N/A"))

    # --- Problems solved: count unique accepted submissions ---
    try:
        status_resp = requests.get(
            "https://codeforces.com/api/user.status",
            params={"handle": handle, "from": 1, "count": 100000},
            headers=CF_HEADERS,
            timeout=15,
        )
        status_resp.raise_for_status()
        status_json = status_resp.json()
        solved = set()
        if status_json.get("status") == "OK":
            for sub in status_json["result"]:
                if sub.get("verdict") == "OK":
                    prob = sub["problem"]
                    solved.add((prob.get("contestId"), prob.get("index")))
        data["Problems Solved"] = str(len(solved))
    except requests.exceptions.RequestException:
        data["Problems Solved"] = "N/A"

    return data


def rating_color(rating_str):
    """Approximate Codeforces rating colors."""
    try:
        r = int(rating_str)
    except (ValueError, TypeError):
        return "#c9ccd6"
    if r < 1200:
        return "#c9ccd6"
    if r < 1400:
        return "#3dd68c"
    if r < 1600:
        return "#20c9c0"
    if r < 1900:
        return "#7c9bff"
    if r < 2100:
        return "#d68bff"
    if r < 2400:
        return "#ffb84d"
    return "#ff6b81"


def to_int(v):
    return int(v) if str(v).isdigit() else 0


st.set_page_config(
    page_title="Codeforces Comparator",
    page_icon="🏆",
    layout="wide"
)

# ----------------------------- Custom CSS -----------------------------
st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* ---------- Animated gradient background ---------- */
    [data-testid="stAppViewContainer"], .stApp {
        background: radial-gradient(circle at 12% 15%, rgba(255, 224, 178, 0.55) 0%, transparent 40%),
                    radial-gradient(circle at 30% 55%, rgba(214, 150, 255, 0.55) 0%, transparent 45%),
                    linear-gradient(125deg, #3a1c71 0%, #4b3fc0 25%, #3f6ee8 55%, #1fa2ff 75%, #12d6df 100%);
        background-size: 200% 200%;
        animation: gradientShift 18s ease infinite;
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    .block-container {
        padding-top: 2.2rem;
        max-width: 1200px;
    }

    /* ---------- Global text contrast on the gradient ---------- */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li {
        color: #f5f6fa;
    }
    .stMarkdown a { color: #9fd8ff; }

    /* Alerts (warning/error/success) keep readable text on their own light backgrounds */
    div[data-testid="stAlert"] { border-radius: 14px; }
    div[data-testid="stAlert"] p { color: #1c1c22 !important; font-weight: 600; }

    /* Spinner + caption text */
    div[data-testid="stSpinner"] p {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* ---------- Title ---------- */
    .cf-title {
        text-align: center;
        font-size: 3.4rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #1c1c22;
        text-shadow: 0 2px 10px rgba(255,255,255,0.25);
        margin-bottom: 0.3rem;
    }
    .cf-subtitle {
        text-align: center;
        color: #1c1c22;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 2.4rem;
    }

    /* ---------- Real glass panel (Streamlit container with key="input_panel") ---------- */
    .st-key-input_panel {
        background: rgba(20, 18, 35, 0.55);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 22px;
        padding: 2rem 2.2rem 1.4rem 2.2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.28);
        margin-bottom: 2rem;
    }

    .cf-vs-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        margin-top: 1.9rem;
    }
    .cf-vs-badge span {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #06121c;
        font-weight: 800;
        font-size: 0.95rem;
        padding: 0.5rem 0.9rem;
        border-radius: 50%;
        box-shadow: 0 4px 14px rgba(79, 172, 254, 0.5);
    }

    /* ---------- Inputs ---------- */
    div[data-testid="column"] {
        padding: 0 14px;
    }
    div[data-testid="stTextInput"] label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.3);
        padding: 0.85rem 1.1rem;
        background-color: rgba(255,255,255,0.1);
        color: #ffffff;
        font-size: 1.08rem;
        font-weight: 500;
        transition: border 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: rgba(255,255,255,0.6);
    }
    div[data-testid="stTextInput"] input:focus {
        border: 1px solid #4facfe;
        background-color: rgba(255,255,255,0.16);
        box-shadow: 0 0 0 4px rgba(79, 172, 254, 0.3);
    }
    div[data-testid="stTextInput"] small,
    div[data-baseweb="tooltip"] {
        color: rgba(255,255,255,0.75) !important;
    }

    /* ---------- Compare button ---------- */
    div.stButton, div[data-testid="stFormSubmitButton"] {
        display: flex;
        justify-content: center;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #06121c;
        font-weight: 700;
        font-size: 1.15rem;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 3.4rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.45);
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-4px) scale(1.04);
        box-shadow: 0 14px 30px rgba(79, 172, 254, 0.6);
        filter: brightness(1.1);
        color: #06121c;
    }
    div.stButton > button:active, div[data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(-1px) scale(0.98);
    }

    /* ---------- Profile cards ---------- */
    .profile-card {
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 1.8rem 1rem 2rem 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
    }
    .profile-card:hover {
        transform: translateY(-6px) scale(1.015);
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.4);
        background: rgba(255,255,255,0.13);
    }
    .profile-handle {
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 1rem;
        margin-bottom: 0.25rem;
        text-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }
    .profile-tag {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.9);
        font-weight: 600;
        text-transform: capitalize;
    }
    .profile-avatar img {
        border-radius: 50%;
        border: 4px solid rgba(255,255,255,0.4);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    /* ---------- Section headers ---------- */
    .section-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1c1c22;
        margin-top: 0.4rem;
        margin-bottom: 1.2rem;
        padding-left: 0.9rem;
        border-left: 6px solid #4facfe;
        text-shadow: none;
    }

    /* ---------- Custom stats table ---------- */
    .stats-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 16px;
        overflow: hidden;
        font-size: 1.05rem;
        background: rgba(15, 14, 26, 0.92);
    }
    .stats-table th {
        background: linear-gradient(90deg, rgba(15,14,26,0.95), rgba(30,28,50,0.95));
        color: #ffffff;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 0.95rem 1rem;
        text-align: center;
    }
    .stats-table th:first-child {
        text-align: left;
    }
    .stats-table td {
        padding: 0.85rem 1rem;
        text-align: center;
        color: #f0f1f5;
        font-weight: 600;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    .stats-table td:first-child {
        text-align: left;
        color: #ffffff;
        font-weight: 700;
    }
    .stats-table tr:nth-child(even) td {
        background: rgba(255,255,255,0.04);
    }
    .stats-table tr:hover td {
        background: rgba(79, 172, 254, 0.25);
        transition: background 0.25s ease;
    }
    .winner-cell {
        position: relative;
        font-weight: 800 !important;
    }
    .winner-cell::after {
        content: " 🏅";
    }

    /* ---------- Plotly chart container ---------- */
    div[data-testid="stPlotlyChart"] {
        border-radius: 18px;
        overflow: hidden;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 0.6rem;
    }

    /* ---------- AI section ---------- */
    .ai-box {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        color: #1c1c22;
        line-height: 1.7;
        backdrop-filter: blur(10px);
    }
    .ai-box, .ai-box p, .ai-box span, .ai-box li, .ai-box div,
    .ai-box strong, .ai-box b, .ai-box em, .ai-box i,
    .ai-box h1, .ai-box h2, .ai-box h3, .ai-box h4,
    .ai-box ul, .ai-box ol, .ai-box blockquote {
        color: #1c1c22 !important;
    }
    .ai-box strong, .ai-box b, .ai-box h1, .ai-box h2, .ai-box h3, .ai-box h4 {
        color: #10131a !important;
    }
    .ai-box a {
        color: #0b63c5 !important;
    }
    .ai-box code {
        color: #a3153e !important;
        background: rgba(0,0,0,0.06);
        padding: 0.1rem 0.35rem;
        border-radius: 6px;
    }

</style>
""", unsafe_allow_html=True)

# ----------------------------- Header -----------------------------
st.markdown('<div class="cf-title">🏆 Codeforces Profile Comparator</div>', unsafe_allow_html=True)
st.markdown('<div class="cf-subtitle">Compare two Codeforces handles side by side, with AI-powered insights</div>', unsafe_allow_html=True)

# ----------------------------- Input panel (real container, not a floating div) -----------------------------
with st.container(key="input_panel"):
    with st.form(key="compare_form"):
        col1, col_vs, col2 = st.columns([1, 0.18, 1], gap="medium")

        with col1:
            user1 = st.text_input(
                "🧑‍💻 First Handle",
                placeholder="e.g. tourist",
                help="Enter a Codeforces username exactly as it appears on codeforces.com",
            )

        with col_vs:
            st.markdown('<div class="cf-vs-badge"><span>VS</span></div>', unsafe_allow_html=True)

        with col2:
            user2 = st.text_input(
                "🧑‍💻 Second Handle",
                placeholder="e.g. Errichto",
                help="Enter a Codeforces username exactly as it appears on codeforces.com",
            )

        compare_clicked = st.form_submit_button("⚔️  Compare")

if compare_clicked:

    if user1 == "" or user2 == "":
        st.warning("Enter both handles.")
        st.stop()

    with st.spinner("Fetching profiles..."):
        p1 = get_profile(user1)
        p2 = get_profile(user2)

    if p1 is None or p2 is None:
        reason = st.session_state.get("last_fetch_error", "")
        msg = "Could not fetch one of the profiles."
        if reason:
            msg += f" ({reason})"
        st.error(msg)
        st.stop()

    # ----------------------------- Profile cards -----------------------------
    st.markdown('<div class="section-header">👨🏼‍💻 Profiles</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    for col, user, p in [(c1, user1, p1), (c2, user2, p2)]:
        with col:
            color = rating_color(p["Rating"])
            avatar_html = f'<img src="{p["Avatar"]}" width="120">' if p["Avatar"] else ""
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-avatar">{avatar_html}</div>
                <div class="profile-handle" style="color:{color};">{user}</div>
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------- Stats table (custom, styled) -----------------------------
    st.markdown('<div class="section-header">📊 Stats</div>', unsafe_allow_html=True)

    r1, r2 = to_int(p1["Rating"]), to_int(p2["Rating"])
    m1, m2 = to_int(p1["Max Rating"]), to_int(p2["Max Rating"])
    s1, s2 = to_int(p1["Problems Solved"]), to_int(p2["Problems Solved"])

    def cell(val, is_winner, color=None):
        style = f'style="color:{color};"' if color else ""
        cls = "winner-cell" if is_winner else ""
        return f'<td class="{cls}" {style}>{val}</td>'

    rows_html = f"""
    <tr>
        <td>Tag</td>
        {cell(p1['Tag'], False, rating_color(p1['Rating']))}
        {cell(p2['Tag'], False, rating_color(p2['Rating']))}
    </tr>
    <tr>
        <td>Rating</td>
        {cell(p1['Rating'], r1 > r2)}
        {cell(p2['Rating'], r2 > r1)}
    </tr>
    <tr>
        <td>Max Rating</td>
        {cell(p1['Max Rating'], m1 > m2)}
        {cell(p2['Max Rating'], m2 > m1)}
    </tr>
    <tr>
        <td>Problems Solved</td>
        {cell(p1['Problems Solved'], s1 > s2)}
        {cell(p2['Problems Solved'], s2 > s1)}
    </tr>
    """

    table_html = f"""
    <table class="stats-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>{user1}</th>
                <th>{user2}</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)

    # ----------------------------- Improved chart -----------------------------
    st.markdown('<div class="section-header">📈 Rating Comparison</div>', unsafe_allow_html=True)

    current_ratings = [r1, r2]
    max_ratings = [m1, m2]
    users = [user1, user2]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=users,
        y=current_ratings,
        name="Current Rating",
        marker_color="#ff6b6b",
        text=current_ratings,
        textposition="outside",
        marker_line_width=0,
    ))

    fig.add_trace(go.Bar(
        x=users,
        y=max_ratings,
        name="Max Rating",
        marker_color="#feca57",
        text=max_ratings,
        textposition="outside",
        marker_line_width=0,
    ))

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#ffffff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(t=30, b=10, l=10, r=10),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        bargap=0.4,
        bargroupgap=0.18,
        height=320,
    )

    # Render the chart smaller by placing it in a centered, narrower column
    chart_col1, chart_col2, chart_col3 = st.columns([1, 2, 1])
    with chart_col2:
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------- AI comparison -----------------------------
    st.markdown('<div class="section-header">🤖 AI Comparative Analysis & Conclusion</div>', unsafe_allow_html=True)

    with st.spinner("Generating AI comparison..."):
        try:
            # Initialize Gemini model (ensure GEMINI_API_KEY or GOOGLE_API_KEY is set in environment/secrets)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7
            )

            prompt = f"""
            You are a competitive programming coach. Compare the following two Codeforces users based on their statistics:

            **User 1: {user1}**
            - Current Tag/Rank: {p1['Tag']}
            - Current Rating: {p1['Rating']}
            - Max Rating: {p1['Max Rating']}
            - Problems Solved: {p1['Problems Solved']}

            **User 2: {user2}**
            - Current Tag/Rank: {p2['Tag']}
            - Current Rating: {p2['Rating']}
            - Max Rating: {p2['Max Rating']}
            - Problems Solved: {p2['Problems Solved']}

            Please provide:
            1. A concise head-to-head comparison highlighting current form, peak rating, and practice volume (problems solved).
            2. A clear, verdict-driven conclusion on who currently has the edge and actionable advice for both users.
            3. Provide an overall score out of 100 for each user based on their performance and potential.
            """

            response = llm.invoke(prompt)
            st.markdown(f'<div class="ai-box">{response.content}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating AI conclusion: {e}")