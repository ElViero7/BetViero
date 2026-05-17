from __future__ import annotations

import pandas as pd
import streamlit as st


def format_percentage(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2%}"


def format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}"


def add_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .bet-card {
            background: linear-gradient(180deg, #141824 0%, #10131d 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }
        .bet-card-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 0.88rem;
            color: #b6bfd6;
        }
        .bet-card-match {
            font-size: 1.02rem;
            font-weight: 700;
            color: #f7f9ff;
            margin-bottom: 8px;
        }
        .bet-card-market {
            font-size: 0.95rem;
            color: #dfe6fb;
            margin-bottom: 10px;
        }
        .bet-card-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }
        .bet-chip {
            background: rgba(255,255,255,0.04);
            border-radius: 12px;
            padding: 10px;
        }
        .bet-chip-label {
            font-size: 0.75rem;
            color: #97a3bf;
            margin-bottom: 4px;
        }
        .bet-chip-value {
            font-size: 0.98rem;
            color: #ffffff;
            font-weight: 700;
        }
        .bet-status {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .bet-status-won { background: rgba(39,174,96,0.18); color: #6ee7a1; }
        .bet-status-lost { background: rgba(231,76,60,0.18); color: #ff8f8f; }
        .bet-status-pending { background: rgba(241,196,15,0.18); color: #ffd86b; }
        .bet-status-void { background: rgba(149,165,166,0.18); color: #d0d6db; }
        .control-box {
            background: linear-gradient(180deg, #141824 0%, #10131d 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def check_password() -> bool:
    def get_expected_password() -> str | None:
        try:
            return st.secrets["APP_PASSWORD"]
        except Exception:
            return None

    def password_entered() -> None:
        expected_password = get_expected_password()
        if not expected_password:
            st.session_state["password_correct"] = False
            return

        if st.session_state.get("app_password") == expected_password:
            st.session_state["password_correct"] = True
            del st.session_state["app_password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct"):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="app_password")

    if "password_correct" in st.session_state:
        if not get_expected_password():
            st.error("Falta configurar APP_PASSWORD en Streamlit secrets.")
        else:
            st.error("Password incorrecto.")

    return False
