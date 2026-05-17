from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="BetViero Dashboard",
    page_icon="🎮",
    layout="wide",
)

from ui.lol_dashboard import render_lol_dashboard
from ui.shared import check_password
from ui.tennis_dashboard import render_tennis_dashboard


def main() -> None:
    if not check_password():
        return

    sport = st.sidebar.selectbox(
        "Deporte",
        ["League of Legends", "Tenis"],
        index=0,
    )

    if sport == "Tenis":
        render_tennis_dashboard()
        return

    render_lol_dashboard()


if __name__ == "__main__":
    main()
