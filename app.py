import streamlit as st
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time

st.set_page_config(page_title="Game Intelligence Dashboard", layout="wide")

st.title("🎮 Game Movement Intelligence Dashboard")

# -------------------------
# MAP SELECTOR
# -------------------------
map_choice = st.selectbox(
    "Select Map",
    ["AmbroseValley", "GrandRift", "Lockdown"]
)

uploaded_file = st.file_uploader("Upload .nakama-0 file")

# -------------------------
# SESSION STATE
# -------------------------
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

if "frame" not in st.session_state:
    st.session_state.frame = 5


@st.cache_data
def load_map(path):
    return mpimg.imread(path)


if uploaded_file:
    try:
        table = pq.read_table(uploaded_file)
        df = table.to_pandas()

        st.success("Data Loaded Successfully")

        # -------------------------
        # CLEAN EVENT
        # -------------------------
        if "event" not in df.columns:
            st.error("Missing 'event' column")
            st.stop()

        df["event"] = df["event"].apply(
            lambda x: x.decode() if isinstance(x, bytes) else x
        )

        df["event"] = df["event"].astype(str).str.strip()

        # -------------------------
        # COORD CHECK
        # -------------------------
        if "x" not in df.columns or "y" not in df.columns:
            st.error("Missing x/y coordinates")
            st.stop()

        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["x", "y"])

        # -------------------------
        # MATCH FILTER
        # -------------------------
        if "match_id" in df.columns:
            match_id = st.selectbox("Select Match", df["match_id"].unique())
            df = df[df["match_id"] == match_id]

        # -------------------------
        # PLAYER FILTER
        # -------------------------
        if "player_id" in df.columns:
            selected_player = st.selectbox(
                "Select Player",
                df["player_id"].unique()
            )
            player_df = df[df["player_id"] == selected_player].copy()
        else:
            player_df = df.copy()

        # -------------------------
        # NORMALIZATION
        # -------------------------
        x_min, x_max = df["x"].min(), df["x"].max()
        y_min, y_max = df["y"].min(), df["y"].max()

        dx = (x_max - x_min) if x_max != x_min else 1e-9
        dy = (y_max - y_min) if y_max != y_min else 1e-9

        df["x_norm"] = (df["x"] - x_min) / dx
        df["y_norm"] = (df["y"] - y_min) / dy

        player_df["x_norm"] = (player_df["x"] - x_min) / dx
        player_df["y_norm"] = (player_df["y"] - y_min) / dy

        # -------------------------
        # MAP LOAD
        # -------------------------
        map_path = f"{map_choice}_Minimap.png"

        if not os.path.exists(map_path):
            st.error(f"Map not found: {map_path}")
            st.stop()

        img = load_map(map_path)

        # -------------------------
        # SAFE EVENT FILTERING (FIXED)
        # -------------------------
        event_lower = df["event"].astype(str).str.lower()

        position_df = df[event_lower == "position"].copy()
        event_df = df[event_lower != "position"].copy()

        player_pos = player_df[player_df["event"].astype(str).str.lower() == "position"].copy()

        # -------------------------
        # METRICS
        # -------------------------
        st.subheader("📊 Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", len(df))
        col2.metric("X Spread", f"{x_max - x_min:.2f}")
        col3.metric("Y Spread", f"{y_max - y_min:.2f}")

        # -------------------------
        # GLOBAL PATH
        # -------------------------
        st.subheader("🗺️ Global Movement Path")

        fig, ax = plt.subplots()
        ax.imshow(img, extent=[0, 1, 0, 1], aspect="auto")

        if len(position_df) > 0:
            ax.plot(position_df["x_norm"], position_df["y_norm"], linewidth=1)

        st.pyplot(fig)
        plt.close(fig)

        # -------------------------
        # PLAYER PATH
        # -------------------------
        st.subheader("🧍 Player Path")

        fig2, ax2 = plt.subplots()
        ax2.imshow(img, extent=[0, 1, 0, 1], aspect="auto")

        if len(player_pos) > 0:
            ax2.plot(player_pos["x_norm"], player_pos["y_norm"], color="blue")
            ax2.scatter(player_pos["x_norm"], player_pos["y_norm"], s=5, color="red")

        st.pyplot(fig2)
        plt.close(fig2)

        # -------------------------
        # GLOBAL HEATMAP
        # -------------------------
        st.subheader("🔥 Global Heatmap")

        if len(position_df) > 5:
            heatmap, _, _ = np.histogram2d(
                position_df["x_norm"],
                position_df["y_norm"],
                bins=50
            )

            fig3, ax3 = plt.subplots()
            ax3.imshow(heatmap.T, origin="lower", cmap="hot", alpha=0.6, extent=[0, 1, 0, 1])
            ax3.imshow(img, extent=[0, 1, 0, 1], alpha=0.3)

            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.warning("Not enough position data for heatmap")

        # -------------------------
        # BOT VS HUMAN
        # -------------------------
        st.subheader("🤖 Bot vs Human")

        if "player_id" in df.columns:
            df["is_bot"] = df["player_id"].astype(str).str.lower().str.contains("bot")
        else:
            df["is_bot"] = False

        bots = df[df["is_bot"]]
        humans = df[~df["is_bot"]]

        fig6, ax6 = plt.subplots()
        ax6.imshow(img, extent=[0, 1, 0, 1], alpha=0.8)

        if len(humans) > 0:
            ax6.scatter(humans["x_norm"], humans["y_norm"], c="cyan", s=3, label="Human")

        if len(bots) > 0:
            ax6.scatter(bots["x_norm"], bots["y_norm"], c="orange", s=3, label="Bot")

        ax6.legend()
        st.pyplot(fig6)
        plt.close(fig6)

        # -------------------------
        # REPLAY SYSTEM (FIXED SAFE)
        # -------------------------
        st.subheader("🎮 Smart Replay System")

        colA, colB, colC = st.columns(3)

        if colA.button("▶️ Play / Pause"):
            st.session_state.auto_play = not st.session_state.auto_play

        speed = colB.select_slider("Speed", options=[1, 2, 5], value=1)

        if colC.button("🔄 Reset"):
            st.session_state.frame = 5
            st.session_state.auto_play = False

        if len(position_df) > 0:
            max_frame = max(5, len(position_df) - 1)

            st.session_state.frame = st.slider(
                "Timeline Scrubber",
                min_value=5,
                max_value=max_frame,
                value=min(st.session_state.frame, max_frame)
            )

        placeholder = st.empty()

        if st.session_state.auto_play and len(position_df) > 0:

            step_size = speed * 2

            while st.session_state.auto_play and st.session_state.frame < len(position_df):

                fig7, ax7 = plt.subplots()
                ax7.imshow(img, extent=[0, 1, 0, 1], alpha=0.9)

                ax7.plot(
                    position_df["x_norm"].iloc[:st.session_state.frame],
                    position_df["y_norm"].iloc[:st.session_state.frame],
                    color="blue"
                )

                ax7.scatter(
                    position_df["x_norm"].iloc[st.session_state.frame - 1],
                    position_df["y_norm"].iloc[st.session_state.frame - 1],
                    c="red",
                    s=40
                )

                placeholder.pyplot(fig7)
                plt.close(fig7)

                st.session_state.frame += step_size
                time.sleep(0.08)

        else:
            if len(position_df) > 0:
                fig7, ax7 = plt.subplots()
                ax7.imshow(img, extent=[0, 1, 0, 1], alpha=0.9)

                ax7.plot(
                    position_df["x_norm"].iloc[:st.session_state.frame],
                    position_df["y_norm"].iloc[:st.session_state.frame],
                    color="blue"
                )

                if st.session_state.frame > 0:
                    ax7.scatter(
                        position_df["x_norm"].iloc[st.session_state.frame - 1],
                        position_df["y_norm"].iloc[st.session_state.frame - 1],
                        c="red",
                        s=40
                    )

                placeholder.pyplot(fig7)
                plt.close(fig7)

    except Exception as e:
        st.error(f"Error: {e}")