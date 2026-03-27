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
if "play" not in st.session_state:
    st.session_state.play = False

if uploaded_file:
    try:
        # -------------------------
        # LOAD DATA
        # -------------------------
        table = pq.read_table(uploaded_file)
        df = table.to_pandas()

        st.success("Data Loaded Successfully")

        # -------------------------
        # SAFE EVENT CLEANING
        # -------------------------
        if "event" in df.columns:
            df["event"] = df["event"].apply(
                lambda x: x.decode() if isinstance(x, bytes) else x
            )
        else:
            st.error("Missing 'event' column")
            st.stop()

        # -------------------------
        # COORDINATE CHECK
        # -------------------------
        if "x" not in df.columns or "y" not in df.columns:
            st.error("Missing x/y coordinates in dataset")
            st.stop()

        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["x", "y"])

        # -------------------------
        # MATCH FILTER (optional)
        # -------------------------
        if "match_id" in df.columns:
            match_id = st.selectbox("Select Match", df["match_id"].unique())
            df = df[df["match_id"] == match_id]

        # -------------------------
        # GLOBAL NORMALIZATION (FIXED)
        # -------------------------
        x_min, x_max = df["x"].min(), df["x"].max()
        y_min, y_max = df["y"].min(), df["y"].max()

        df["x_norm"] = (df["x"] - x_min) / (x_max - x_min if x_max != x_min else 1)
        df["y_norm"] = (df["y"] - y_min) / (y_max - y_min if y_max != y_min else 1)

        # -------------------------
        # MAP LOAD
        # -------------------------
        map_path = f"{map_choice}_Minimap.png"

        if not os.path.exists(map_path):
            st.error(f"Map not found: {map_path}")
            st.stop()

        img = mpimg.imread(map_path)

        # -------------------------
        # METRICS
        # -------------------------
        st.subheader("📊 Movement Intelligence Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Points", len(df))
        col2.metric("X Range", f"{x_max - x_min:.2f}")
        col3.metric("Y Range", f"{y_max - y_min:.2f}")

        # -------------------------
        # POSITION DATA
        # -------------------------
        position_df = df[df["event"] == "Position"].copy()
        event_df = df[df["event"] != "Position"].copy()

        # -------------------------
        # PATH VIEW
        # -------------------------
        st.subheader("🗺️ Movement Path")

        fig, ax = plt.subplots()
        ax.imshow(img, extent=[0, 1, 0, 1], aspect="auto")

        if not position_df.empty:
            ax.plot(position_df["x_norm"], position_df["y_norm"], linewidth=1)
            ax.scatter(position_df["x_norm"], position_df["y_norm"], s=4)

        st.pyplot(fig)
        plt.close(fig)

        # -------------------------
        # HEATMAP
        # -------------------------
        st.subheader("🔥 Hot Zone Detection")

        if not position_df.empty:
            heatmap, _, _ = np.histogram2d(
                position_df["x_norm"],
                position_df["y_norm"],
                bins=80
            )

            fig2, ax2 = plt.subplots()

            ax2.imshow(heatmap.T, origin="lower", cmap="hot",
                       alpha=0.7, extent=[0, 1, 0, 1])

            ax2.imshow(img, extent=[0, 1, 0, 1], alpha=0.25)

            st.pyplot(fig2)
            plt.close(fig2)

            # -------------------------
            # HIGH ACTIVITY ZONES
            # -------------------------
            st.subheader("🧠 High Activity Zones")

            threshold = np.percentile(heatmap, 95)

            fig3, ax3 = plt.subplots()

            ax3.imshow(heatmap.T, origin="lower", cmap="hot",
                       extent=[0, 1, 0, 1], alpha=0.6)

            ax3.contour(heatmap.T, levels=[threshold], colors="cyan")

            ax3.imshow(img, extent=[0, 1, 0, 1], alpha=0.2)

            st.pyplot(fig3)
            plt.close(fig3)

        # -------------------------
        # ⚔️ EVENT SYSTEM (SAFE)
        # -------------------------
        st.subheader("⚔️ Game Events Map")

        fig4, ax4 = plt.subplots()
        ax4.imshow(img, extent=[0, 1, 0, 1], alpha=0.8)

        if not event_df.empty:
            for e_type, color in {
                "Kill": "red",
                "Death": "black",
                "Loot": "green",
                "StormDeath": "blue"
            }.items():

                temp = event_df[event_df["event"] == e_type]

                if not temp.empty:
                    ax4.scatter(
                        temp["x_norm"],
                        temp["y_norm"],
                        c=color,
                        label=e_type,
                        s=20
                    )

        ax4.legend()
        st.pyplot(fig4)
        plt.close(fig4)

        # -------------------------
        # 👥 BOT VS HUMAN
        # -------------------------
        st.subheader("🤖 Bot vs Human Movement")

        if "player_id" in df.columns:
            df["is_bot"] = df["player_id"].astype(str).str.contains("bot", case=False)
        elif "is_bot" not in df.columns:
            df["is_bot"] = False

        bots = df[df["is_bot"] == True]
        humans = df[df["is_bot"] == False]

        fig5, ax5 = plt.subplots()
        ax5.imshow(img, extent=[0, 1, 0, 1], alpha=0.8)

        if not humans.empty:
            ax5.scatter(humans["x_norm"], humans["y_norm"], c="cyan", s=3, label="Human")

        if not bots.empty:
            ax5.scatter(bots["x_norm"], bots["y_norm"], c="orange", s=3, label="Bot")

        ax5.legend()
        st.pyplot(fig5)
        plt.close(fig5)

        # -------------------------
        # 🎮 REPLAY SYSTEM
        # -------------------------
        st.subheader("🎮 Replay System")

        colA, colB = st.columns(2)

        start = colA.button("▶️ Start Replay")
        stop = colB.button("⏹ Stop Replay")

        if start:
            st.session_state.play = True

        if stop:
            st.session_state.play = False

        placeholder = st.empty()

        if st.session_state.play and not position_df.empty:
            for i in range(10, len(position_df), 10):

                if not st.session_state.play:
                    break

                fig6, ax6 = plt.subplots()
                ax6.imshow(img, extent=[0, 1, 0, 1], alpha=0.9)

                ax6.plot(
                    position_df["x_norm"].iloc[:i],
                    position_df["y_norm"].iloc[:i],
                    color="blue"
                )

                ax6.scatter(
                    position_df["x_norm"].iloc[i - 1],
                    position_df["y_norm"].iloc[i - 1],
                    color="red",
                    s=30
                )

                placeholder.pyplot(fig6)
                plt.close(fig6)

                time.sleep(0.08)

    except Exception as e:
        st.error(f"Error: {e}")