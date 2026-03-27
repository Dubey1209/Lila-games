import streamlit as st
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

st.set_page_config(page_title="LILA Player Viewer", layout="wide")

st.title("🎮 LILA Player Data Viewer")

# -------------------------
# MAP SELECTION
# -------------------------
map_choice = st.selectbox(
    "Select Map",
    ["AmbroseValley", "GrandRift", "Lockdown"]
)

uploaded_file = st.file_uploader("Upload .nakama-0 file")

if uploaded_file is not None:
    try:
        # -------------------------
        # LOAD DATA
        # -------------------------
        table = pq.read_table(uploaded_file)
        df = table.to_pandas()

        st.success("File loaded successfully!")

        # Decode event column safely
        if 'event' in df.columns:
            df['event'] = df['event'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
            )

        # Filter position data
        position_df = df[df['event'] == 'Position'].copy()

        if position_df.empty:
            st.warning("No position data found.")
            st.stop()

        # Clean numeric data
        position_df['x'] = pd.to_numeric(position_df['x'], errors='coerce')
        position_df['y'] = pd.to_numeric(position_df['y'], errors='coerce')
        position_df = position_df.dropna(subset=['x', 'y'])

        st.subheader("📊 Position Stats")
        st.dataframe(position_df[['x', 'y']].describe())

        # -------------------------
        # LOAD MAP IMAGE (DYNAMIC)
        # -------------------------
        map_path = f"{map_choice}_Minimap.png"

        if not os.path.exists(map_path):
            st.error(f"Map image not found: {map_path}")
            st.stop()

        img = mpimg.imread(map_path)

        # coordinate bounds
        xmin, xmax = position_df['x'].min(), position_df['x'].max()
        ymin, ymax = position_df['y'].min(), position_df['y'].max()

        # -------------------------
        # 🗺️ FULL PATH VIEW
        # -------------------------
        st.subheader("🗺️ Player Path")

        fig, ax = plt.subplots()

        ax.imshow(img, extent=[xmin, xmax, ymin, ymax], aspect='auto')

        ax.plot(
            position_df['x'],
            position_df['y'],
            color='blue',
            linewidth=1,
            alpha=0.7
        )

        ax.scatter(
            position_df['x'],
            position_df['y'],
            s=6,
            c='red'
        )

        ax.set_title(f"{map_choice} - Player Path")

        st.pyplot(fig)
        plt.close(fig)

        # -------------------------
        # 🔥 HEATMAP
        # -------------------------
        st.subheader("🔥 Movement Heatmap")

        heatmap, _, _ = np.histogram2d(
            position_df['x'],
            position_df['y'],
            bins=60
        )

        fig2, ax2 = plt.subplots()

        ax2.imshow(
            heatmap.T,
            origin='lower',
            cmap='hot',
            alpha=0.65,
            extent=[xmin, xmax, ymin, ymax],
            aspect='auto'
        )

        # map overlay (important for visual clarity)
        ax2.imshow(
            img,
            extent=[xmin, xmax, ymin, ymax],
            alpha=0.25,
            aspect='auto'
        )

        ax2.set_title(f"{map_choice} - Heatmap")

        st.pyplot(fig2)
        plt.close(fig2)

        # -------------------------
        # 🎮 REPLAY FEATURE
        # -------------------------
        st.subheader("🎮 Movement Replay")

        step = st.slider(
            "Replay Progress",
            min_value=1,
            max_value=len(position_df),
            value=min(100, len(position_df))
        )

        temp_df = position_df.iloc[:step]

        fig3, ax3 = plt.subplots()

        ax3.imshow(img, extent=[xmin, xmax, ymin, ymax], aspect='auto')

        ax3.plot(
            temp_df['x'],
            temp_df['y'],
            color='blue',
            linewidth=1,
            alpha=0.8
        )

        ax3.scatter(
            temp_df['x'],
            temp_df['y'],
            c='red',
            s=6
        )

        ax3.set_title(f"{map_choice} - Replay Frame {step}")

        st.pyplot(fig3)
        plt.close(fig3)

    except Exception as e:
        st.error(f"Error: {e}")