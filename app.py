import streamlit as st
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

st.title("LILA Player Data Viewer")
st.write("Upload your .nakama-0 file below")

uploaded_file = st.file_uploader("Upload file", type=None)

if uploaded_file is not None:
    try:
        table = pq.read_table(uploaded_file)
        df = table.to_pandas()

        st.success("File loaded successfully!")

        # Decode event column
        if 'event' in df.columns:
            df['event'] = df['event'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
            )

        position_df = df[df['event'] == 'Position'].copy()

        if not position_df.empty:

            st.subheader("📊 Position Stats")
            st.write(position_df[['x', 'y']].describe())

            # Load map
            img = mpimg.imread("AmbroseValley_Minimap.png")

            xmin, xmax = position_df['x'].min(), position_df['x'].max()
            ymin, ymax = position_df['y'].min(), position_df['y'].max()

            # -------------------------
            # 🗺️ FULL PATH VIEW
            # -------------------------
            st.subheader("🗺️ Player Path")

            fig, ax = plt.subplots()

            ax.imshow(img, extent=[xmin, xmax, ymin, ymax], aspect='auto')

            ax.plot(position_df['x'], position_df['y'], color='blue', alpha=0.6)

            ax.scatter(position_df['x'], position_df['y'], s=5, c='red')

            st.pyplot(fig)

            # -------------------------
            # 🔥 HEATMAP
            # -------------------------
            st.subheader("🔥 Movement Heatmap")

            heatmap, xedges, yedges = np.histogram2d(
                position_df['x'],
                position_df['y'],
                bins=50
            )

            fig2, ax2 = plt.subplots()

            ax2.imshow(
                heatmap.T,
                origin='lower',
                cmap='hot',
                alpha=0.6,
                extent=[xmin, xmax, ymin, ymax]
            )

            ax2.set_title("Player Heatmap (High Activity Zones)")

            st.pyplot(fig2)

            # -------------------------
            # 🎮 REPLAY FEATURE (NEW)
            # -------------------------
            st.subheader("🎮 Movement Replay")

            step = st.slider(
                "Replay Progress",
                1,
                len(position_df),
                len(position_df) // 2
            )

            temp_df = position_df.iloc[:step]

            fig3, ax3 = plt.subplots()

            ax3.imshow(img, extent=[xmin, xmax, ymin, ymax], aspect='auto')

            ax3.plot(temp_df['x'], temp_df['y'], color='blue', alpha=0.6)

            ax3.scatter(temp_df['x'], temp_df['y'], c='red', s=5)

            ax3.set_title("Live Movement Replay")

            st.pyplot(fig3)

        else:
            st.write("No position data found.")

    except Exception as e:
        st.error(f"Error: {e}")