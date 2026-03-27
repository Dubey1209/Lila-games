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

        st.subheader("Data Preview")
        st.dataframe(df.head())

        # 🔥 Filter position data
        position_df = df[df['event'] == 'Position'].copy()

        st.subheader("Map Visualization (Real Map)")

        if not position_df.empty:

            # ✅ DEBUG stats
            st.write("📊 Position Data Stats (x, y)")
            st.write(position_df[['x', 'y']].describe())

            # 🔥 SPEED CALCULATION
            position_df = position_df.sort_index()

            position_df['dx'] = position_df['x'].diff()
            position_df['dy'] = position_df['y'].diff()
            position_df['speed'] = np.sqrt(position_df['dx']**2 + position_df['dy']**2)

            st.subheader("⚡ Speed Stats")
            st.write(position_df['speed'].describe())

            # Load map image
            img = mpimg.imread("AmbroseValley_Minimap.png")

            fig, ax = plt.subplots()

            # coordinate bounds
            xmin, xmax = position_df['x'].min(), position_df['x'].max()
            ymin, ymax = position_df['y'].min(), position_df['y'].max()

            # Show map aligned
            ax.imshow(img, extent=[xmin, xmax, ymin, ymax], aspect='auto')

            # 🔥 movement path
            ax.plot(
                position_df['x'],
                position_df['y'],
                linewidth=1.5,
                alpha=0.7,
                color='blue',
                label='Path'
            )

            # 🔴 positions
            ax.scatter(
                position_df['x'],
                position_df['y'],
                s=8,
                c='red',
                label='Points'
            )

            ax.set_title("Player Movement on Map")

            # lock axes
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            ax.legend()

            st.pyplot(fig)

        else:
            st.write("No position data found.")

    except Exception as e:
        st.error(f"Error: {e}")