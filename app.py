import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import os
from  scipy.fftpack import fft, fftfreq,ifft


def process_data(data):
    try:        
        # Check if all values in the DataFrame are NaNs
        if data.isna().all().all():
            raise ValueError("All values in the data are NaNs")

        # Replace NaNs and -inf values with the average of neighboring values
        for column in data.columns:
            # Convert column values to numeric
            data[column] = pd.to_numeric(data[column], errors='coerce')

            # Replace NaNs with the average of neighboring values
            nan_indices = data[column].index[data[column].isna()]
            data.loc[nan_indices, column] = (data[column].shift(-1) + data[column].shift(1)) / 2
            
            # Replace -inf with NaN to be handled later
            data[column].replace(-np.inf, np.nan, inplace=True)

        # Check if any NaNs remain after replacing with the average
        if data.isna().any().any():
            raise ValueError("Data contains NaNs after replacing with average")

        # Process the data further as needed
        # For example, perform FFT filtering, peak detection, etc.

        # Return processed data or any other result
        return data
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        raise ValueError("Error reading the CSV file or empty data")

def fft_lowpass_filter(signal, F1=0, F2=2e6):
    spectrum = fft(signal)
    N_samples = len(signal)
    F = fftfreq(N_samples)
    spectrum_lowpass = spectrum * (np.logical_and(abs(F) < F2, abs(F) > F1))
    signal_lowpass = ifft(spectrum_lowpass)
    return np.real(signal_lowpass)

def moving_average_filter(signal, window_size=3):
    """Apply moving average filter to the signal."""
    return signal.rolling(window=window_size, min_periods=1).mean()



def main():

    logo_image = "./csem.webp"
    st.sidebar.image(logo_image, use_column_width=True)

    st.title('DROP TCR Peak Viewer')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, skiprows=20)
            processed_data = process_data(data)

            # Assuming 'data' is your DataFrame and 'CH1', 'CH2', 'CH4' are the columns you want to process
            #covert to mv
            CH1 = processed_data['CH1'] * (-1) * 1000
            CH2 = processed_data['CH2'] * (-1) * 1000
            CH4 = processed_data['CH4'] * (-1) * 1000

            # Sidebar widgets for moving average window size input
            window_size_CH1 = st.sidebar.number_input('Moving Average Window Size CH1', min_value=1, max_value=100, value=1)
            window_size_CH2 = st.sidebar.number_input('Moving Average Window Size CH2', min_value=1, max_value=100, value=1)
            window_size_CH4 = st.sidebar.number_input('Moving Average Window Size CH4', min_value=1, max_value=100, value=1)


            # Apply moving average filter to each channel with user-defined window size
            denoised_CH1 = moving_average_filter(CH1, window_size=window_size_CH1)
            denoised_CH2 = moving_average_filter(CH2, window_size=window_size_CH2)
            denoised_CH4 = moving_average_filter(CH4, window_size=window_size_CH4)

            # Shift the signal to zero level by subtracting the mean
            denoised_CH1 -= denoised_CH1.mean()
            denoised_CH2 -= denoised_CH2.mean()
            denoised_CH4 -= denoised_CH4.mean()


            # Sidebar widgets for peak height and width selection
            min_peak_height_CH1, max_peak_height_CH1 = st.sidebar.slider('Peak Height Range CH1 (mV)', 0.01, abs(max(denoised_CH1)), (0.01, np.max(denoised_CH1)), step=0.01)
            min_peak_height_CH2, max_peak_height_CH2 = st.sidebar.slider('Peak Height Range CH2 (mV)', 0.01, abs(max(denoised_CH2)), (0.01, np.max(denoised_CH2)), step=0.01)
            min_peak_height_CH4, max_peak_height_CH4 = st.sidebar.slider('Peak Height Range CH4 (mV)', 0.01, abs(max(denoised_CH4)), (0.01, np.max(denoised_CH4)), step=0.01)

            # denoised_CH1 = CH1
            # denoised_CH2 = CH2
            # denoised_CH4 = CH4

            # Find peaks in the denoised signals with user-defined parameters
            peaks_CH1, _ = find_peaks(denoised_CH1, height=(min_peak_height_CH1, max_peak_height_CH1))
            peaks_CH2, _ = find_peaks(denoised_CH2, height=(min_peak_height_CH2, max_peak_height_CH2))
            peaks_CH4, _ = find_peaks(denoised_CH4, height=(min_peak_height_CH4, max_peak_height_CH4))

            # Calculate histograms of peak heights
            hist_CH1, bin_edges_CH1 = np.histogram(denoised_CH1[peaks_CH1], bins='auto', density=False)
            hist_CH2, bin_edges_CH2 = np.histogram(denoised_CH2[peaks_CH2], bins='auto', density=False)
            hist_CH4, bin_edges_CH4 = np.histogram(denoised_CH4[peaks_CH4], bins='auto', density=False)

            # Create subplots
            fig = make_subplots(rows=3, cols=2, 
                                subplot_titles=("Peaks: " + str(len(peaks_CH1)), "Histogram: Peak heights", 
                                                "Peaks: " + str(len(peaks_CH2)), "Histogram: Peak heights", 
                                                "Peaks: " + str(len(peaks_CH4)), "Histogram: Peak heights"),
                                shared_xaxes=False, shared_yaxes=False)

            # Add traces to the subplots
            fig.add_trace(go.Scatter(x=np.arange(len(CH1)), y=CH1, mode='lines', name='Raw CH1', line=dict(color='skyblue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=np.arange(len(CH2)), y=CH2, mode='lines', name='Raw CH2', line=dict(color='lightgreen')), row=2, col=1)
            fig.add_trace(go.Scatter(x=np.arange(len(CH4)), y=CH4, mode='lines', name='Raw CH4', line=dict(color='pink')), row=3, col=1)

            fig.add_trace(go.Scatter(x=np.arange(len(denoised_CH1)), y=denoised_CH1, mode='lines', name='Filtered CH1', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=np.arange(len(denoised_CH2)), y=denoised_CH2, mode='lines', name='Filtered CH2', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=np.arange(len(denoised_CH4)), y=denoised_CH4, mode='lines', name='Filtered CH4', line=dict(color='purple')), row=3, col=1)

            fig.add_trace(go.Scatter(x=peaks_CH1, y=denoised_CH1[peaks_CH1], mode='markers', marker=dict(color='red'), name='Peaks CH1'), row=1, col=1)
            fig.add_trace(go.Scatter(x=peaks_CH2, y=denoised_CH2[peaks_CH2], mode='markers', marker=dict(color='red'), name='Peaks CH2'), row=2, col=1)
            fig.add_trace(go.Scatter(x=peaks_CH4, y=denoised_CH4[peaks_CH4], mode='markers', marker=dict(color='red'), name='Peaks CH4'), row=3, col=1)

            fig.add_trace(go.Bar(x=bin_edges_CH1[:-1], y=hist_CH1, name='Peak Heights CH1', marker=dict(color='blue')), row=1, col=2)
            fig.add_trace(go.Bar(x=bin_edges_CH2[:-1], y=hist_CH2, name='Peak Heights CH2', marker=dict(color='green')), row=2, col=2)
            fig.add_trace(go.Bar(x=bin_edges_CH4[:-1], y=hist_CH4, name='Peak Heights CH4', marker=dict(color='purple')), row=3, col=2)

            # Iterate through each subplot and update layout
            for i in range(1, 4):  # Assuming 3 subplots
                fig.update_xaxes(title_text="Time (ms)", row=i, col=1)
                fig.update_yaxes(title_text="Signal (mV)", row=i, col=1)
                fig.update_layout(height=700, width=900, showlegend=False)
    
                fig.update_xaxes(title_text="Count", row=i, col=2)
                fig.update_yaxes(title_text="Peak height (mV)", row=i, col=2)
            


            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == '__main__':
    main()
