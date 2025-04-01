# calculations.py

import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, filtfilt, ellip
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from .wrappers import SignalWrapper
from .config import *
import logging

logger = logging.getLogger(__name__)
logger.propagate = False

# Remove all handlers associated with the logger
if logger.hasHandlers():
    logger.handlers.clear()

# Create a new handler with your desired format
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Set the desired log level
logger.setLevel(LOGGING_LEVEL)

logger.disabled = LOGGING_STATE

class SignalAnalyzer:
    '''
    SignalAnalyzer Class
    --------------------

    The SignalAnalyzer class provides methods to analyze signals, including filtering, trend calculation,
    normalization, and curve fitting. It supports Gaussian and Half Gaussian fitting for signal peaks.

    Methods:
    - calculate_trend: Calculate the trend of a signal using a Savitzky-Golay filter.
    - normalize_signal: Normalize a signal between -1 and 1.
    - bandpass_filter: Apply a bandpass filter to a signal.
    - findpeaks: Find peaks in a signal.
    - findbottoms: Find valleys (negative peaks) in a signal.
    - fGuas: Define a Gaussian function.
    - halfgauss: Define a Half Gaussian function.
    - fit_halfgauss: Fit the Half Gaussian function to peaks in a signal.
    - fit_fGuas: Fit a Gaussian function to peaks in a signal.
    '''

    @staticmethod
    def trend(signal: np.ndarray, window_length: int = 1000, polyorder: int = 2) -> np.ndarray:
        '''
        Calculate Signal Trend
        ----------------------

        Calculate the trend of a signal using a Savitzky-Golay filter.

        Args:
        - signal (array-like): The input signal.
        - time (array-like): The time array corresponding to the signal.
        - window_length (int): The length of the filter window (default: 3000).
        - polyorder (int): The order of the polynomial used in the filter (default: 2).

        Returns:
        - array-like: The trend of the signal.
        '''
        trend = savgol_filter(signal, window_length=window_length, polyorder=polyorder)
        return trend


    @staticmethod
    def poly4func(x, a1, b1, a2, b2, c):
        # 4th degree polynomial
        return a1 * x**4 + b1 * x**3 + a2 * x**2 + b2 * x + c
    


    @staticmethod
    def poly4_trend(signal: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        Calculate the trend of a signal using a 4th degree polynomial model.

        Args:
            signal (np.ndarray): The input signal.
            time (np.ndarray): The time array corresponding to the signal.

        Returns:
            np.ndarray: The trend of the signal computed using the best-fit 4th degree polynomial model.
        """
        # Initial guess for the parameters [a1, b1, a2, b2, c]
        initial_guess = [1, 1, 1, 1, np.min(signal)]
        
        # For a polynomial, you typically don't need bounds, but they can be added if necessary.
        # Here we proceed without bounds.
        try:
            popt, pcov = curve_fit(SignalAnalyzer.poly4func, time, signal, p0=initial_guess)
        except RuntimeError as e:
            print("An error occurred during curve fitting:", e)
            return np.full_like(signal, np.nan)  # return an array of NaNs if fitting fails

        logger.debug(f"Fitted parameters: {popt}")
        trend = SignalAnalyzer.poly4func(time, *popt)
        return trend

    @staticmethod
    def exp2func(x, a1, b1, a2, b2, c):
        return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + c
    
    @staticmethod
    def exp_trend(signal: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        Calculate the trend of a signal using an exponential model.

        Args:
            signal (np.ndarray): The input signal.
            time (np.ndarray): The time array corresponding to the signal.

        Returns:
            np.ndarray: The trend of the signal computed using the best-fit exponential model.
        """

        # Initial guess for the parameters [a1, b1, a2, b2, c]
        initial_guess = [1, -0.1, 1, -0.01, np.min(signal)]  # Example initial guess
        param_bounds = (
            [0, -10, 0, -10, -np.inf],  # Lower bounds (b1, b2 constrained to be negative)
            [np.inf, 0, np.inf, 0, np.inf]  # Upper bounds (b1, b2 capped at 0 to prevent explosion)
        )
        # Fit the model to the data
        try:
            popt, pcov = curve_fit(SignalAnalyzer.exp2func, time, signal, p0=initial_guess, bounds=param_bounds)
        except RuntimeError as e:
            print("An error occurred during curve fitting:", e)
            return np.full_like(signal, np.nan)  # return an array of NaNs if fitting fails

        # Calculate the trend using the best-fit parameters
        logger.debug(f"Fitted parameters: {popt}")
        trend = SignalAnalyzer.exp2func(time, *popt)
        return trend
    
    @staticmethod
    def normalize_signal(signal: np.ndarray, target_range: tuple = (0,1)) -> np.ndarray:
        '''
        Normalize Signal
        ----------------

        Normalize a signal to a specified range.

        Args:
            signal (array-like): The input signal.
            target_range (tuple): The target range for normalization (default: (0, 1)).

        Returns:
            array-like: The normalized signal.
        '''
        a, b = target_range
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val) * (b - a) + a
        return normalized_signal
        # return (signal - np.min(signal)) / np.ptp(signal)

    @staticmethod
    def bandpass_filter(signal: np.ndarray, Fs: float = 250) -> np.ndarray:
        '''
        Bandpass Filter
        ---------------

        Apply a bandpass filter to the input signal.

        Args:
        - signal (array-like): The input signal to filter.
        - Fs (float): Sampling frequency of the signal in Hz.

        Returns:
        - array-like: The bandpass-filtered signal.
        '''
        low_pass_band = 0.6  # Hz
        high_pass_band = 4
        nyquist = Fs / 2
        order = 3
        low = low_pass_band / nyquist
        high = high_pass_band / nyquist
        b, a = ellip(order, 0.5, 20, [low, high], btype='bandpass')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    @staticmethod
    def findpeaks(signal: np.ndarray, threshold: float = 2, distance: int = 100) -> np.ndarray:
        '''
        Find Peaks
        ----------

        Find peaks in a signal.

        Args:
        - signal (array-like): The input signal to analyze.
        - threshold (float): Minimum height of peaks (default: 2).
        - distance (int): Minimum distance between peaks (default: 100).

        Returns:
        - array-like: Indices of the peaks in the signal.
        '''
        peaks, _ = find_peaks(signal, height=threshold, distance=distance)
        return peaks

    @staticmethod
    def findbottoms(signal: np.ndarray, threshold: float = 2, distance: int = 100) -> np.ndarray:
        '''
        Find Bottoms
        -----------

        Find valleys (negative peaks) in a signal.

        Args:
        - signal (array-like): The input signal to analyze.
        - threshold (float): Minimum depth of valleys (default: 2).
        - distance (int): Minimum distance between valleys (default: 100).

        Returns:
        - array-like: Indices of the valleys in the signal.
        '''
        bottoms, _ = find_peaks(-signal, height=threshold, distance=distance)
        return bottoms


    @staticmethod
    def halfgauss(x, A, B, s1, s2, sc1, sc2=1.0):
        '''
        Half Gaussian Function
        ----------------------

        Define a Half Gaussian function, where the behavior differs for values
        below and above the midpoint (B).

        Args:
        - x (array-like): The input array.
        - A (float): Amplitude of the Gaussian.
        - B (float): Midpoint of the Gaussian.
        - s1 (float): Width parameter for the left side of the Gaussian.
        - s2 (float): Width parameter for the right side of the Gaussian.
        - sc1 (float): Scaling factor for the left side.
        - sc2 (float): Scaling factor for the right side (default: 1.0).

        Returns:
        - array-like: Computed Half Gaussian values.
        '''
        y = np.zeros_like(x)
        for i in range(len(x)):
            if x[i] <= B:
                y[i] = (A * np.exp(-1 * abs((x[i] - B) / s1)**2)) * sc1 + (1 - sc1) * A
            else:
                y[i] = (A * np.exp(-1 * abs((x[i] - B) / s2)**2)) * sc2 + (1 - sc2) * A
        return y

    @staticmethod
    def fit_halfgauss(time, signal, peaks_indices, initial_guess):
        '''
        Fit Half Gaussian
        -----------------

        Fit the Half Gaussian function to peaks in a signal.

        Args:
        - time (array-like): Time values corresponding to the signal.
        - signal (array-like): Signal values.
        - peaks_indices (array-like): Indices of the peaks.
        - initial_guess (tuple): Initial guess for [A, B, s1, s2, sc1, sc2].

        Returns:
        - array: Optimal parameters for the Half Gaussian function.
        '''
        peak_times = time[peaks_indices]
        peak_values = signal[peaks_indices]

        try:
            popt, _ = curve_fit(
                SignalAnalyzer.halfgauss,
                peak_times,
                peak_values,
                p0=initial_guess
            )
            return popt
        except Exception as e:
            print(f"Error during curve fitting: {e}")
            return None

    @staticmethod
    def fGuas(x, A, mu, sigma):
        '''
        Gaussian Function
        -----------------

        Define a Gaussian function.

        Args:
            x (array-like): The input array.
            A (float): Amplitude of the Gaussian.
            mu (float): Mean (center) of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            array-like: Computed Gaussian values.
        '''
        return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

    @staticmethod
    def fit_fGuas(x: np.ndarray, y: np.ndarray, 
                       peaks_indices: np.ndarray, initial_guess: tuple):
        '''
        Fit Gaussian
        ------------

        Fit a Gaussian function to peaks in a signal.

        Args:
            x (array-like): x values corresponding to the signal.
            y (array-like): y values.
            peaks_indices (array-like): Indices of the peaks in the signal.
            initial_guess (tuple): Initial guess for Gaussian parameters (A, mu, sigma).

        Returns:
            array: Optimal Gaussian parameters (A, mu, sigma).
        '''
        peak_x = x[peaks_indices]
        peak_values = y[peaks_indices]

        try:
            popt, _ = curve_fit(SignalAnalyzer.fGuas, peak_x, peak_values, p0=initial_guess)
            return popt
        except Exception as e:
            print(f"Error during curve fitting: {e}")
            return None


    # Unit step function
    @staticmethod
    def u(x):
        '''
        Unit Step Function
        ------------------

        Define a unit step function.

        Args:
            x (array-like): The input array.

        Returns:
            array-like: Computed unit step values.
        '''
        
        return np.where(x >= 0, 1, 0)

    @staticmethod
    def g_P(P, alpha, beta, gamma):
        part1 = gamma * np.exp(P / alpha) * (-P / alpha + 1) * SignalAnalyzer.u(-P)
        part2 = gamma * np.exp(-P / beta) * (P / beta + 1) * SignalAnalyzer.u(P)
        return part1 + part2

    @staticmethod
    def oscillogram(P_e, P_s, P_d, alpha, beta, kgamma):
        '''
        Oscillogram Function
        --------------------

        Compute the oscillometric signal using the given parameters.

        Args:
            P_e (float): External pressure.
            P_s (float): Systolic pressure.
            P_d (float): Diastolic pressure.
            alpha (float): Decay parameter.
            beta (float): Rise parameter.
            kgamma (float): Gain parameter.

        Returns:
            float: Oscillometric signal value.
        '''
        term1 = kgamma * (
            (P_d - P_e + 2 * beta) * np.exp(-(P_d - P_e) / beta) -
            (P_s - P_e + 2 * beta) * np.exp(-(P_s - P_e) / beta)
        ) * SignalAnalyzer.u(P_d - P_e)
        
        term2 = kgamma * (
            2 * (alpha + beta) +
            (P_d - P_e - 2 * alpha) * np.exp((P_d - P_e) / alpha) -
            (P_s - P_e + 2 * beta) * np.exp(-(P_s - P_e) / beta)
        ) * (SignalAnalyzer.u(P_e - P_d) - SignalAnalyzer.u(P_e - P_s))
        
        term3 = kgamma * (
            (P_d - P_e - 2 * alpha) * np.exp((P_d - P_e) / alpha) -
            (P_s - P_e - 2 * alpha) * np.exp((P_s - P_e) / alpha)
        ) * SignalAnalyzer.u(P_e - P_s)
        
        return term1 + term2 + term3

    @staticmethod
    def fit_oscillogram(Pe: np.ndarray, delta_O: np.ndarray):

        '''
        Fit Oscillogram
        ---------------

        Fit the oscillogram function to the oscillometric signal.

        Args:
            Pe (array-like): External pressure values.
            delta_O (array-like): Oscillometric signal values.

        Returns:
            array: Optimal parameters for the oscillogram function.

        '''
        k_guess = np.max(delta_O) - np.min(delta_O)
        # Compute numerical derivative
        derivative = np.gradient(delta_O, Pe)
        
        # Estimate P_d and P_s from the derivative extrema
        P_d_guess = Pe[np.argmax(derivative)]
        P_s_guess = Pe[np.argmin(derivative)]
        
        # Ensure P_d_guess is less than P_s_guess; swap if necessary
        if P_d_guess > P_s_guess:
            P_d_guess, P_s_guess = P_s_guess, P_d_guess
        
        # Use the difference as a characteristic width for the transition
        width = abs(P_s_guess - P_d_guess)
        
        # Use a fraction of the width as initial guesses for shape parameters
        alpha_guess = width / 2.0
        beta_guess = width / 2.0

        initial_guess = [P_s_guess, P_d_guess, alpha_guess, beta_guess, k_guess]
        # initial_guess = [0.35, 0.65, 0.04, 0.09, 15]

        try:
            popt, _ = curve_fit(SignalAnalyzer.oscillogram, Pe, delta_O, p0=initial_guess)
            return popt
        except Exception as e:
            print(f"Error during curve fitting: {e}")
            return None


    @staticmethod
    def estimatebp(time: np.ndarray, signal: np.ndarray):
        '''
        Estimate the blood pressure value from the oscillometric signal

        Steps:
        1. find the trend of the signal using exponential trend -> we call this variable "Pe" meaning external pressure
        2. normalize the trend to the range [0, 1] -> we call this variable "norm_trend"
        3. copmute the filtered signal using bandpass filter -> we call this variable "filtered_signal"
        4. find the peaks of the filtered signal -> we call this variable "peaks"
        5. find the bottoms of the filtered signal -> we call this variable "bottoms"
        6. fit the half gaussian function to the peaks and bottoms vs Pe-> we call this variable "fit_peaks" and "fit_bottoms"
        7. substract the fit_peaks and fit_bottoms from the filtered signal -> we call this variable "DO"
        8. find the derivative of the DO signal -> we call this variable "dDO"
        9. find where the maximum of the dDO signal occurs -> we call the corresponding Pe value "Pemax"
        10. find the minimum of the dDO signal -> we call the corresponding Pe value "Pemin"
        11. plug in the Pemax and Pemin into the following equation to get the systolic and diastolic blood pressure values:
            SBP = Pemin + 0.33 * (Pemin - Pemax)
            DBP = Pemax - 0.33 * (Pemin - Pemax)
        12. return the SBP and DBP values as a tuple
        ----------------
        Args:
        - signal (array-like): The input oscillometric signal. The cropped signal is used for the analysis.
        - time (array-like): Time values corresponding to the signal.


        returns:
        - tuple: Systolic and diastolic blood pressure values.
        '''

        # Step 1: Calculate the trend of the signal using exponential trend
        trend, _ = SignalAnalyzer.exp_trend(signal, time)

        # Step 2: Normalize the trend to the range [0, 1]
        norm_trend = SignalAnalyzer.normalize_signal(trend)
        # Interpolate the trend to match the time array of the filtered signal
        trend_interpolator = interp1d(time, norm_trend, kind='linear', fill_value='extrapolate')
        Pe_norm = trend_interpolator(time)
        Pe_norm = Pe_norm[::-1] # flip the trend because we are starting from highest pressure to lowest pressure

        # Step 3: Compute the filtered signal using bandpass filter
        filtered_signal = SignalAnalyzer.bandpass_filter(signal, 250)
        O = filtered_signal[::-1] # flip the filtered signal because we are starting from highest pressure to lowest pressure

        # Step 4: Find the peaks of the filtered signal
        peak_indices = SignalAnalyzer.findpeaks(O, threshold=0.5, distance=150)
        peaks = O[peak_indices]
        peaks_Pe = Pe_norm[peak_indices]

        # Step 5: Find the bottoms of the filtered signal
        bottom_indices = SignalAnalyzer.findbottoms(O, threshold=0.5, distance=150)
        bottoms = O[bottom_indices]
        bottoms_Pe = Pe_norm[bottom_indices]


        # Step 6: Fit the half gaussian function to the peaks and bottoms vs Pe

        DO = np.zeros(len(O))
        peak_val = np.max(peaks)             # estimate for amplitude
        peak_time = peaks_Pe[np.argmax(peaks)]
        time_range = np.max(peaks_Pe) - np.min(peaks_Pe)

        # plt.figure(figsize=(10, 5))

        # Example refined guess
        initial_guess = [
            peak_val,                   # A
            peak_time,                  # B
            time_range * 0.1,          # s1
            time_range * 0.1,          # s2
            0.8,                       # sc1 (example)
            1.0                        # sc2 (example)
        ]

        # Fit the Half Gaussian function
        halfgauss_params = SignalAnalyzer.fit_halfgauss(Pe_norm, O, peak_indices, initial_guess)
        # Plot the fitted Half Gaussian function
        if halfgauss_params is not None:
            fitted_curve = SignalAnalyzer.halfgauss(Pe_norm, *halfgauss_params)
            DO = fitted_curve
            # plt.plot(Pe_norm, fitted_curve, label='Fitted Half Gaussian')


        bottom_val = np.min(bottoms)             # estimate for amplitude
        bottom_time = bottoms_Pe[np.argmin(bottoms)]
        time_range = np.max(bottoms_Pe) - np.min(bottoms_Pe)

        # Example refined guess
        initial_guess = [
            bottom_val,                   # A
            bottom_time,                  # B
            time_range * 0.1,          # s1
            time_range * 0.1,          # s2
            0.8,                       # sc1 (example)
            1.0                        # sc2 (example)
        ]

        halfgauss_params_bottom = SignalAnalyzer.fit_halfgauss(Pe_norm, -O, bottom_indices, initial_guess)
        # Plot the fitted Half Gaussian function
        if halfgauss_params_bottom is not None:
            fitted_curve = SignalAnalyzer.halfgauss(Pe_norm, *halfgauss_params_bottom)
            DO = DO + fitted_curve
            # plt.plot(Pe_norm, -fitted_curve, color = 'black' , label='Fitted Half Gaussian Bottom')

        # Step 8: Find the derivative of the DO signal
        ddOdT = np.gradient(DO, Pe_norm)

        # Step 9: Find where the maximum of the dDO signal occurs
        Pemax = Pe_norm[np.argmax(ddOdT)]

        # Step 10: Find the minimum of the dDO signal
        Pemin = Pe_norm[np.argmin(ddOdT)]

        # Step 11: Calculate the systolic and diastolic blood pressure values
        alpha = 0.1
        beta = 0.2
        SBP = Pemin + alpha * (Pemin - Pemax)
        DBP = Pemax - beta * (Pemin - Pemax)

        SBP = Pemin
        DBP = Pemax


        # plt.plot(Pe_norm, O)
        # plt.scatter(bottoms_Pe, bottoms, color='red', label='Bottoms')
        # plt.scatter(peaks_Pe, peaks, color='green', label='Peaks')
        # plt.plot(Pe_norm, DO, label='$\\Delta O$')
        # # plt.plot(Pe_norm, 0.05*ddOdT, label='$\\frac{d\\Delta O}{dT}$')
        # plt.xlabel('Normalized Strain ($\\mu V$)', fontsize=FONTSIZE1)
        # plt.ylabel('Amplitude ($\\mu V$)', fontsize=FONTSIZE1)
        # plt.title("Oscillation vs Pressure", fontsize=FONTSIZE2)

        # plt.legend(fontsize = 20)
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.show()

        return SBP, DBP

    @staticmethod
    def estimatebp_figure(time: np.ndarray, signal: np.ndarray):
        '''
        Estimate the blood pressure value from the oscillometric signal

        Steps:
        1. find the trend of the signal using exponential trend -> we call this variable "Pe" meaning external pressure
        2. normalize the trend to the range [0, 1] -> we call this variable "norm_trend"
        3. copmute the filtered signal using bandpass filter -> we call this variable "filtered_signal"
        4. find the peaks of the filtered signal -> we call this variable "peaks"
        5. find the bottoms of the filtered signal -> we call this variable "bottoms"
        6. fit the half gaussian function to the peaks and bottoms vs Pe-> we call this variable "fit_peaks" and "fit_bottoms"
        7. substract the fit_peaks and fit_bottoms from the filtered signal -> we call this variable "DO"
        8. find the derivative of the DO signal -> we call this variable "dDO"
        9. find where the maximum of the dDO signal occurs -> we call the corresponding Pe value "Pemax"
        10. find the minimum of the dDO signal -> we call the corresponding Pe value "Pemin"
        11. plug in the Pemax and Pemin into the following equation to get the systolic and diastolic blood pressure values:
            SBP = Pemin + 0.33 * (Pemin - Pemax)
            DBP = Pemax - 0.33 * (Pemin - Pemax)
        12. return the SBP and DBP values as a tuple
        ----------------
        Args:
        - signal (array-like): The input oscillometric signal. The cropped signal is used for the analysis.
        - time (array-like): Time values corresponding to the signal.


        returns:
        - tuple: Systolic and diastolic blood pressure values.
        '''

        # Step 1: Calculate the trend of the signal using exponential trend
        trend, _ = SignalAnalyzer.exp_trend(signal, time)

        # Step 2: Normalize the trend to the range [0, 1]
        norm_trend = SignalAnalyzer.normalize_signal(trend)
        # Interpolate the trend to match the time array of the filtered signal
        trend_interpolator = interp1d(time, norm_trend, kind='linear', fill_value='extrapolate')
        Pe_norm = trend_interpolator(time)
        Pe_norm = Pe_norm[::-1] # flip the trend because we are starting from highest pressure to lowest pressure

        # Step 3: Compute the filtered signal using bandpass filter
        filtered_signal = SignalAnalyzer.bandpass_filter(signal, 250)
        O = filtered_signal[::-1] # flip the filtered signal because we are starting from highest pressure to lowest pressure

        # Step 4: Find the peaks of the filtered signal
        peak_indices = SignalAnalyzer.findpeaks(O, threshold=0.5, distance=150)
        peaks = O[peak_indices]
        peaks_Pe = Pe_norm[peak_indices]

        # Step 5: Find the bottoms of the filtered signal
        bottom_indices = SignalAnalyzer.findbottoms(O, threshold=0.5, distance=150)
        bottoms = O[bottom_indices]
        bottoms_Pe = Pe_norm[bottom_indices]


        # Step 6: Fit the half gaussian function to the peaks and bottoms vs Pe

        DO = np.zeros(len(O))
        peak_val = np.max(peaks)             # estimate for amplitude
        peak_time = peaks_Pe[np.argmax(peaks)]
        time_range = np.max(peaks_Pe) - np.min(peaks_Pe)

        plt.figure(figsize=(10, 5))

        # Example refined guess
        initial_guess = [
            peak_val,                   # A
            peak_time,                  # B
            time_range * 0.1,          # s1
            time_range * 0.1,          # s2
            0.8,                       # sc1 (example)
            1.0                        # sc2 (example)
        ]

        # Fit the Half Gaussian function
        halfgauss_params = SignalAnalyzer.fit_halfgauss(Pe_norm, O, peak_indices, initial_guess)
        # Plot the fitted Half Gaussian function
        if halfgauss_params is not None:
            fitted_curve = SignalAnalyzer.halfgauss(Pe_norm, *halfgauss_params)
            DO = fitted_curve
            plt.plot(Pe_norm, fitted_curve, label='Fitted Half Gaussian')


        bottom_val = np.min(bottoms)             # estimate for amplitude
        bottom_time = bottoms_Pe[np.argmin(bottoms)]
        time_range = np.max(bottoms_Pe) - np.min(bottoms_Pe)

        # Example refined guess
        initial_guess = [
            bottom_val,                   # A
            bottom_time,                  # B
            time_range * 0.1,          # s1
            time_range * 0.1,          # s2
            0.8,                       # sc1 (example)
            1.0                        # sc2 (example)
        ]

        halfgauss_params_bottom = SignalAnalyzer.fit_halfgauss(Pe_norm, -O, bottom_indices, initial_guess)
        # Plot the fitted Half Gaussian function
        if halfgauss_params_bottom is not None:
            fitted_curve = SignalAnalyzer.halfgauss(Pe_norm, *halfgauss_params_bottom)
            DO = DO + fitted_curve
            plt.plot(Pe_norm, -fitted_curve, color = 'black' , label='Fitted Half Gaussian Bottom')

        # Step 8: Find the derivative of the DO signal
        ddOdT = np.gradient(DO, Pe_norm)

        # Step 9: Find where the maximum of the dDO signal occurs
        Pemax = Pe_norm[np.argmax(ddOdT)]

        # Step 10: Find the minimum of the dDO signal
        Pemin = Pe_norm[np.argmin(ddOdT)]

        # Step 11: Calculate the systolic and diastolic blood pressure values
        alpha = 0.1
        beta = 0.2
        SBP = Pemin + alpha * (Pemin - Pemax)
        DBP = Pemax - beta * (Pemin - Pemax)

        SBP = Pemin
        DBP = Pemax


        plt.plot(Pe_norm, O)
        plt.scatter(bottoms_Pe, bottoms, color='red', label='Bottoms')
        plt.scatter(peaks_Pe, peaks, color='green', label='Peaks')
        plt.plot(Pe_norm, DO, label='$\\Delta O$')
        plt.plot(Pe_norm, 0.05*ddOdT, label='$\\frac{d\\Delta O}{dT}$')
        plt.xlabel('Normalized Strain ($\\mu V$)', fontsize=FONTSIZE1)
        plt.ylabel('Amplitude ($\\mu V$)', fontsize=FONTSIZE1)
        plt.title("Oscillation vs Pressure", fontsize=FONTSIZE2)

        plt.legend(fontsize = 20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        return SBP, DBP
    
    @staticmethod
    def estimate_dbp(time: np.ndarray, raw_signal: np.ndarray, id=1) -> tuple:
        '''
        Find Diastolic Blood Pressure (DBP)
        -----------------

        Find the diastolic blood pressure value from the oscillometric signal.

        Args:
            time (array-like): Time values corresponding to the signal.
            raw_signal (array-like): Signal values.

        Returns:
            float: Diastolic blood pressure value.
        '''

# Find the Region of Interest
        # Step1: Find the oscillation by filtering the signal
        if id>=17:
            fs = 250
        else:
            fs = 250

        filtered_signal = SignalAnalyzer.bandpass_filter(raw_signal, fs)

        # Step2: normalize the filtered signal
        # filtered_signal = SignalAnalyzer.normalize_signal(filtered_signal, target_range=(-1, 1))
        # filtered_signal = filtered_signal - np.mean(filtered_signal)

        # Step2: Find the peaks of the filtered signal
        peak_indices = SignalAnalyzer.findpeaks(filtered_signal, threshold=0.15 * np.max(filtered_signal), distance=70)
        # print(f"Number of peaks: {len(peak_indices)}")
        # print(f"Number of peaks: {len(peak_indices)}")
        # print(f"Peak indices: {peak_indices}")
        # print(f"Peak values: {peak_indices - peak_indices[0]}")
        # Step 3: crop the signal to the region of interest
        a, b = peak_indices[0], peak_indices[-1]
        b = b + 1
        # a = 0
        # b = len(filtered_signal) - 1
        # print(a, b)
        # a = 3 * 250
        # b = b + 250 * 10
        signal = raw_signal[a:b]

        # Step 4: Find the trend of the signal using exponential trend
        trend_mvavg = SignalAnalyzer.trend(signal)
        trend, _ = SignalAnalyzer.exp_trend(trend_mvavg, time[a:b])
        trend_intrp = interp1d(time[a:b], trend, kind='linear', fill_value='extrapolate')
        trend = trend_intrp(time[a:b])
        Pe = trend[::-1]
        Pe_norm = SignalAnalyzer.normalize_signal(Pe)
        # print(f"Pe shape: {Pe.shape}")
        O = filtered_signal[a:b][::-1]

        peak_indices_O = SignalAnalyzer.findpeaks(O, threshold=0.15 * np.max(O), distance=150)
        peaks_O = O[peak_indices_O]
        peaks_Pe = Pe_norm[peak_indices_O]
        bottom_indices_O = SignalAnalyzer.findbottoms(O, threshold=0.15 * np.max(O), distance=150)
        bottoms_O = O[bottom_indices_O]
        bottoms_Pe = Pe_norm[bottom_indices_O]
        # Step 5: Find the bottoms of the filtered signal
        # bottom_indices = SignalAnalyzer.findbottoms(filtered_signal, threshold=0.2, distance=150)
        # bottoms = O[bottom_indices - a]
        # bottoms_Pe = Pe_norm[bottom_indices - a]
        # peaks = O[peak_indices - a]
        # peaks_Pe = Pe_norm[peak_indices - a]


        # Step 6: Fit the half gaussian function to the peaks and bottoms vs Pe
        DO = np.zeros(len(O))


        #### Fit the Half Gaussian function to the peaks and bottoms vs Pe

        # peak_val = np.max(peaks_O)             # estimate for amplitude
        # peak_time = peaks_Pe[np.argmax(peaks_O)]
        # time_range = np.max(peaks_Pe) - np.min(peaks_Pe)

        # # Example refined guess
        # initial_guess = [
        #     peak_val,                   # A
        #     peak_time,                  # B
        #     time_range * 0.1,          # s1
        #     time_range * 0.1,          # s2
        #     0.8,                       # sc1 (example)
        #     1.0                        # sc2 (example)
        # ]


        # # Fit the Half Gaussian function
        # halfgauss_params = SignalAnalyzer.fit_halfgauss(Pe_norm, O, peak_indices_O, initial_guess)
        # fitted_curve_peaks = SignalAnalyzer.halfgauss(Pe_norm, *halfgauss_params)

        # bottom_val = np.min(bottoms_O)             # estimate for amplitude
        # bottom_time = bottoms_Pe[np.argmin(bottoms_O)]
        # time_range = np.max(bottoms_Pe) - np.min(bottoms_Pe)

        # # Example refined guess
        # initial_guess = [
        #     bottom_val,                   # A
        #     bottom_time,                  # B
        #     time_range * 0.1,          # s1
        #     time_range * 0.1,          # s2
        #     0.8,                       # sc1 (example)
        #     1.0                        # sc2 (example)
        # ]

        # halfgauss_params_bottom = SignalAnalyzer.fit_halfgauss(Pe_norm, -O, bottom_indices_O, initial_guess)
        # fitted_curve_bottoms = SignalAnalyzer.halfgauss(Pe_norm, *halfgauss_params_bottom)

        # DO = fitted_curve_peaks + fitted_curve_bottoms

        
        #### End of fitting the Half Gaussian function


        #### Fitting the Gaussian function to the peaks and bottoms vs Pe

        # Overlay fitted Gaussian
        # Fit Gaussian function
        initial_guess = (1, Pe_norm[peak_indices_O[len(peak_indices_O) // 2]], 1)  # A=1, mu=middle peak, sigma=1
        gaussian_params = SignalAnalyzer.fit_fGuas(Pe_norm, O, peak_indices_O, initial_guess)

        # Fit gaussian to the bottoms
        initial_guess = (1, Pe_norm[bottom_indices_O[len(bottom_indices_O) // 2]], 1)  # A=1, mu=middle peak, sigma=1
        gaussian_params_bottom = SignalAnalyzer.fit_fGuas(Pe_norm, O, bottom_indices_O, initial_guess)
        print(f"Gaussian params: {gaussian_params}")
        print(f"Gaussian params bottom: {gaussian_params_bottom}")
        print(f"Peaks: {peaks_O}")
        print(f"Bottoms: {bottoms_O}")
        if gaussian_params is not None:
            A, mu, sigma = gaussian_params
            gaussian_fit_p = SignalAnalyzer.fGuas(Pe_norm, A, mu, sigma)
            DO = gaussian_fit_p
            # plt.plot(x, gaussian_fit, color='orange', label=f'Gaussian Fit: A={A:.2f}, μ={mu:.2f}, σ={sigma:.2f}')

        # if gaussian_params_bottom is not None:
        A, mu, sigma = gaussian_params_bottom
        gaussian_fit_b = SignalAnalyzer.fGuas(Pe_norm, A, mu, sigma)
        DO = DO - gaussian_fit_b

            # plt.plot(x, gaussian_fit, color='green', label=f'Gaussian Fit: A={A:.2f}, μ={mu:.2f}, σ={sigma:.2f}')

        # Step 8: Find the derivative of the DO signal
        dDOdP = np.gradient(DO, Pe_norm)



        #### End of fitting the Gaussian function

        # Step 9: Find where the maximum of the dDO signal occurs
        Pemax = Pe_norm[np.argmax(dDOdP)]

        # Step 10: Find the minimum of the dDO signal
        Pemin = Pe_norm[np.argmin(dDOdP)]

        # Step 11: Calculate the systolic and diastolic blood pressure values
        # print(f"Pemax: {Pemax}, Pemin: {Pemin}")

        

        plt.figure(figsize=(10, 5))
        plt.plot(time, raw_signal)
        plt.scatter(time[peak_indices], raw_signal[peak_indices], color='red', label='Peaks')
        plt.plot(time[a:b], raw_signal[a:b], label='Trend', color='black')
        plt.plot(time[a:b], trend, label='Trend', color='green')
        #overlay the start point
        plt.scatter(time[peak_indices[1]], trend[peak_indices[1] - a], color='purple', label='Start', marker='*', s=100, zorder=10)
        plt.xlabel('Time (s)', fontsize=FONTSIZE1)
        plt.ylabel('Amplitude ($\\mu V$)', fontsize=FONTSIZE1)
        plt.title("Oscillation vs Time", fontsize=FONTSIZE2)
        plt.legend(fontsize = 20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(Pe_norm, O)
        plt.scatter(Pe_norm[peak_indices_O], O[peak_indices_O], color='red', label='Peaks')
        plt.scatter(Pe_norm[bottom_indices_O], O[bottom_indices_O], color='green', label='Bottoms')
        plt.plot(Pe_norm, gaussian_fit_p, label='Fitted Half Gaussian')
        plt.plot(Pe_norm, gaussian_fit_b, color = 'black' , label='Fitted Half Gaussian Bottom')
        plt.plot(Pe_norm, DO, label='$\\Delta O$')
        plt.plot(Pe_norm, 0.1*dDOdP, label='$\\frac{d\\Delta O}{dP}$')
        plt.xlabel('Normalized Strain ($\\mu V$)', fontsize=FONTSIZE1)
        plt.ylabel('Amplitude ($\\mu V$)', fontsize=FONTSIZE1)
        plt.title("Oscillation vs Pressure", fontsize=FONTSIZE2)
        plt.legend(fontsize = 15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(time, filtered_signal)
        plt.scatter(time[peak_indices], filtered_signal[peak_indices], color='red', label='Peaks')
        plt.xlabel('Time (s)', fontsize=FONTSIZE1)
        plt.ylabel('Amplitude ($\\mu V$)', fontsize=FONTSIZE1)
        plt.title("Oscillation vs Time", fontsize=FONTSIZE2)
        plt.legend(fontsize = 20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        return Pemin, Pemax

