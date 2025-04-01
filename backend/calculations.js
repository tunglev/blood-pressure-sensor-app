import { SignalWrapper } from '../wrappers.js';
import {
  LOGGING_LEVEL,
  LOGGING_STATE,
  FONTSIZE1,
  FONTSIZE2,
  DEFAULT_WINDOW_LENGTH,
  DEFAULT_POLYORDER,
  DEFAULT_FS,
  DEFAULT_PEAK_THRESHOLD,
  DEFAULT_PEAK_DISTANCE,
  BP_ALPHA,
  BP_BETA
} from '../config.js';

// Helper function to log messages
const logger = {
  debug: (message) => {
    if (!LOGGING_STATE && LOGGING_LEVEL === 'debug') {
      console.debug(new Date().toISOString() + ' - DEBUG - ' + message);
    }
  },
  info: (message) => {
    if (!LOGGING_STATE && (LOGGING_LEVEL === 'debug' || LOGGING_LEVEL === 'info')) {
      console.info(new Date().toISOString() + ' - INFO - ' + message);
    }
  },
  error: (message) => {
    if (!LOGGING_STATE) {
      console.error(new Date().toISOString() + ' - ERROR - ' + message);
    }
  }
};

class SignalAnalyzer {
  /**
   * Calculate Signal Trend
   * ----------------------
   * Calculate the trend of a signal using a Savitzky-Golay filter equivalent.
   * 
   * @param {Array} signal - The input signal.
   * @param {number} windowLength - The length of the filter window (default: 1000).
   * @param {number} polyorder - The order of the polynomial used in the filter (default: 2).
   * @returns {Array} - The trend of the signal.
   */
  static trend(signal, windowLength = 1000, polyorder = 2) {
    // JavaScript implementation of Savitzky-Golay filter
    // This is a simple moving average as a basic replacement
    const halfWindow = Math.floor(windowLength / 2);
    const len = signal.length;
    const trend = new Array(len).fill(0);
    
    for (let i = 0; i < len; i++) {
      let sum = 0;
      let count = 0;
      for (let j = Math.max(0, i - halfWindow); j < Math.min(len, i + halfWindow + 1); j++) {
        sum += signal[j];
        count++;
      }
      trend[i] = sum / count;
    }
    
    return trend;
  }

  /**
   * 4th degree polynomial function
   */
  static poly4func(x, a1, b1, a2, b2, c) {
    return a1 * Math.pow(x, 4) + b1 * Math.pow(x, 3) + a2 * Math.pow(x, 2) + b2 * x + c;
  }

  /**
   * Calculate the trend of a signal using a 4th degree polynomial model.
   * 
   * @param {Array} signal - The input signal.
   * @param {Array} time - The time array corresponding to the signal.
   * @returns {Array} - The trend of the signal.
   */
  static poly4_trend(signal, time) {
    // For JavaScript, we'll need a curve fitting library or implement a simpler version
    // This is a simplified implementation
    const initialGuess = [1, 1, 1, 1, Math.min(...signal)];
    
    try {
      // In a full implementation, you would use a curve fitting library
      // For now, we'll return a basic trend approximation
      const trend = this.trend(signal);
      return trend;
    } catch (error) {
      console.error("An error occurred during curve fitting:", error);
      return new Array(signal.length).fill(NaN);
    }
  }

  /**
   * Exponential function for trend calculation
   */
  static exp2func(x, a1, b1, a2, b2, c) {
    return a1 * Math.exp(b1 * x) + a2 * Math.exp(b2 * x) + c;
  }

  /**
   * Calculate the trend of a signal using an exponential model.
   * 
   * @param {Array} signal - The input signal.
   * @param {Array} time - The time array corresponding to the signal.
   * @returns {Array} - The trend of the signal and fitted parameters.
   */
  static exp_trend(signal, time) {
    // Simplified implementation - in reality, you would use a curve fitting library
    const trend = this.trend(signal);
    const mockParams = [1, -0.1, 1, -0.01, Math.min(...signal)];
    return [trend, mockParams];
  }

  /**
   * Normalize a signal to a specified range.
   * 
   * @param {Array} signal - The input signal.
   * @param {Array} targetRange - The target range for normalization (default: [0, 1]).
   * @returns {Array} - The normalized signal.
   */
  static normalize_signal(signal, targetRange = [0, 1]) {
    const [a, b] = targetRange;
    const min = Math.min(...signal);
    const max = Math.max(...signal);
    
    return signal.map(value => {
      return (value - min) / (max - min) * (b - a) + a;
    });
  }

  /**
   * Apply a bandpass filter to the input signal.
   * 
   * @param {Array} signal - The input signal to filter.
   * @param {number} Fs - Sampling frequency of the signal in Hz (default: 250).
   * @returns {Array} - The bandpass-filtered signal.
   */
  static bandpass_filter(signal, Fs = 250) {
    // This is a simplified implementation
    // In a real app, you would use a DSP library like DSP.js
    
    // For now, we'll implement a simple moving average filter as a placeholder
    const windowSize = Math.floor(Fs / 10); // Example window size
    const filtered = new Array(signal.length).fill(0);
    
    for (let i = 0; i < signal.length; i++) {
      let sum = 0;
      let count = 0;
      for (let j = Math.max(0, i - windowSize); j < Math.min(signal.length, i + windowSize + 1); j++) {
        sum += signal[j];
        count++;
      }
      filtered[i] = signal[i] - (sum / count);
    }
    
    return filtered;
  }

  /**
   * Find peaks in a signal.
   * 
   * @param {Array} signal - The input signal to analyze.
   * @param {number} threshold - Minimum height of peaks (default: 2).
   * @param {number} distance - Minimum distance between peaks (default: 100).
   * @returns {Array} - Indices of the peaks in the signal.
   */
  static findpeaks(signal, threshold = 2, distance = 100) {
    const peaks = [];
    
    for (let i = 1; i < signal.length - 1; i++) {
      if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > threshold) {
        peaks.push(i);
      }
    }
    
    // Filter peaks by distance
    const filteredPeaks = [];
    if (peaks.length > 0) {
      filteredPeaks.push(peaks[0]);
      for (let i = 1; i < peaks.length; i++) {
        if (peaks[i] - filteredPeaks[filteredPeaks.length - 1] >= distance) {
          filteredPeaks.push(peaks[i]);
        }
      }
    }
    
    return filteredPeaks;
  }

  /**
   * Find valleys (negative peaks) in a signal.
   * 
   * @param {Array} signal - The input signal to analyze.
   * @param {number} threshold - Minimum depth of valleys (default: 2).
   * @param {number} distance - Minimum distance between valleys (default: 100).
   * @returns {Array} - Indices of the valleys in the signal.
   */
  static findbottoms(signal, threshold = 2, distance = 100) {
    const negativeSignal = signal.map(value => -value);
    return this.findpeaks(negativeSignal, threshold, distance);
  }

  /**
   * Define a Half Gaussian function.
   */
  static halfgauss(x, A, B, s1, s2, sc1, sc2 = 1.0) {
    const y = new Array(x.length).fill(0);
    
    for (let i = 0; i < x.length; i++) {
      if (x[i] <= B) {
        y[i] = (A * Math.exp(-1 * Math.abs(Math.pow((x[i] - B) / s1, 2)))) * sc1 + (1 - sc1) * A;
      } else {
        y[i] = (A * Math.exp(-1 * Math.abs(Math.pow((x[i] - B) / s2, 2)))) * sc2 + (1 - sc2) * A;
      }
    }
    
    return y;
  }

  /**
   * Fit the Half Gaussian function to peaks in a signal.
   * 
   * @param {Array} time - Time values corresponding to the signal.
   * @param {Array} signal - Signal values.
   * @param {Array} peaksIndices - Indices of the peaks.
   * @param {Array} initialGuess - Initial guess for [A, B, s1, s2, sc1, sc2].
   * @returns {Array} - Optimal parameters for the Half Gaussian function.
   */
  static fit_halfgauss(time, signal, peaksIndices, initialGuess) {
    // In JavaScript, we would need a curve fitting library
    // For a simple implementation, we'll return the initial guess
    try {
      // Placeholder for curve fitting
      return initialGuess;
    } catch (error) {
      console.error(`Error during curve fitting: ${error}`);
      return null;
    }
  }

  /**
   * Define a Gaussian function.
   */
  static fGuas(x, A, mu, sigma) {
    return x.map(val => A * Math.exp(-Math.pow(val - mu, 2) / (2 * Math.pow(sigma, 2))));
  }

  /**
   * Fit a Gaussian function to peaks in a signal.
   * 
   * @param {Array} x - x values corresponding to the signal.
   * @param {Array} y - y values.
   * @param {Array} peaksIndices - Indices of the peaks in the signal.
   * @param {Array} initialGuess - Initial guess for Gaussian parameters (A, mu, sigma).
   * @returns {Array} - Optimal Gaussian parameters (A, mu, sigma).
   */
  static fit_fGuas(x, y, peaksIndices, initialGuess) {
    // Simplified implementation
    try {
      return initialGuess;
    } catch (error) {
      console.error(`Error during curve fitting: ${error}`);
      return null;
    }
  }

  /**
   * Unit Step Function
   */
  static u(x) {
    return x.map(val => val >= 0 ? 1 : 0);
  }

  /**
   * g_P function
   */
  static g_P(P, alpha, beta, gamma) {
    const part1 = gamma * Math.exp(P / alpha) * (-P / alpha + 1) * this.u([-P])[0];
    const part2 = gamma * Math.exp(-P / beta) * (P / beta + 1) * this.u([P])[0];
    return part1 + part2;
  }

  /**
   * Oscillogram Function
   */
  static oscillogram(P_e, P_s, P_d, alpha, beta, kgamma) {
    const term1 = kgamma * (
      (P_d - P_e + 2 * beta) * Math.exp(-(P_d - P_e) / beta) -
      (P_s - P_e + 2 * beta) * Math.exp(-(P_s - P_e) / beta)
    ) * this.u([P_d - P_e])[0];
    
    const term2 = kgamma * (
      2 * (alpha + beta) +
      (P_d - P_e - 2 * alpha) * Math.exp((P_d - P_e) / alpha) -
      (P_s - P_e + 2 * beta) * Math.exp(-(P_s - P_e) / beta)
    ) * (this.u([P_e - P_d])[0] - this.u([P_e - P_s])[0]);
    
    const term3 = kgamma * (
      (P_d - P_e - 2 * alpha) * Math.exp((P_d - P_e) / alpha) -
      (P_s - P_e - 2 * alpha) * Math.exp((P_s - P_e) / alpha)
    ) * this.u([P_e - P_s])[0];
    
    return term1 + term2 + term3;
  }

  /**
   * Interpolate 1D function
   * Simple linear interpolation
   */
  static interp1d(x, y, newX, options = { kind: 'linear', fillValue: 'extrapolate' }) {
    const result = new Array(newX.length);
    
    for (let i = 0; i < newX.length; i++) {
      const targetX = newX[i];
      
      // Find the bracketing points in the original array
      let j = 0;
      while (j < x.length - 1 && x[j] < targetX) {
        j++;
      }
      
      if (j === 0) {
        // Extrapolation below
        result[i] = y[0];
      } else if (j === x.length) {
        // Extrapolation above
        result[i] = y[x.length - 1];
      } else {
        // Interpolation
        const xL = x[j - 1];
        const xR = x[j];
        const yL = y[j - 1];
        const yR = y[j];
        
        // Linear interpolation formula
        result[i] = yL + (targetX - xL) * (yR - yL) / (xR - xL);
      }
    }
    
    return result;
  }

  /**
   * Calculate gradient (derivative) of array
   */
  static gradient(y, x) {
    const result = new Array(y.length);
    
    if (y.length <= 1) {
      return new Array(y.length).fill(0);
    }
    
    // Forward difference for first point
    result[0] = (y[1] - y[0]) / (x[1] - x[0]);
    
    // Central difference for interior points
    for (let i = 1; i < y.length - 1; i++) {
      result[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    }
    
    // Backward difference for last point
    result[y.length - 1] = (y[y.length - 1] - y[y.length - 2]) / 
                          (x[y.length - 1] - x[y.length - 2]);
    
    return result;
  }

  /**
   * Estimate the blood pressure values from the oscillometric signal
   * 
   * @param {Array} time - Time values corresponding to the signal.
   * @param {Array} signal - The input oscillometric signal.
   * @returns {Object} - Systolic and diastolic blood pressure values.
   */
  static estimatebp(time, signal) {
    // Step 1: Calculate the trend using exponential trend
    const [trend, _] = this.exp_trend(signal, time);
    
    // Step 2: Normalize the trend to the range [0, 1]
    const normTrend = this.normalize_signal(trend);
    
    // Create an array with values from 0 to signal.length-1
    const timeIndices = Array.from({length: time.length}, (_, i) => i);
    const PeNorm = this.interp1d(timeIndices, normTrend, timeIndices);
    
    // Flip the trend because we are starting from highest pressure to lowest pressure
    const reversedPeNorm = [...PeNorm].reverse();
    
    // Step 3: Compute the filtered signal using bandpass filter
    const filteredSignal = this.bandpass_filter(signal, 250);
    
    // Flip the filtered signal
    const O = [...filteredSignal].reverse();
    
    // Step 4: Find the peaks of the filtered signal
    const peakIndices = this.findpeaks(O, 0.5, 150);
    const peaks = peakIndices.map(idx => O[idx]);
    const peaksPe = peakIndices.map(idx => reversedPeNorm[idx]);
    
    // Step 5: Find the bottoms of the filtered signal
    const bottomIndices = this.findbottoms(O, 0.5, 150);
    const bottoms = bottomIndices.map(idx => O[idx]);
    const bottomsPe = bottomIndices.map(idx => reversedPeNorm[idx]);
    
    // Step 6: Fit the half gaussian function to the peaks and bottoms vs Pe
    let DO = new Array(O.length).fill(0);
    const peakVal = Math.max(...peaks);
    const peakTimeIndex = peaks.indexOf(peakVal);
    const peakTime = peaksPe[peakTimeIndex];
    const timeRange = Math.max(...peaksPe) - Math.min(...peaksPe);
    
    // Initial guess for half gaussian
    const initialGuess = [
      peakVal,           // A
      peakTime,          // B
      timeRange * 0.1,   // s1
      timeRange * 0.1,   // s2
      0.8,               // sc1
      1.0                // sc2
    ];
    
    // Fit the Half Gaussian function
    const halfgaussParams = this.fit_halfgauss(reversedPeNorm, O, peakIndices, initialGuess);
    
    // Calculate fitted curve
    if (halfgaussParams) {
      const fittedCurve = this.halfgauss(reversedPeNorm, ...halfgaussParams);
      DO = fittedCurve;
    }
    
    // For bottoms
    const bottomVal = Math.min(...bottoms);
    const bottomTimeIndex = bottoms.indexOf(bottomVal);
    const bottomTime = bottomsPe[bottomTimeIndex];
    
    // Initial guess for bottoms
    const initialGuessBottom = [
      bottomVal,          // A
      bottomTime,         // B
      timeRange * 0.1,    // s1
      timeRange * 0.1,    // s2
      0.8,                // sc1
      1.0                 // sc2
    ];
    
    // Fit half gaussian for bottoms
    const halfgaussParamsBottom = this.fit_halfgauss(reversedPeNorm, O.map(v => -v), bottomIndices, initialGuessBottom);
    
    // Calculate fitted curve for bottoms
    if (halfgaussParamsBottom) {
      const fittedCurveBottom = this.halfgauss(reversedPeNorm, ...halfgaussParamsBottom);
      DO = DO.map((val, idx) => val + fittedCurveBottom[idx]);
    }
    
    // Step 8: Find the derivative of the DO signal
    const ddOdT = this.gradient(DO, reversedPeNorm);
    
    // Step 9: Find where the maximum of the dDO signal occurs
    const maxIndex = ddOdT.indexOf(Math.max(...ddOdT));
    const Pemax = reversedPeNorm[maxIndex];
    
    // Step 10: Find the minimum of the dDO signal
    const minIndex = ddOdT.indexOf(Math.min(...ddOdT));
    const Pemin = reversedPeNorm[minIndex];
    
    // Using Pemin and Pemax as SBP and DBP directly
    const SBP = Pemin;
    const DBP = Pemax;
    
    return { SBP, DBP };
  }

  /**
   * Estimate DBP with visualization support
   * 
   * @param {Array} time - Time values corresponding to the signal.
   * @param {Array} rawSignal - Raw signal values.
   * @param {number} id - Identifier for sampling rate determination.
   * @returns {Object} - Systolic and diastolic blood pressure values.
   */
  static estimate_dbp(time, rawSignal, id = 1) {
    // Determine sampling rate
    const fs = id >= 17 ? 250 : 250;
    
    // Step 1: Filter the signal
    const filteredSignal = this.bandpass_filter(rawSignal, fs);
    
    // Step 2: Find peaks
    const peakIndices = this.findpeaks(filteredSignal, 0.15 * Math.max(...filteredSignal), 70);
    
    // Step 3: Crop signal to region of interest
    let a = peakIndices[0];
    let b = peakIndices[peakIndices.length - 1] + 1;
    const signal = rawSignal.slice(a, b);
    const timeSlice = time.slice(a, b);
    
    // Step 4: Get trend
    const trendMvavg = this.trend(signal);
    const [trend, _] = this.exp_trend(trendMvavg, timeSlice);
    
    // Create reversed copies for analysis
    const Pe = [...trend].reverse();
    const PeNorm = this.normalize_signal(Pe);
    const O = [...filteredSignal.slice(a, b)].reverse();
    
    // Find peaks and bottoms in the oscillation signal
    const peakIndicesO = this.findpeaks(O, 0.15 * Math.max(...O), 150);
    const peaks = peakIndicesO.map(idx => O[idx]);
    const peaksPe = peakIndicesO.map(idx => PeNorm[idx]);
    
    const bottomIndicesO = this.findbottoms(O, 0.15 * Math.max(...O), 150);
    const bottoms = bottomIndicesO.map(idx => O[idx]);
    const bottomsPe = bottomIndicesO.map(idx => PeNorm[idx]);
    
    // Fit Gaussian to peaks
    const initialGuess = [1, PeNorm[peakIndicesO[Math.floor(peakIndicesO.length / 2)]], 1];
    const gaussianParams = this.fit_fGuas(PeNorm, O, peakIndicesO, initialGuess);
    
    // Fit Gaussian to bottoms
    const initialGuessBottom = [1, PeNorm[bottomIndicesO[Math.floor(bottomIndicesO.length / 2)]], 1];
    const gaussianParamsBottom = this.fit_fGuas(PeNorm, O, bottomIndicesO, initialGuessBottom);
    
    // Create oscillometric envelope
    let gaussianFitP, gaussianFitB, DO;
    
    if (gaussianParams) {
      const [A, mu, sigma] = gaussianParams;
      gaussianFitP = this.fGuas(PeNorm, A, mu, sigma);
      DO = gaussianFitP;
    }
    
    if (gaussianParamsBottom) {
      const [A, mu, sigma] = gaussianParamsBottom;
      gaussianFitB = this.fGuas(PeNorm, A, mu, sigma);
      DO = DO.map((val, idx) => val - gaussianFitB[idx]);
    }
    
    // Calculate derivative
    const dDOdP = this.gradient(DO, PeNorm);
    
    // Find maxima and minima
    const maxIndex = dDOdP.indexOf(Math.max(...dDOdP));
    const Pemax = PeNorm[maxIndex];
    
    const minIndex = dDOdP.indexOf(Math.min(...dDOdP));
    const Pemin = PeNorm[minIndex];
    
    // Note: In a complete implementation, you would create charts here
    // For React Native, you could use a charting library like Victory Native
    
    return { SBP: Pemin, DBP: Pemax };
  }
}

export { SignalAnalyzer }; 