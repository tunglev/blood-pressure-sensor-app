// config.js
// Configuration settings for the signal analyzer

// Logging configuration
export const LOGGING_LEVEL = 'info'; // 'debug', 'info', 'warning', 'error'
export const LOGGING_STATE = false;  // Set to true to disable logging

// Font sizes for plots (when used with a charting library)
export const FONTSIZE1 = 14;
export const FONTSIZE2 = 16;

// Filter parameters
export const DEFAULT_WINDOW_LENGTH = 1000;
export const DEFAULT_POLYORDER = 2;
export const DEFAULT_FS = 250; // Default sampling frequency in Hz

// Peak detection parameters
export const DEFAULT_PEAK_THRESHOLD = 2;
export const DEFAULT_PEAK_DISTANCE = 100;

// Blood pressure estimation parameters
export const BP_ALPHA = 0.1;
export const BP_BETA = 0.2;

// Export all constants as a default object for easy import
export default {
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
}; 