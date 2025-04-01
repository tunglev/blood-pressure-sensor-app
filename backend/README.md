# Blood Pressure Sensor App - JavaScript Library

This package contains JavaScript implementations of signal analysis functions for blood pressure estimation from oscillometric signals. It has been translated from the original Python code to work with JavaScript/React Native environments.

## Features

- Signal trend calculation
- Normalization functions
- Bandpass filtering
- Peak detection
- Blood pressure estimation algorithms
- Signal wrapper utility

## Installation

No additional installation is required since this code is part of your Expo app. The functions can be imported directly into your components.

## Usage

### Basic Usage

```javascript
import { SignalAnalyzer } from './calculations';
import { SignalWrapper } from './wrappers';

// Sample data
const time = Array.from({length: 1000}, (_, i) => i / 250); // 4 seconds at 250Hz
const signal = time.map(t => Math.sin(2 * Math.PI * 1 * t) * Math.exp(-t/2));

// Estimate blood pressure
const { SBP, DBP } = SignalAnalyzer.estimatebp(time, signal);
console.log(`Systolic: ${SBP}, Diastolic: ${DBP}`);
```

### Using the SignalWrapper

```javascript
const signals = [
  {key: "signal1", signal: [...], time: [...]},
  {key: "signal2", signal: [...], time: [...]},
];

const wrapper = new SignalWrapper(signals);

// Access by index
const firstSignal = wrapper.getItem(0);

// Access by key
const specificSignal = wrapper.getItem("signal2");

// Iterate through signals
for (const signal of wrapper) {
  // Process each signal
}
```

## Implementation Notes

1. **Curve Fitting**: The JavaScript implementation provides simplified versions of the curve fitting functions. For production use, consider integrating a proper curve fitting library.

2. **Digital Signal Processing**: The bandpass filter is a simplified version. For more accurate results, consider using a DSP library such as DSP.js.

3. **Visualization**: The code focuses on the calculation aspects. For visualization in Expo/React Native, you can use libraries like Victory Native or react-native-svg-charts.

## Functions

### SignalAnalyzer

- `trend(signal, windowLength, polyorder)`: Calculate the trend of a signal
- `normalize_signal(signal, targetRange)`: Normalize a signal to a specific range
- `bandpass_filter(signal, Fs)`: Apply a bandpass filter to a signal
- `findpeaks(signal, threshold, distance)`: Find peaks in a signal
- `findbottoms(signal, threshold, distance)`: Find valleys in a signal
- `estimatebp(time, signal)`: Estimate blood pressure from an oscillometric signal

### SignalWrapper

- `getItem(query)`: Get a signal by index or key
- `length`: Get the number of signals
- `[Symbol.iterator]()`: Iterator for the signals collection

## License

This code is part of your proprietary application. 