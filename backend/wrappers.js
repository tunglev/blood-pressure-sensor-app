// wrappers.js

class SignalWrapper {
  /**
   * A wrapper for signals to allow querying by both index and key.
   * 
   * @param {Array} signals - A list of objects, each containing a signal with a key, signal array, and time array.
   */
  constructor(signals) {
    this.signals = signals;
    this.keyMap = {};
    
    // Create a map for quick access by key
    for (const signal of signals) {
      this.keyMap[signal.key] = signal;
    }
  }

  /**
   * Retrieve a signal by index or key.
   * 
   * @param {number|string} query - The index (number) or key (string) of the signal to retrieve.
   * @returns {Object} - The signal object with keys "key", "signal", and "time".
   */
  getItem(query) {
    if (typeof query === 'number') {
      return this.signals[query];
    } else if (typeof query === 'string') {
      return this.keyMap[query] || null;
    } else {
      throw new TypeError("Query must be a number (index) or a string (key).");
    }
  }

  /**
   * Get the number of signals in the wrapper.
   * 
   * @returns {number} - The number of signals.
   */
  get length() {
    return this.signals.length;
  }

  /**
   * Allow iteration over the signals.
   * 
   * @returns {Iterator} - An iterator for the signals.
   */
  [Symbol.iterator]() {
    return this.signals[Symbol.iterator]();
  }
}

// Example usage
if (typeof require !== 'undefined' && require.main === module) {
  const signals = [
    {key: "signal1", signal: [1, 2, 3], time: [0.1, 0.2, 0.3]},
    {key: "signal2", signal: [4, 5, 6], time: [0.4, 0.5, 0.6]},
  ];

  const signalWrapper = new SignalWrapper(signals);

  // Iterate over the SignalWrapper instance
  for (const signal of signalWrapper) {
    console.log(signal);
  }
  
  // Access by index
  console.log(signalWrapper.getItem(0));
  
  // Access by key
  console.log(signalWrapper.getItem("signal2"));
}

export { SignalWrapper }; 