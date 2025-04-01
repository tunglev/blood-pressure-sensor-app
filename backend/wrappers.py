class SignalWrapper:
    def __init__(self, signals):
        """
        A wrapper for signals to allow querying by both index and key.

        Args:
        - signals (list): A list of dictionaries, each containing a signal with a key, signal array, and time array.
        """
        self.signals = signals
        self.key_map = {signal["key"]: signal for signal in signals}

    def __getitem__(self, query):
        """
        Retrieve a signal by index or key.

        Args:
        - query (int or str): The index (int) or key (str) of the signal to retrieve.

        Returns:
        - dict: The signal dictionary with keys "key", "signal", and "time".
        """
        if isinstance(query, int):
            return self.signals[query]
        elif isinstance(query, str):
            return self.key_map.get(query, None)
        else:
            raise TypeError("Query must be an integer (index) or a string (key).")

    def __len__(self):
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)






if __name__ == "__main__":
    signals = [
        {"key": "signal1", "signal": [1, 2, 3], "time": [0.1, 0.2, 0.3]},
        {"key": "signal2", "signal": [4, 5, 6], "time": [0.4, 0.5, 0.6]},
    ]

    signal_wrapper = SignalWrapper(signals)

    # Iterate over the SignalWrapper instance
    for signal in signal_wrapper:
        print(signal)