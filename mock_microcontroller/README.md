# Blood Pressure Mock Data Sender

This script sends mock blood pressure data through a serial port in a continuous loop.

## Requirements

- Python 3.6 or higher
- PySerial library

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Finding Available Serial Ports

To find which ports are available on your computer:

```
python serial_sender.py --list
```

This will display all available serial ports and their descriptions, helping you identify which port your device is connected to.

Alternatively, you can use Device Manager (Windows) or `ls /dev/tty*` (Linux/macOS) to find connected devices.

## Usage

```
python serial_sender.py PORT [--baud BAUD_RATE] [--rate DATA_RATE]
```

Arguments:
- `PORT`: The serial port to use (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)
- `--baud`: Baud rate (default: 9600)
- `--rate`: Data points per second (default: 10)

## Example

```
python serial_sender.py COM3 --baud 115200 --rate 20
```

This will send data at 20 data points per second at 115200 baud rate through COM3.

## Stopping the Script

Press Ctrl+C to stop the script. 