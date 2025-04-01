import serial
import serial.tools.list_ports
import csv
import time
import sys

def list_available_ports():
    """List all available serial ports with descriptions."""
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("No serial ports detected.")
        return []
    
    print("Available serial ports:")
    for port in ports:
        print(f"- {port.device}: {port.description}")
    
    return [port.device for port in ports]

def send_mock_data(port, baud_rate=9600, data_rate=10):
    """
    Send mock blood pressure data over serial port.
    
    Args:
        port: Serial port name (e.g., 'COM3' on Windows)
        baud_rate: Serial baud rate
        data_rate: How many data points to send per second
    """
    try:
        # Initialize serial connection
        ser = serial.Serial(port, baud_rate)
        print(f"Connected to {port} at {baud_rate} baud rate")
        
        # Calculate sleep time between data points
        sleep_time = 1.0 / data_rate
        
        # Read data from CSV
        data = []
        with open('mockData.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                try:
                    sample_index = row[0]
                    value = row[1]
                    data.append((sample_index, value))
                except IndexError:
                    continue
        
        print(f"Loaded {len(data)} data points. Sending at {data_rate} points per second.")
        
        # Continuously send data
        try:
            while True:
                for sample_index, value in data:
                    # Format: "sample_index,value\n"
                    message = f"{sample_index},{value}\n"
                    ser.write(message.encode())
                    print(f"Sent: {message.strip()}")
                    time.sleep(sleep_time)
                print("Restarting data sequence...")
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            ser.close()
            print("Serial connection closed")
            
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Send mock blood pressure data via serial port')
    parser.add_argument('port', nargs='?', help='Serial port (e.g., COM3 on Windows)')
    parser.add_argument('--baud', type=int, default=9600, help='Baud rate (default: 9600)')
    parser.add_argument('--rate', type=int, default=10, help='Data points per second (default: 10)')
    parser.add_argument('--list', action='store_true', help='List available serial ports')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_ports()
        sys.exit(0)
    
    if not args.port:
        available_ports = list_available_ports()
        if not available_ports:
            print("No ports found. Please specify a port manually.")
            sys.exit(1)
        
        print("\nPlease specify a port using: python serial_sender.py PORT")
        sys.exit(1)
    
    send_mock_data(args.port, args.baud, args.rate) 