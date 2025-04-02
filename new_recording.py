import serial
import csv
import time

ser = serial.Serial('/dev/cu.usbserial-0001', 115200)

with open("data_new.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time", "xAccl", "yAccl", "zAccl", "encoder"])  # Updated header to include time
    
    while True:
        try:
            line = ser.readline().decode().strip()
            values = line.split(",")
            print(values)
            if len(values) == 5:  # Ensure we have all 4 values
                current_time = time.time()  # Get the current time
                writer.writerow([current_time] + values)  # Prepend time to the values
                time.sleep(0.005)
                print([current_time] + values)
        except KeyboardInterrupt:
            print("Stopped by user")
            break