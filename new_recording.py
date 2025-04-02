import serial
import csv
import time

ser = serial.Serial('/dev/cu.usbserial-0001', 115200)

# 163
# 97
# 74
# 30

with open("counter_clockwise_90_theta_1.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["xAccl", "yAccl", "zAccl", "newPosition", "time",'pcTime'])  # Updated header to include time
    
    start_time = time.time()  
    while True:
        try:
            line = ser.readline().decode('ISO-8859-1').strip()
            values = line.split(",")
            current_time = (time.time() - start_time) * 1000
            writer.writerow(values + [current_time])
            print(values)

        except KeyboardInterrupt:
            print("Stopped by user")
            break