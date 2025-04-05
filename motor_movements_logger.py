import csv
import time
import serial
import threading
import sys

#   case 'a':
#     movemotor(HIGH, LOW, 200);
#     break;
#   case 's':
#     movemotor(HIGH, LOW, 150);
#     break;
#   case 'd':
#     movemotor(HIGH, LOW, 100);
#     break;
#   case 'f':
#     movemotor(HIGH, LOW, 50);
#     break;
#   case ';':
#     movemotor(LOW, HIGH, 200);
#     break;
#   case 'l':
#     movemotor(LOW, HIGH, 150);
#     break;
#   case 'k':
#     movemotor(LOW, HIGH, 100);
#     break;
#   case 'j':
#     movemotor(LOW, HIGH, 50);


# movements  = ['d', 'f', 'l', 'k', 'j', 'd', 'd', 'l','f']


# Configure your serial connection
ser = serial.Serial('/dev/cu.usbserial-0001', 115200)

# Global variable to track current action
current_action = "a"

def move_motor(command):
    ser.write(command.encode())  # Send command as a byte
    print(f"Sent: {command}")

def input_thread():
    """Thread function to handle user input"""
    global current_action
    
    print("Enter commands at any time. Press 'q' to quit.")
    while True:
        cmd = input().strip()
        if cmd == 'q':
            print("Shutting down...")
            break
        else:
            current_action = cmd
            move_motor(cmd)

def log_pendulum_data():
    """Function to log pendulum data to CSV"""
    with open("log_video_a_l_test.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["xAccl", "yAccl", "zAccl", "newPosition", "time",'pcTime'])
        
        start_time = time.time()
        
        while True:
            try:
                line = ser.readline().decode('ISO-8859-1').strip()
                values = line.split(",")
                
                if len(values) >= 5:
                    pc_time = (time.time() - start_time) * 1000
                    
                    # Add PC time and current action to the logged data
                    writer.writerow(values)
                    print(f"Data: {values}")
                    
                    # Flush to ensure data is written
                    file.flush()
                    
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue

# Main execution
if __name__ == "__main__":
    logging_thread = threading.Thread(target=log_pendulum_data, daemon=True)
    logging_thread.start()
    # move_motor('j')
    
    print("Logging started. Enter motor commands to move the motor.")
    
    while True:
        try:
            command = input("Enter command (or 'q' to quit): ").strip()
            if command.lower() == 'q':
                print("Exiting...")
                break
            
            # elif command.lower() == 'b':
            #     for move in movements:
            #         move_motor(move)
            #         time.sleep(0.5)
            
            move_motor(command)
        except KeyboardInterrupt:
            print("\nProgram terminated by user")
            break
