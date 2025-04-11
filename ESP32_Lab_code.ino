#include <Wire.h>
#include <ESP32Encoder.h>
// MMA8452Q I2C address is 0x1C(28)
#define Addr 0x1C

#define CLK 18  // CLK ENCODER
#define DT 19   // DT ENCODER

int ENA = 15;
int IN1 = 2;
int IN2 = 4;
char incomingByte;
int movement = 0;

ESP32Encoder encoder;
boolean data_stream = false;
int sign = 1;
unsigned long previous_time = micros();
void movemotor(int p1, int p2, int num) {
  digitalWrite(IN1, p1);
  digitalWrite(IN2, p2);
  delay(num);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  digitalWrite(ENA, HIGH);
  // Initialise I2C communication as MASTER
  Wire.begin();
  // Initialise Serial Communication, set baud rate = 9600
  Serial.begin(115200);
  // Start I2C Transmission
  Wire.beginTransmission(Addr);
  // Select control register
  Wire.write(0x2A);
  // StandBy mode
  Wire.write((byte)0x00);
  // Stop I2C Transmission
  Wire.endTransmission();

  // Start I2C Transmission
  Wire.beginTransmission(Addr);
  // Select control register
  Wire.write(0x2A);
  // Active mode
  Wire.write(0x01);
  // Stop I2C Transmission
  Wire.endTransmission();

  // Start I2C Transmission
  Wire.beginTransmission(Addr);
  // Select control register
  Wire.write(0x0E);
  // Set range to +/- 2g
  Wire.write((byte)0x00);
  // Stop I2C Transmission
  Wire.endTransmission();
  delay(300);

  encoder.attachHalfQuad(DT, CLK);
  encoder.setCount(0);
}

void loop() {
  unsigned long current_time = micros();
  unsigned long delta_time = current_time - previous_time;
  while ((Serial.available() > 0)) {

    incomingByte = Serial.read();

    switch (incomingByte) {
      case 'a':
        movemotor(HIGH, LOW, 200);
        break;
      case 's':
        movemotor(HIGH, LOW, 150);
        break;
      case 'd':
        movemotor(HIGH, LOW, 100);
        break;
      case 'f':
        movemotor(HIGH, LOW, 50);
        break;
      case ';':
        movemotor(LOW, HIGH, 200);
        break;
      case 'l':
        movemotor(LOW, HIGH, 150);
        break;
      case 'k':
        movemotor(LOW, HIGH, 100);
        break;
      case 'j':
        movemotor(LOW, HIGH, 50);
        break;
      default:
        // Handle other characters or do nothing
        // Serial.println(incomingByte);
        break;
    }
    // Serial.print("test");
  }
    if ((delta_time) > 25000) {
      previous_time = current_time;

      unsigned int data[7];

      // Request 7 bytes of data
      Wire.requestFrom(Addr, 7);

      // Read 7 bytes of data
      // staus, xAccl lsb, xAccl msb, yAccl lsb, yAccl msb, zAccl lsb, zAccl msb
      if (Wire.available() == 7) {
        data[0] = Wire.read();
        data[1] = Wire.read();
        data[2] = Wire.read();
        data[3] = Wire.read();
        data[4] = Wire.read();
        data[5] = Wire.read();
        data[6] = Wire.read();
      }

      // Convert the data to 12-bits
      int xAccl = ((data[1] * 256) + data[2]) / 16;
      if (xAccl > 2047) {
        xAccl -= 4096;
      }

      int yAccl = ((data[3] * 256) + data[4]) / 16;
      if (yAccl > 2047) {
        yAccl -= 4096;
      }

      int zAccl = ((data[5] * 256) + data[6]) / 16;
      if (zAccl > 2047) {
        zAccl -= 4096;
      }
      long newPosition = encoder.getCount();
      unsigned long time = micros();
      // Output data to serial monitor
      Serial.print(xAccl);
      Serial.print(",");
      Serial.print(yAccl);
      Serial.print(",");
      Serial.print(zAccl);
      Serial.print(",");
      Serial.print(newPosition);
      Serial.print(",");
      Serial.print(time);
      Serial.println("");
    }
  
}
