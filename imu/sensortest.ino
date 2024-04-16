#ifndef _LSM9DS1_H_
#define _LSM9DS1_H_

#include "sensor.h"

#endif

void setup() {
  Serial.begin(115200);
  while(!Serial);
  // Serial.println("go");

  if(!IMU.begin()){
    // Serial.println("--");
    while(1);
  }
}

char output[64];
int16_t data[4] = {0x00,0x00,0x00,0xFFFF};
int16_t data_z;

#define LSM9DS1_ADDRESS 0x6b
#define LSM9DS1_OUT_X_XL 0x28

void loop() {
  if (IMU.accelerationAvailable()) {
    if(!IMU.readRegisters(LSM9DS1_ADDRESS, LSM9DS1_OUT_X_XL, (uint8_t*)data, sizeof(data)))
    {
      data[0] = 0x00;
      data[1] = 0x00;
      data[2] = 0x00;
    }
    data[3] = 0xFFFF;
    Serial.write((uint8_t *)&data,sizeof(int16_t) * 4);
  }
}

