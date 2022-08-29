/*
 * @Description: 
 * @Author: YiPei_Fang
 * @Date: 2022-08-24 12:00:45
 */
#include <ESP32Servo.h>
#include "include/pid.h"

//pid
PID Pitch(0.4,0,0.05,90);//(2.5,0,0,90)  2.1 1.7  0.5
PID Yaw(0.4,0,0.05,90);
#define MAX_PITCH 90
#define MAX_YAW 90
#define dt 20  //10ms
//servo
Servo myservo1,myservo2;
int pos1 = 90;int pos2 = 90;    
int servo1Pin = 18;// Recommended PWM GPIO pins on the ESP32 include 2,4,12-19,21-23,25-27,32-33 
int servo2Pin = 17;
typedef struct
{
  int data[10] ={0,0};
  int len = 0;
}List;
List list;
void getList();
void clearList();

void setup() {
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  //servo1 and servo2
  myservo1.setPeriodHertz(50);
  myservo1.attach(servo1Pin, 1000, 2000);
  myservo1.write(pos1);

  myservo2.setPeriodHertz(50);
  myservo2.attach(servo2Pin, 1000, 2000);
  myservo2.write(pos2);
}

void loop() {
  if(Serial.available())
  {
    getList();
    for (int i=0; i<list.len; i++)
   {
      Serial.print(list.data[i]);Serial.print(" ");//set real
   }
    Serial.println("");
  //pid
  Pitch.output = constrain(Pitch._calculate(list.data[2],list.data[3],dt),-MAX_PITCH,MAX_PITCH);
  Yaw.output = constrain(Yaw._calculate(list.data[0],list.data[1],dt),-MAX_YAW,MAX_YAW);
  //myservo1.write(pos1-Pitch.output);  //blob
  myservo1.write(pos1+Pitch.output); //face
  myservo2.write(pos2+Yaw.output);
  clearList();
  }
  //Serial.write("2022-YiPei-Fang\r\n");
}


String detectString()
{
  while(Serial.read() != '{');
  return(Serial.readStringUntil('}'));
}
void clearList()
{
  memset(list.data, sizeof(list.data),0);
  list.len = 0;
}
void getList()
{
  String s = detectString();
  String strnum = "";
  for(uint8_t i = 0; i<s.length(); i++){
    if(s[i] == ','){
      list.data[list.len] = strnum.toInt();
      list.len++;
      strnum = "";
    }
    else{
      strnum += s[i];
    }

  }
  

}
