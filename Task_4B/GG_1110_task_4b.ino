// Motor A
int motor1Pin2 = 15; // IN1
int motor1Pin1 = 5;  // IN2
int enable1Pin = 17; // ENA

// Motor B
int motor2Pin2 = 18; // IN3
int motor2Pin1 = 19; // IN4
int enable2Pin = 21; // ENB

const int irSensorPin1 = 13;
const int irSensorPin2 = 12;
const int irSensorPin3 = 14;
const int irSensorPin4 = 27;
const int irSensorPin5 = 26;

const int buzzerPin = 25;
const int ledPin = 23;

// Setting PWM properties
const int freq = 30000;
const int pwmChannel = 0;
const int resolution = 8;
int dutyCycle = 200;

void setup() {
  // Initialize motor control pins as outputs
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
  pinMode(enable2Pin, OUTPUT);

  // Initialize IR sensor pins as inputs
  pinMode(irSensorPin1, INPUT);
  pinMode(irSensorPin2, INPUT);
  pinMode(irSensorPin3, INPUT);
  pinMode(irSensorPin4, INPUT);
  pinMode(irSensorPin5, INPUT);

  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  // configure LED PWM functionalities
  ledcSetup(pwmChannel, freq, resolution);

  // attach the channel to the GPIO to be controlled
  ledcAttachPin(enable1Pin, pwmChannel);
  ledcAttachPin(enable2Pin, pwmChannel);

  Serial.begin(115200);
  Serial.println("Line Following Robot Initialized");

  buzzerlight(1000);
}

// Function to stop both motors
void stopMotors() {
  Serial.println("STOP");
  ledcWrite(pwmChannel, 0); // Stop motors by setting PWM to 0
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
}


// Function to turn the robot to the right
void turnLeft() {
  Serial.println("LEFT");
  ledcWrite(pwmChannel, 200); // Adjust duty cycle for turning
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW); // Reverse direction for this motor
  digitalWrite(motor2Pin2, HIGH); // Reverse direction for this motor
}

// Function to turn the robot to the left
void turnRight() {
  Serial.println("RIGHT");
  ledcWrite(pwmChannel, 200); // Adjust duty cycle for turning
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH); // Reverse direction for this motor
  digitalWrite(motor2Pin2, LOW); // Reverse direction for this motor
}

// Function to move the robot straight
void moveStraight() {
  Serial.println("STRAIGHT");
  ledcWrite(pwmChannel, 255); // Maximum speed for moving straight
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW); // Reverse direction for this motor
  digitalWrite(motor2Pin2, HIGH); // Reverse direction for this motor
}

void buzzerlight(int duration) {
  analogWrite(buzzerPin,255);
  //delay(duration);
   
  delay(duration);
  digitalWrite(ledPin, HIGH); 
  tone(buzzerPin,0);
  
  delay(duration);
  digitalWrite(ledPin, LOW);

}

void buzzerWithoutLight(int duration) {
  analogWrite(buzzerPin,255);
  //delay(duration);
   
  delay(duration);
  tone(buzzerPin,0);
  
  delay(duration);
}


int intersectionCount = 0; // Initialize a counter for intersections


// Define an array to hold actions for each intersection
// Modify these based on your desired actions
const int TOTAL_INTERSECTIONS =  11; // Change this according to the number of intersections
int intersectionActions[TOTAL_INTERSECTIONS] = {0,0,2,1,2,2,0,2,0,1,0}; // Example actions (0 = straight, 1 = left, 2 = right)

void loop() {
  int sensorValue1 = digitalRead(irSensorPin1);
  int sensorValue2 = digitalRead(irSensorPin2);
  int sensorValue3 = digitalRead(irSensorPin3);
  int sensorValue4 = digitalRead(irSensorPin4);
  int sensorValue5 = digitalRead(irSensorPin5);

  // Combine sensor values into an integer for comparison
  int sensorPattern = (sensorValue1 << 4) | (sensorValue2 << 3) | (sensorValue3 << 2) | (sensorValue4 << 1) | sensorValue5;

  if (intersectionCount == TOTAL_INTERSECTIONS) {
    Serial.print("intersection last");
    Serial.println(intersectionCount);
    int currentAction = intersectionActions[TOTAL_INTERSECTIONS - 1];
    Serial.println(currentAction);

    int bleh = 0;
    while(bleh<700){
      Serial.println(bleh);
    
      int sensorValue1 = digitalRead(irSensorPin1);
      int sensorValue2 = digitalRead(irSensorPin2);
      int sensorValue3 = digitalRead(irSensorPin3);
      int sensorValue4 = digitalRead(irSensorPin4);
      int sensorValue5 = digitalRead(irSensorPin5);

      // Combine sensor values into an integer for comparison
      int sensorPattern = (sensorValue1 << 4) | (sensorValue2 << 3) | (sensorValue3 << 2) | (sensorValue4 << 1) | sensorValue5;

    switch (sensorPattern) {
    case B00000:
    case B10001:
    case B00100:
    case B10101:
      Serial.println("Moving Straight");
      moveStraight();
      break;

 
    case B01000:
    case B01001:
    case B01100:
    case B01101:
    case B11001:
    case B11101:
      Serial.println("Turning Left");
      turnLeft();
      delay(5);
      break;

    case B00010:
    case B00110:
    case B10010:
    case B10011:   
    case B10110:
    case B10111:
      Serial.println("Turning Right");
      turnRight();
      delay(5);
      break;
      
        case B10000: // CASES TO REDUCE WAGGING AT TYPE 1
    case B10100:
      Serial.println("Right type 1");
      turnRight();
      delay(1);
      moveStraight();
      delay(0.5);
      break;

    case B00001: // CASES TO REDUCE WAGGING AT TYPE 1
    case B00101:
      Serial.println("Left type 1");
      turnLeft();
      delay(1);
      moveStraight();
      delay(0.5);
      break;

       }
      bleh++;
    }
    stopMotors();
    buzzerlight(5000);

    while(true){
       // Serial.println("entering while loop");
        analogWrite(buzzerPin,0);
    }

  }

  // Print individual sensor values and the combined sensor pattern to Serial Monitor
  Serial.print("Sensor Values: ");
  Serial.print(sensorValue1);
  Serial.print(sensorValue2);
  Serial.print(sensorValue3);
  Serial.print(sensorValue4);
  Serial.println(sensorValue5);

  int currentAction = 0; // Declare outside the switch statement

  // Determine action based on sensor pattern using switch case
  switch (sensorPattern) {
    case B00000:
    case B10001:
    case B00100:
    case B10101:
      Serial.println("Moving Straight");
      moveStraight();
      break;

 
    case B01000:
    case B01001:
    case B01100:
    case B01101:
    case B11001:
    case B11101:
      Serial.println("Turning Left");
      turnLeft();
      delay(5);
      break;

    case B00010:
    case B00110:
    case B10010:
    case B10011:   
    case B10110:
    case B10111:
      Serial.println("Turning Right");
      turnRight();
      delay(5);
      break;
    
    case B10000: // CASES TO REDUCE WAGGING AT TYPE 1
    case B10100:
      Serial.println("Right type 1");
      turnRight();
      delay(1);
      moveStraight();
      delay(0.5);
      break;

    case B00001: // CASES TO REDUCE WAGGING AT TYPE 1
    case B00101:
      Serial.println("Left type 1");
      turnLeft();
      delay(1);
      moveStraight();
      delay(0.5);
      break;

    // case B01010:
    case B01110:
    case B01111:
    case B11011:
    case B11110:
    case B11111:
      Serial.println("Stopping");
      stopMotors();
      buzzerWithoutLight(1000);
      currentAction = intersectionActions[intersectionCount];



  // Check if the robot is not already in a straight configuration after moving 100 units
  if (currentAction == 0) {
    // If not in a straight configuration, move an additional 300 units
    moveStraight();
    delay(300); // Move straight for an additional 400 milliseconds (adjust as needed)
  }
  else {
    moveStraight();
    delay(450);
  }


      Serial.print("Intersection Count: "); // Print the count to Serial Monitor
      Serial.println(intersectionCount);

      Serial.print("Action at Intersection: "); // Print the action to Serial Monitor
      Serial.println(currentAction);

      switch (currentAction) {
        case 0:
          Serial.println("Moving Straight");
          // Continue moving straight as usual
          intersectionCount++;
          break;
          
        case 1:
          Serial.println("Turning Left");
          turnLeft();
          delay(1100); //TURNS LEFT 90 DEGREES
          stopMotors(); // Stop the robot once it reaches the straight pattern
          intersectionCount++;
          break;
          
        case 2:
          Serial.println("Turning Right");
          turnRight();
          delay(1100); // TURNS RIGHT 90 DEGREES
          stopMotors(); // Stop the robot once it reaches the straight pattern
          intersectionCount++;
          break;
      }
      //intersectionCount++; // Increment the intersection count
      //break;
     default:
      Serial.println("No matching pattern found");
      break;
  }
  // delay(5); // DELAY REDUCTION HERE INTRODUCES WAGGING BUT INCREASES ACCURACY
}
