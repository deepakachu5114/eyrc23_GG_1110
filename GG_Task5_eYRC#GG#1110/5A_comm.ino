#include <WiFi.h>
#include <ArduinoJson.h>

const char* ssid = "Chaitanyas phone";
const char* password = "suckmydick";
const int port = 8266;  // Choose any available port

WiFiServer server(port);




// Motor A
int motor1Pin2 = 15;  // IN1
int motor1Pin1 = 5;   // IN2
int enable1Pin = 17;  // ENA

// Motor B
int motor2Pin2 = 18;  // IN3
int motor2Pin1 = 19;  // IN4
int enable2Pin = 21;  // ENB

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



int* intersectionActions = nullptr;  // Dynamically allocated array
// int intersectionActions[TOTAL_INTERSECTIONS] = {0};
int TOTAL_INTERSECTIONS = 0;
bool arrayProcessed = false;
WiFiClient client;


void handleArrayData(WiFiClient& client) {
  // Read the incoming JSON string
  String jsonStr = client.readStringUntil('\n');

  // Parse the JSON
  DynamicJsonDocument doc(256);
  deserializeJson(doc, jsonStr);


  // Access the array size
  TOTAL_INTERSECTIONS = doc["size"];

  // Dynamically allocate the array
  intersectionActions = new int[TOTAL_INTERSECTIONS];

  // Access the array elements
  int arraySize = doc["size"];
  for (int i = 0; i < arraySize; i++) {
    intersectionActions[i] = doc["data"][i];
  }

  // Acknowledge receipt
  client.println("Array received successfully");

  // Set the flag to indicate that the array has been processed
  arrayProcessed = true;
}





void setup() {

  Serial.begin(115200);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.print("Conectado.  ");
  Serial.print(" Dirección IP del módulo: ");
  Serial.println(WiFi.localIP());

  // Start the server
  server.begin();
  Serial.println("Server started");

  // Wait for a client to connect
  // WiFiClient client = server.available();
  while (!client.connected()) {
    delay(100);
    // Serial.println("Waiting for client connection...");
    client = server.available();
  }

  // Handle the array data only once during setup
  handleArrayData(client);






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
  ledcWrite(pwmChannel, 0);  // Stop motors by setting PWM to 0
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
}


// Function to turn the robot to the right
void turnLeft() {
  Serial.println("LEFT");
  ledcWrite(pwmChannel, 200);  // Adjust duty cycle for turning
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW);   // Reverse direction for this motor
  digitalWrite(motor2Pin2, HIGH);  // Reverse direction for this motor
}

// Function to turn the robot to the left
void turnRight() {
  Serial.println("RIGHT");
  ledcWrite(pwmChannel, 200);  // Adjust duty cycle for turning
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH);  // Reverse direction for this motor
  digitalWrite(motor2Pin2, LOW);   // Reverse direction for this motor
}

// Function to move the robot straight
void moveStraight() {
  Serial.println("STRAIGHT");
  ledcWrite(pwmChannel, 255);  // Maximum speed for moving straight
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);   // Reverse direction for this motor
  digitalWrite(motor2Pin2, HIGH);  // Reverse direction for this motor
}

void buzzerlight(int duration) {
  analogWrite(buzzerPin, 255);
  //delay(duration);

  delay(duration);
  digitalWrite(ledPin, HIGH);
  tone(buzzerPin, 0);

  delay(duration);
  digitalWrite(ledPin, LOW);
  // digitalWrite(ledPin, LOW);
  // analogWrite(buzzerPin, 0);
}

void buzzerWithoutLight(int duration) {
  analogWrite(buzzerPin, 255);
  //delay(duration);

  delay(duration);
  tone(buzzerPin, 0);

  delay(duration);
}


int intersectionCount = 0;  // Initialize a counter for intersections


// Define an array to hold actions for each intersection
// // Modify these based on your desired actions`

void default_action() {



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

    // if (currentAction == 0) {
    //   // If not in a straight configuration, move an additional 300 units
    //   moveStraight();
    //   delay(200); // Move straight for an additional 400 milliseconds (adjust as needed)
    // }
    // else {
    //   moveStraight();
    //   delay(450);
    // }

    if (currentAction != 0) {
      turnRight();
      delay(500);

    }
    int bleh = 0;
    while (bleh < 700) {
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

        case B10000:  // CASES TO REDUCE WAGGING AT TYPE 1
        case B10100:
          Serial.println("Right type 1");
          turnRight();
          delay(4);
          moveStraight();
          delay(0.5);
          break;

        case B00001:  // CASES TO REDUCE WAGGING AT TYPE 1
        case B00101:
          Serial.println("Left type 1");
          turnLeft();
          delay(4);
          moveStraight();
          delay(0.5);
          break;
      }

      bleh++;
    }
    stopMotors();
    buzzerlight(5000);

    // stopMotors();
    // delay(100000);
    while (true) {
      //   // Serial.println("entering while loop");
      analogWrite(buzzerPin, 0);
    }
  }

  // Print individual sensor values and the combined sensor pattern to Serial Monitor
  Serial.print("Sensor Values: ");
  Serial.print(sensorValue1);
  Serial.print(sensorValue2);
  Serial.print(sensorValue3);
  Serial.print(sensorValue4);
  Serial.println(sensorValue5);

  int currentAction = 0;  // Declare outside the switch statement

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
      delay(2);
      break;

    case B00010:
    case B00110:
    case B10010:
    case B10011:
    case B10110:
    case B10111:
      Serial.println("Turning Right");
      turnRight();
      delay(2);
      break;

    case B10000:  // CASES TO REDUCE WAGGING AT TYPE 1
    case B10100:
      Serial.println("Right type 1");
      turnRight();
      delay(3);
      moveStraight();
      delay(0.5);
      break;

    case B00001:  // CASES TO REDUCE WAGGING AT TYPE 1
    case B00101:
      Serial.println("Left type 1");
      turnLeft();
      delay(3);
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
      // stopMotors();
      // buzzerWithoutLight(1000);
      // if(intersectionCount == 0){
      //   buzzerlight(1000);
      // }
      // else{
      //   buzzerWithoutLight(1000);
      // }

      currentAction = intersectionActions[intersectionCount];
      // break;



      // Check if the robot is not already in a straight configuration after moving 100 units
      if (currentAction == 0) {
        // If not in a straight configuration, move an additional 300 units
        moveStraight();
        delay(300);  // Move straight for an additional 400 milliseconds (adjust as needed)
      } else {
        moveStraight();
        delay(450);
      }


      Serial.print("Intersection Count: ");  // Print the count to Serial Monitor
      Serial.println(intersectionCount);

      Serial.print("Action at Intersection: ");  // Print the action to Serial Monitor
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
          delay(1000);   //TURNS LEFT 90 DEGREES
          stopMotors();  // Stop the robot once it reaches the straight pattern
          intersectionCount++;
          break;

        case 2:
          Serial.println("Turning Right");
          turnRight();
          delay(1000);   // TURNS RIGHT 90 DEGREES
          stopMotors();  // Stop the robot once it reaches the straight pattern
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


void loop() {
  default_action();

  // Check if there is a client connection
  if (client.connected()) {
    // Read the data from the client
    if (client.available() > 0) {
      // Read the data from the client
      String receivedDataStr = client.readStringUntil('\n');
      int receivedData = receivedDataStr.toInt();

      // Print the received message
      Serial.print("Received Message: ");
      Serial.println(receivedData);

      // Optional: Send an acknowledgment back to the client
      // client.print("Message Received");

      if (receivedData != 4) {
        stopMotors();
        buzzerWithoutLight(1000);
        moveStraight();
        delay(500);
        client.flush();
        // client.print("ACK");
      }
    }
  } else {
    Serial.println("No client connection yet...");
    client = server.available();
  }
}

