/*

*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 6 of Geo Guide (GG) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************

*/
/*
* Team Id: GG_1110
* Author List: Aishini Bhattacharjee, Adithya Ubaradka, Deepak C Nayak, Upasana Nayak
* Filename: CatBot.ino
* Theme: GeoGuide
* Functions: setup(), handleArrayData(WiFiClient&), stopMotors(), turnLeft(), turnRight(),
             moveStraight(), buzzerlight(int), buzzerWithoutLight(int), default_action(), loop()
* Global Variables: ssid, password, port, server,
                    motor1Pin2, motor1Pin1, enable1Pin,
                    motor2Pin2, motor2Pin1, enable2Pin,
                    irSensorPin1, irSensorPin2, irSensorPin3, irSensorPin4, irSensorPin5,
                    buzzerPin, ledPin,
                    freq, pwmChannel, resolution, dutyCycle,
                    intersectionActions, TOTAL_INTERSECTIONS, arrayProcessed, client

*/

#include <WiFi.h>
#include <ArduinoJson.h>

const char* ssid = "Chaitanyas phone";
const char* password = "rainbow12345";
const int port = 8266;  // Choose any available port

WiFiServer server(port);

// Motor A
// Variable Names: motor1Pin2, motor1Pin1, enable1Pin
// Description: Pins for controlling Motor A, including input pins (IN1, IN2) and enable pin (ENA).
// Expected Values: Integers representing GPIO pin numbers.
int motor1Pin2 = 15; // IN1
int motor1Pin1 = 5;  // IN2
int enable1Pin = 21; // ENA

// Motor B
// Variable Name: motor2Pin2, motor2Pin1, enable2Pin
// Description: Pins for controlling Motor B, including input pins (IN3, IN4) and enable pin (ENB).
// Expected Values: Integers representing GPIO pin numbers.
int motor2Pin2 = 4; // IN3
int motor2Pin1 = 2; // IN4
int enable2Pin = 23; // ENB

const int irSensorPin1 = 13;
const int irSensorPin2 = 12;
const int irSensorPin3 = 14;
const int irSensorPin4 = 27;
const int irSensorPin5 = 26;

const int buzzerPin = 25;
const int ledPin = 23;

// Setting PWM properties
// Variable Names: freq, pwmChannel, resolution, dutyCycle
// Description: Variables for setting PWM properties, including frequency, channel, resolution, and duty cycle.
// Expected Values: Integers with specific values for configuring PWM.
const int freq = 30000;
const int pwmChannel = 0;
const int resolution = 8;
int dutyCycle = 200;
int TURN_SPEED = 240;


// Variable Name: intersectionActions
// Description: Dynamically allocated array to store actions for each intersection.
// Expected Values: Pointer to an array of integers, dynamically allocated during runtime once data regarding the size has been received.
int* intersectionActions = nullptr;  // Dynamically allocated array

int TOTAL_INTERSECTIONS = 0;

// Variable Name: intersectionCount
// Description: Counter variable to keep track of intersections and access the action from intersectionActions
// Expected Values: An integer representing the count of intersections.
int intersectionCount = 0;

bool arrayProcessed = false;
WiFiClient client;


/*
* Function Name: handleArrayData
* Input: WiFiClient& client - a reference to the WiFiClient object for communication
* Output: None
* Logic: This function reads an incoming JSON string from the provided WiFiClient,
*        parses the JSON, extracts the array size, dynamically allocates an array
*        of integers (intersectionActions), and populates it with the data from the JSON.
*        It also acknowledges the receipt by sending a success message to the client
*        and sets a flag (arrayProcessed) to indicate that the array has been processed.
* Example Call: handleArrayData(client);
*/
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



/*
 * Function Name: setup
 * Input: None
 * Output: None
 * Logic: Initializes and sets up the necessary configurations for the Line Following Robot using Arduino ESP32.
 *        Connects to WiFi, starts the server, handles array data (recieves the path that the bot has to follow in form an array),
 *        and initializes pins for motor control, IR sensors, buzzer, and LED.
 *        Configures LED PWM functionalities and attaches channels to GPIO pins.
 *        Prints initialization messages to Serial Monitor.
 *        Finally, calls the buzzerlight function to indicate successful initialization.
 * Example Call: setup();
 */
void setup() {
  // Begin Serial communication
  Serial.begin(115200);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.print("Connected.  ");
  Serial.print("Module IP address: ");
  Serial.println(WiFi.localIP());

  // Start the server
  server.begin();
  Serial.println("Server started");

  // Wait for a client to connect
  while (!client.connected()) {
    delay(100);
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

  // Initialize buzzer and LED pins as outputs
  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  // Configure LED PWM functionalities
  ledcSetup(pwmChannel, freq, resolution);

  // Attach the channel to the GPIO to be controlled
  ledcAttachPin(enable1Pin, pwmChannel);
  ledcAttachPin(enable2Pin, pwmChannel);

  // Print initialization message
  Serial.begin(115200);
  Serial.println("Line Following Robot Initialized");

  // Call the buzzerlight function to indicate successful initialization
  buzzerlight(1000);
}


/*
 * Function Name: stopMotors
 * Input: None
 * Output: None
 * Logic: Stops the motors by setting the PWM to 0 and turning off motor control pins.
 *        Prints "STOP" to the Serial Monitor for debugging or informational purposes.
 * Example Call: stopMotors();
 */
void stopMotors() {
  Serial.println("STOP");
  ledcWrite(pwmChannel, 0);  // Stop motors by setting PWM to 0
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
}



/*
 * Function Name: turnLeft
 * Input: None
 * Output: None
 * Logic: Turns the robot to the left based on the duty cycle for turning. Here the duty cycle is set to global variable TURN_SPEED.
 *        Prints "LEFT" to the Serial Monitor for debugging or informational purposes.
 * Example Call: turnLeft();
 */
void turnLeft() {
  Serial.println("LEFT");
  ledcWrite(pwmChannel, TURN_SPEED);
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW);   // Reverse direction for this motor
  digitalWrite(motor2Pin2, HIGH);  // Reverse direction for this motor
}

/*
 * Function Name: turnRight
 * Input: None
 * Output: None
 * Logic: Turns the robot to the left based on the duty cycle for turning. Here the duty cycle is set to global variable TURN_SPEED.
 *        Prints "RIHT" to the Serial Monitor for debugging or informational purposes.
 * Example Call: turnRight();
 */void turnRight() {
  Serial.println("RIGHT");
  ledcWrite(pwmChannel, TURN_SPEED);
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH);  // Reverse direction for this motor
  digitalWrite(motor2Pin2, LOW);   // Reverse direction for this motor
}

/*
 * Function Name: moveStraight
 * Input: None
 * Output: None
 * Logic: Moves the robot straight by setting the maximum speed using PWM.
 *        Prints "STRAIGHT" to the Serial Monitor for debugging or informational purposes.
 * Example Call: moveStraight();
 */
void moveStraight() {
  Serial.println("STRAIGHT");
  ledcWrite(pwmChannel, 255);  // Maximum speed for moving straight
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);   // Reverse direction for this motor
  digitalWrite(motor2Pin2, HIGH);  // Reverse direction for this motor
}

/*
 * Function Name: buzzerlight
 * Input: int duration - Duration of the buzzer and LED activity
 * Output: None
 * Logic: Activates the buzzer and LED simultaneously for a specified duration.
 * Example Call: buzzerlight(1000);
 */
void buzzerlight(int duration) {
  analogWrite(buzzerPin, 255);

  delay(duration);
  digitalWrite(ledPin, HIGH);
  tone(buzzerPin, 0);

  delay(duration);
  digitalWrite(ledPin, LOW);
}

/*
 * Function Name: buzzerWithoutLight
 * Input: int duration - Duration of the buzzer activity
 * Output: None
 * Logic: Activates the buzzer without accompanying LED activity for a specified duration.
 * Example Call: buzzerWithoutLight(1000);
 */
void buzzerWithoutLight(int duration) {
  analogWrite(buzzerPin, 255);

  delay(duration);
  tone(buzzerPin, 0);

  delay(duration);
}



/*
 * Function Name: default_action
 * Input: None
 * Output: None
 * Logic: Implements the default action of the robot for follwing both type1 and type2 roads using five IR sensors.
 *        Determines the robot's behavior based on the sensor pattern and performs actions at intersections
 *        as defined by the global variable 'intersectionActions'. If the robot reaches the last intersection,
 *        it executes a predefined end sequence.
 * Example Call: default_action();
 */
void default_action() {

  int sensorValue1 = digitalRead(irSensorPin1);
  int sensorValue2 = digitalRead(irSensorPin2);
  int sensorValue3 = digitalRead(irSensorPin3);
  int sensorValue4 = digitalRead(irSensorPin4);
  int sensorValue5 = digitalRead(irSensorPin5);


  // Combine sensor values into an integer for comparison
  // The sensor values are combined into a single integer to represent the current sensor pattern.
  // This is achieved by shifting each sensor value to the left by a certain number of bits and then performing bitwise OR operation.
  // For example, if sensorValue1 = 0, sensorValue2 = 1, sensorValue3 = 0, sensorValue4 = 1, sensorValue5 = 0,
  // the combined sensor pattern would be 01010 in binary or 10 in decimal.
  int sensorPattern = (sensorValue1 << 4) | (sensorValue2 << 3) | (sensorValue3 << 2) | (sensorValue4 << 1) | sensorValue5;

  // Check if the robot has reached the last intersection
  if (intersectionCount == TOTAL_INTERSECTIONS) {
    Serial.print("Last intersection");
    Serial.println(intersectionCount);

    // Get the action at the last intersection
    int currentAction = intersectionActions[TOTAL_INTERSECTIONS - 1];
    Serial.println(currentAction);

    // Perform a right turn if the last action is not a straight movement (only 2 ways to reach the start/end from the last intersection)
    if (currentAction != 0) {
      turnRight();
      delay(300);
    }

  // Variable Name: numIterations
  // Description: Dynamically allocated array to store actions for each intersection.
  // Expected Values: Pointer to an array of integers, dynamically allocated during runtime once data regarding the size has been received.
    int numIterations = 0;


    // At the last intersection, the bot has to follow the small curved section upto the stard/end box.
    // We run the same rules (turn based on sensor pattern) for a small amount of time. To achieve this we run the
    // simplified version of default_action() for a fixed number of iterations (the value was decided on trail and error)
    // until the bot has reached the start/end box. Then we stop and activate the buzzer for 5 seconds, then go into an
    // infinite while loop to keep the bot at place until turned off.
    while (numIterations < 500) {
      Serial.println(bleh);

      int sensorValue1 = digitalRead(irSensorPin1);
      int sensorValue2 = digitalRead(irSensorPin2);
      int sensorValue3 = digitalRead(irSensorPin3);
      int sensorValue4 = digitalRead(irSensorPin4);
      int sensorValue5 = digitalRead(irSensorPin5);

      // Combine sensor values into an integer for comparison
      int sensorPattern = (sensorValue1 << 4) | (sensorValue2 << 3) | (sensorValue3 << 2) | (sensorValue4 << 1) | sensorValue5;

      // The
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
          delay(1);
          break;

        case B00010:
        case B00110:
        case B10010:
        case B10011:
        case B10110:
        case B10111:
          Serial.println("Turning Right");
          turnRight();
          delay(1);
          break;

        case B10000:
        case B10100:
          Serial.println("Right type 1");
          turnRight();
          delay(2);
          moveStraight();
          delay(0.5);
          break;

        case B00001:
        case B00101:
          Serial.println("Left type 1");
          turnLeft();
          delay(2);
          moveStraight();
          delay(0.5);
          break;
      }
      numIterations++;
    }
    stopMotors();
    buzzerlight(5000);
    while (true) {
      // Serial.println("entering while loop");
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

  // Variable that stores the action to be performed at the current node, this is being initialised here.
  // The maneuver to be performed at the current node will later be assigned to this varible by fetching
  // the value from the intersectionActions array.
  int currentAction = 0;

  // Determine action based on sensor pattern using switch case
  switch (sensorPattern) {
    /*
    The patterns used here represent the on/off states of each of the 5 IR sensors.
    1st one to keep the bot on track for type 1 and turn left on type 1 (if line sensed, turn left)
    2nd one to keep the bot on track for type 2 and turn left on type 2 (if line sensed, turn left)
    3rd one to keep the bot on track for type 2 and sense the middle line (if line sensed, do nothing, keep moving straight)
    4th one to keep the bot on track for type 2 and turn right on type 2 (if line sensed, turn right)
    5th one to keep the bot on track for type 1 and turn right on type 1 (if line sensed, turn right)

    For example:
    B00000 means all the 5 IR sensors are not detecting any black lines, B00001 means the 5th
    sensor is detecting a line and the rest 4 are not.


    We considered all the possible 32 patterns and then eliminated the ones that have almost 0 chance
    of happening (B10010) and then a few more based on trail and error.

    The action to follow at a pattern was based on straigtforward logic as mentioned above. We have just considered
    edge cases for and cases where a node is detected based on the remaining patterns.
    */
    case B00000:
    case B10001:
    case B00100:
    case B10101:
      Serial.println("Moving Straight");
      moveStraight();
      break;

    // For type 2 roads
    case B01000:
    case B01100:
    case B01101:
    case B11001:
    case B11101:
      Serial.println("Turning Left");
      turnLeft();
      // A small delay so the turn is more effective
      delay(0.5);
      break;

    case B00010:
    case B00110:
    case B10011:
    case B10110:
    case B10111:
      Serial.println("Turning Right");
      turnRight();
      delay(0.5);
      break;

    // For type 1 roads
    case B10000:
    case B10100:
      Serial.println("Right type 1");
      turnRight();
      // The bot would sometimes get stuck on a type 1 road in an oscillating loop by turning left and right indefinitely.
      // To avoid it, we turn and then also move straight for a small duration so that the bot can slowly unstuck itself.
      delay(4);
      moveStraight();
      delay(1.5);
      break;

    case B00001:
    case B00101:
      Serial.println("Left type 1");
      turnLeft();
      delay(4);
      moveStraight();
      delay(1.5);
      break;

    // The below cases indicate a node, hence the bot will now retrive the action it has to perform at that particular node.
    case B01110:
    case B01111:
    case B11011:
    case B11110:
    case B11111:
      Serial.println("Stopping");

      // The retrieved action from the array is assigned to the currentAction variable that was initialised earlier.
      currentAction = intersectionActions[intersectionCount];
      // If the action is to go straight (0), the bot moves straight for an additional 300 miliseconds (subject to finetuning) so that it does
      //not count the same node twice.

      if (currentAction == 0) {
        moveStraight();
        delay(300);
      } else {
        // If the action is left/right, the bot will traverse straight for 450 miliseconds (subject to finetuning) so that it has enough room
        // to take a turn
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
        // The bot performs the turnLeft() function in the loop function for a specified amount of time (620 ms - subject to finetuning)
        // We chose this approach as it proved to be more reliable and faster than a dynamic turning scheme only based on sensor readings
          Serial.println("Turning Left");
          turnLeft();
          delay(620);   //TURNS LEFT 90 DEGREES
          // After the turn has been completed, the intersection count is incremented sigifying the visiting of a node.
          intersectionCount++;
          break;

        case 2:
          Serial.println("Turning Right");
          turnRight();
          delay(620);   // TURNS RIGHT 90 DEGREES
          intersectionCount++;
          break;
      }
    default:
      Serial.println("No matching pattern found");
      break;
  }
  // delay(5); // DELAY REDUCTION HERE INTRODUCES WAGGING BUT INCREASES ACCURACY (Wagging refers to the indefinite left-right motion on type 1 roads)
  // We have removed the delay so that we get sensor readings at the shortest time intervals possible so that the bot can take accurate decisions
}



/*
 * Function Name: loop
 * Input: None
 * Output: None
 * Logic: Repeatedly the runs the codeblock inside the function. The main function for arduino interfaces.
 *        Takes care of line following and stopping at events based on the messages recieved by the client (python script).
 *        The bot keeps performing the default action (line following) and keeps checking for the messages recieved from
 *        the client. In case a message has been recieved indicating approaching an event, the bot stops and beeps for a second.
 * Example Call: -
 */
void loop() {

  // We keep performing the default action (take decisions based on sensor readings) irrespective of everything.
  // This is the first line of code that runs on every iteration.
  default_action();

  // For stopping at the events, we are using the laptop (python script) as a client who sends messages over the common Wi-Fi network.
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

      // If the recieved data is 4, it means that nothing alarming is happening, and everything is fine. So the bot can just
      // keep follwing the path by performing the defined default actions.
      // If any other number is recieved (different events have different numbers), the bot stops and beeps for a second.
      if (receivedData != 4) {
        stopMotors();
        buzzerWithoutLight(1000);
        // We move straight for a while right after stopping so that the bot does not recieve a false signal again instructing it to stop.
        // More robust mechanisms are implemented on the client side so as to avoid this issue.
        moveStraight();
        delay(500);
        // We flush the buffer to make sure any messages sent from the client during the whole stop maneuver are
        // discarded soo as to avoid false positives.
        client.flush();
      }
    }
  } else {
    // If not connected to any client, attempt to establish a connection.
    Serial.println("No client connection yet...");
    client = server.available();
  }
}

