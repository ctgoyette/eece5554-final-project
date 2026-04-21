// THIS IS THE MOBILE TAG (on the robot)
//


#define TAG_ID 0
const String TAG_ID_STRING = "0";
int curr_vals[3] = {-1, -1, -1};

// UWB stands for Ultra-Wideband and it's how the beacons ("anchors") find the distance to this "tag" board that's sitting on the robot
// Here's a link for reference on thier "AT" protocol: https://reyax.com//upload/products_download/download_file/AT_Command_RYUW122.pdf
// Tag configuration
const String TAG_ADDRESS = "TAG00010";        // This tag's address (8 bytes) — must match what anchors expect
const String ANCHOR_ADDRESS = "ANCH0010";     // Anchor to talk to (8 bytes)
const String NETWORK_ID = "NEULUNAB";         // Must match on both sides
const String PASSWORD = "FABC0002EEDCAA90FABC0002EEDCAA90";
const String MODE = "0";   // set module to TAG mode

HardwareSerial uwb1(1);
// Sends an AT command to the module and prints it to Serial
void sendCommand(const String &cmd) {
  uwb1.print(cmd + "\r\n");
  // Serial.print("Sent command: ");
  // Serial.println(cmd);
  String response = "";
  // Wait until we get a response
  while (response.length() == 0) {
    if (uwb1.available()) {
      response = uwb1.readStringUntil('\n');
    }
    delay(10);
  }
  delay(10);
}

void tagSend(const String &payload) {
  char cmd[64];
  snprintf(cmd, sizeof(cmd), "AT+TAG_SEND=%d,%s", payload.length(), payload.c_str());
  sendCommand(String(cmd));
}

void setup() {
  Serial.begin(9600);
  uwb1.begin(9600, SERIAL_8N1, 26, 25);

  delay(1000);
  // Serial.println("Setting up TAG...");
  delay(1000);

  while (!uwb1)
      delay(10);


  // Configure the UWB module for TAG mode
  sendCommand("AT");
  sendCommand("AT");
  sendCommand("AT+IPR=9600");
  sendCommand("AT+MODE=" + MODE);
  sendCommand("AT+NETWORKID=" + NETWORK_ID);
  sendCommand("AT+ADDRESS=" + TAG_ADDRESS);
  sendCommand("AT+CPIN=" + PASSWORD);
  sendCommand("AT+CHANNEL=5");
  sendCommand("AT+VER?");
  sendCommand("AT+BANDWIDTH=0");
  sendCommand("AT+CRFOP?");
  sendCommand("AT+RSSI=0");

  // Serial.println("Tag module configured. Listening for anchors...");
}

void loop() {
  // Print anything received from anchors
  if (uwb1.available()) {
    String response = uwb1.readStringUntil('\n');
    response.trim();
    if (response.length() > 0) {
      // Split response by comma
        int startIndex = 0;
        int commaIndex;

        while ((commaIndex = response.indexOf(',', startIndex)) != -1) {
          startIndex = commaIndex + 1;
        }
        String curr_reading = response.substring(startIndex);
        String index = String(curr_reading[0]);
        if (curr_reading.substring(1).toInt() == 0){
          return;
        }
        int value = curr_reading.substring(1).toInt();
        Serial.print(index);
        Serial.print(", ");
        Serial.println(value);        
    }
  }

  // Forward serial monitor input to module (for manual AT commands)
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    sendCommand(command);
  }
}