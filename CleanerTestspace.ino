// Screen
#include <Wire.h>
#include <TFT_eSPI.h>
#include <TFT_eWidget.h>
#include <CST816S.h>

// Operating System
#include <LittleFS.h>
#include <ESP32Time.h>

// Sleep
#include <esp_sleep.h>
#include <driver/rtc_io.h>

// MAX30102 Sensor
#include "MAX30105.h"      // Library for MAX3010x sensors
#include "heartRate.h"     // Helper functions for heart rate calculation
#include "spo2_algorithm.h"

// ================= PIN DEFINITIONS =================
#define SDA_PIN 21
#define SCL_PIN 22
#define INT_PIN 38
#define RST_PIN 33

#define ADCPIN A7 // Battery divider input (PIN 35)


// Slide switch will disconnect and re-connect battery through hardware
#define WAKEUP_GPIO GPIO_NUM_12   // Change whenever we get the actual this should be deep sleep pin
// #define LIGHT_SLEEP_WAKEUP GPIO_NUM_14 // Change whenever we assemble the device this should be pin for shake switch

// ================= MISC. DEFINITIONS =================
#define BUFFER_SIZE 50

// ================= GLOBAL OBJECTS =================
TFT_eSPI tft = TFT_eSPI();
CST816S touch(SDA_PIN, SCL_PIN, RST_PIN, INT_PIN);
TFT_eSPI_Button btnUnlock, btnBack, btnSample, btnProfile, btnConfirm;
TFT_eSPI_Button btnPrevField, btnNextField, btnMinus, btnPlus;
ESP32Time rtc(0);
MAX30105 particleSensor;

TwoWire I2C_2 = TwoWire(1);  // Second I2C bus for MAX30102

// ================= SCREEN STATE =================
enum ScreenState {
  HOME_SCREEN,
  MENU_SCREEN,
  PROFILE_SCREEN,
  SAMPLE_SCREEN
};
ScreenState currentScreen = HOME_SCREEN;

// ================= BATTERY VARIABLES =================
int batteryValue = 0;
float voltValue = 0.0f;
float estimation = 0.0f;

// ================= PROFILE VARIABLES =================
String fields[] = {"Age", "Sex", "Height", "Weight", "Diagnosis"};
int values[] = {21, 0, 65, 130, 1}; 
// defaults = (Age, Female, 65", 130lbs, Type 1)

int fieldIndex = 0;
const int NUM_FIELDS = 5;

// ================= WAKEUP VARIABLES =================
RTC_DATA_ATTR int bootCount = 0;

unsigned long lastTouch = 0;
const unsigned long SLEEP_TIMEOUT = 30000; // 30 seconds

// ================= FUNCTIONS =================
void displayHome();
String getDisplayValue(int idx);


// ================= OPERATING SYSTEM FUNCTIONS =================

void writeProfileData() {
  File file = LittleFS.open("/profile.txt", "w");
  if (!file) {
    Serial.println("Failed to open profile file for writing");
    return;
  }

  String line = "";
  for (int i = 0; i < NUM_FIELDS; i++) {
    line += String(values[i]);
    if (i < NUM_FIELDS - 1) line += ",";
  }

  file.println(line);  
  file.close();
  Serial.println("Write successful");
}

void loadProfileData() {
  File file = LittleFS.open("/profile.txt", "r");
  if (!file) {
    Serial.println("No saved profile data found");
    return;
  }

  while (file.available()) {
    String line = file.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;

    Serial.println("Loaded profile: " + line);

    int start = 0, idx = 0;
    while (idx < NUM_FIELDS && start < line.length()) {
      int comma = line.indexOf(',', start);
      if (comma == -1) comma = line.length();
      String token = line.substring(start, comma);
      values[idx++] = token.toInt();
      start = comma + 1;
    }
  file.close();
  }
}

// ================= WAKEUP FUNCTIONS =================
void pinsetup() {
  pinMode(WAKEUP_GPIO, INPUT_PULLUP);  // Set GPIO as input with pull-up
}

void sleep_param() {
  static unsigned long lastCheck = 0;

  if (millis() - lastCheck < 100) return;
  lastCheck = millis();

  int slideState = digitalRead(WAKEUP_GPIO); // slide switch
  int vibState   = digitalRead(12);          // shake switch vibration pin

  // Manual deep sleep if slide switch is OFF
  if (slideState == LOW) {
    Serial.println("Slide switch OFF → entering deep sleep...");
    particleSensor.shutDown();    // turn off LEDs and ADC
    digitalWrite(RST_PIN, LOW);   // hold TFT in reset
    delay(50);
    esp_deep_sleep_start();       // ESP32 enters deep sleep
  }


  // Automatic deep sleep after 30s inactivity
  if (slideState == HIGH && (millis() - lastTouch > SLEEP_TIMEOUT)) {
    Serial.println("Idle timeout → entering deep sleep...");
    delay(100);

    // Configure GPIO12 as external wake-up source
    esp_sleep_enable_ext1_wakeup((1ULL << 12), ESP_EXT1_WAKEUP_ANY_HIGH);

    // Shut down peripherals for clean power down
    pinMode(RST_PIN, OUTPUT);
    digitalWrite(RST_PIN, LOW);   
    Wire.end();
    pinMode(SDA_PIN, INPUT);
    pinMode(SCL_PIN, INPUT);
    pinMode(INT_PIN, INPUT_PULLUP);

    Serial.println("Going into deep sleep now...");
    delay(50);
    esp_deep_sleep_start();
  }
}



// ================= DISPLAY FUNCTIONS =================

void displayBattery() {
  batteryValue = analogRead(ADCPIN);
  voltValue = (batteryValue * 3.3f) / 4095.0f;
  estimation = (voltValue / 1.96f) * 100.0f;

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(2);
  tft.drawString(String(estimation, 0) + "%", tft.width() / 2, 20);
}

void drawBackArrow(int cx, int cy) {
  int16_t shaft = 20; // body length
  int head = 10;      // arrowhead size

  int startX = cx + shaft/2; // shift right to center
  int endX   = cx - shaft/2;
  
  // Horizontal shaft (centered)
  tft.drawLine(startX, cy, endX, cy, TFT_WHITE);
  // Arrowhead (centered at endX)
  tft.drawLine(endX, cy, endX + head, cy - head, TFT_WHITE);
  tft.drawLine(endX, cy, endX + head, cy + head, TFT_WHITE);
}



void displayHome() {
  currentScreen = HOME_SCREEN;
  tft.fillScreen(TFT_BLACK);

  // Full-screen invisible touch button 
  btnUnlock.initButton(&tft,
      tft.width()/2, tft.height()/2,
      tft.width(), tft.height(),
      TFT_BLACK, TFT_BLACK, TFT_BLACK,
      "", 1);

  displayBattery();

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);

  tft.setTextSize(2);
  tft.drawString(rtc.getTime("%A"), tft.width()/2, 85);

  tft.setTextSize(3);
  tft.drawString(rtc.getTime("%H:%M"), tft.width()/2, 120);

  tft.setTextSize(2);
  tft.drawString(rtc.getTime("%B %d, %G"), tft.width()/2, 150);

  tft.setTextSize(1);
  tft.drawString("Tap to unlock", tft.width()/2, 220);
}

void displayMenu() {
  currentScreen = MENU_SCREEN;
  tft.fillScreen(TFT_BLACK);

  btnSample.initButton(&tft, 60, 120, 90, 40, TFT_WHITE, TFT_PINK, TFT_WHITE, "Sample", 2);
  btnSample.drawButton();

  btnProfile.initButton(&tft, 180, 120, 90, 40, TFT_WHITE, TFT_SKYBLUE, TFT_WHITE, "Profile", 2);
  btnProfile.drawButton();

  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30, TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);
}

void displayProfileMenu() {
  currentScreen = PROFILE_SCREEN;
  tft.fillScreen(TFT_BLACK);

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);

  // FIELD LABEL (top middle)
  tft.setTextSize(2);
  tft.drawString(fields[fieldIndex], tft.width()/2, 70);

  // FIELD VALUE (center)
  tft.setTextSize(3);
  tft.drawString(getDisplayValue(fieldIndex), tft.width()/2, 130);

  // < and > buttons
  btnPrevField.initButton(&tft, 45, 70, 40, 40,
                          TFT_WHITE, TFT_BLACK, TFT_WHITE,
                          "<", 2);
  btnNextField.initButton(&tft, 195, 70, 40, 40,
                          TFT_WHITE, TFT_BLACK, TFT_WHITE,
                          ">", 2);
  btnPrevField.drawButton();
  btnNextField.drawButton();

  // + and - buttons
  btnMinus.initButton(&tft, 45, 130, 50, 45,
                      TFT_WHITE, TFT_BLACK, TFT_WHITE,
                      "-", 3);
  btnPlus.initButton(&tft, 195, 130, 50, 45,
                     TFT_WHITE, TFT_BLACK, TFT_WHITE,
                     "+", 3);
  btnMinus.drawButton();
  btnPlus.drawButton();

  // Back button
  btnBack.initButton(&tft, tft.width()/2, 30,
                     50, 30,
                     TFT_BLACK, TFT_BLACK, TFT_WHITE,
                     "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);

  // Save button
  btnConfirm.initButton(&tft, 
                        tft.width()/2, 190,
                        70, 40,
                        TFT_WHITE, TFT_PINK, TFT_WHITE,
                        "Save", 2);
  btnConfirm.drawButton();
}

// Profile Menu with increment/decrement options
String getDisplayValue(int idx) {
  if (idx == 1) {  // Sex
    return (values[1] == 0) ? "F" : "M";
  }
  if (idx == 4) {  // Diagnosis
    if (values[4] == 1) return "T1";
    if (values[4] == 2) return "T2";
    return "None";
  }
  return String(values[idx]);
}

void displaySampleMenu() {
  currentScreen = SAMPLE_SCREEN;
  tft.fillScreen(TFT_BLACK);

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(3);
  tft.drawString("Measuring...", tft.width()/2, tft.height()/2);

  // Run the sampling process
  runSampling();

  // Back button
  btnBack.initButton(&tft, tft.width()/2, 30,
                     50, 30,
                     TFT_BLACK, TFT_BLACK, TFT_WHITE,
                     "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);
}

// ================= SENSOR SAMPLING FUNCTION =================
void runSampling() {
  Serial.println("Starting MAX30102 sampling...");

  uint32_t irBuffer[BUFFER_SIZE];
  uint32_t redBuffer[BUFFER_SIZE];

  // Fill buffer with samples
  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (!particleSensor.check()) {
      delay(1);
    }
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
  }

  // Variables for results
  int32_t spo2;
  int8_t validSPO2;
  int32_t heartRate;
  int8_t validHeartRate;

  // Run MAXIM algorithm
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, BUFFER_SIZE, redBuffer,
    &spo2, &validSPO2, &heartRate, &validHeartRate
  );

  // Print to Serial Monitor
  Serial.print("Heart Rate: ");
  if (validHeartRate) Serial.print(heartRate);
  else Serial.print("Invalid");

  Serial.print(" bpm\tSpO2: ");
  if (validSPO2) Serial.print(spo2);
  else Serial.print("Invalid");
  Serial.println(" %");

  // Display on screen too (optional)
  tft.fillScreen(TFT_BLACK);
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(3);

  if (validHeartRate && validSPO2) {
    tft.drawString("HR: " + String(heartRate) + " bpm", tft.width()/2, 100);
    tft.drawString("SpO2: " + String(spo2) + " %", tft.width()/2, 140);
  } else {
    tft.drawString("Reading Invalid", tft.width()/2, 120);
  }

  delay(1000); // Wait before allowing another sample
}


// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(200);

  bootCount++;
  Serial.printf("Boot number: %d\n", bootCount);
  pinsetup();
  esp_sleep_enable_ext1_wakeup((1ULL << 12), ESP_EXT1_WAKEUP_ANY_HIGH);

  tft.init();
  tft.setRotation(0);
  Wire.begin(SDA_PIN, SCL_PIN);
  touch.begin();

  I2C_2.begin(7, 8);
  if (!particleSensor.begin(I2C_2, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found. Check connections!");
    while (1);
  }


  // Configure sensor settings to default
  particleSensor.setup();

  // Optional: customize parameters
  // particleSensor.setup(60, 4, 2, 100, 400, 4096); 
  // (sampleRate, ledBrightness, pulseWidth, adcRange, sampleAverage)

  if (!LittleFS.begin()) {
    Serial.println("An error has occurred while mounting LittleFS");
    delay(1000);
    return;
  }
  Serial.println("LittleFS mounted successfully.");

  rtc.setTime(0, 21, 11, 3, 11, 2025); // Second, Minute, Hour, Day, Month, Year
  loadProfileData();  // Test reading saved data

  displayHome();
  lastTouch = millis();
}

// ================= LOOP =================
void loop() {
  if (touch.available()) {
    lastTouch = millis();
    uint16_t x = touch.data.x, y = touch.data.y;
    int mappedX = map(x, 0, 240, 0, tft.width());
    int mappedY = map(y, 0, 240, 0, tft.height());

    if (currentScreen == HOME_SCREEN)
      btnUnlock.press(btnUnlock.contains(mappedX, mappedY));
    else if (currentScreen == MENU_SCREEN) {
      btnBack.press(btnBack.contains(mappedX, mappedY));
      btnProfile.press(btnProfile.contains(mappedX, mappedY));
      btnSample.press(btnSample.contains(mappedX, mappedY));
    }
    else if (currentScreen == PROFILE_SCREEN) {
      btnPrevField.press(btnPrevField.contains(mappedX, mappedY));
      btnNextField.press(btnNextField.contains(mappedX, mappedY));
      btnMinus.press(btnMinus.contains(mappedX, mappedY));
      btnPlus.press(btnPlus.contains(mappedX, mappedY));
      btnBack.press(btnBack.contains(mappedX, mappedY));
      btnConfirm.press(btnConfirm.contains(mappedX, mappedY));
    }
    else if (currentScreen == SAMPLE_SCREEN) {
      btnBack.press(btnBack.contains(mappedX, mappedY));
    }
  } 
  else {
    btnUnlock.press(false);
    btnBack.press(false);
    btnProfile.press(false);
    btnSample.press(false);
    btnConfirm.press(false);
    btnPrevField.press(false);
    btnNextField.press(false);
    btnMinus.press(false);
    btnPlus.press(false);
  }

  // === ACTIONS BY SCREEN ===
  if (currentScreen == HOME_SCREEN) {
    if (btnUnlock.justReleased()) displayMenu();
  }
  else if (currentScreen == MENU_SCREEN) {
    if (btnBack.justReleased()) displayHome();
    if (btnProfile.justReleased()) displayProfileMenu();
    if (btnSample.justReleased()) displaySampleMenu();
  }
  else if (currentScreen == PROFILE_SCREEN) {

    if (btnBack.justReleased()) {
      loadProfileData();
      displayMenu();
    }

    if (btnConfirm.justReleased()) {
      writeProfileData();
      displayMenu(); 
    }

    if (btnPrevField.justReleased()) {
      fieldIndex = (fieldIndex - 1 + NUM_FIELDS) % NUM_FIELDS;
      displayProfileMenu();
    }

    if (btnNextField.justReleased()) {
      fieldIndex = (fieldIndex + 1) % NUM_FIELDS;
      displayProfileMenu();
    }

    if (btnMinus.justReleased()) {
      // Sex toggle
      if (fieldIndex == 1)
        values[1] = !values[1];

      // Diagnosis cycle
      else if (fieldIndex == 4)
        values[4] = (values[4] - 1 + 3) % 3;

      else
        values[fieldIndex]--;

      displayProfileMenu();
    }

    if (btnPlus.justReleased()) {
      if (fieldIndex == 1)
        values[1] = !values[1];

      else if (fieldIndex == 4)
        values[4] = (values[4] + 1) % 3;

      else
        values[fieldIndex]++;

      displayProfileMenu();
  }
}
  else if (currentScreen == SAMPLE_SCREEN) {
    if (btnBack.justReleased()) {
      displayMenu();
    }
  }

  sleep_param();  // checks for deep sleep via slide switch or light sleep via timer
  delay(15);

}
