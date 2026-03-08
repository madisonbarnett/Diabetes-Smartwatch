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
#include "Arduino.h"
#include "arduinoFFT.h"
#include "math.h"

// HELPFUL LINKS
// Button colors: https://doc-tft-espi.readthedocs.io/tft_espi/colors/
// RTCTime Commands: https://github.com/fbiego/ESP32Time
// Create icons for buttons: https://javl.github.io/image2cpp/ 

// NOTES
// Sleep might not be working LOL
// add PM/AM function? Convert to 24 scale time before saving time in user settings?
// Look into adding history menu
// Improve look of GUI
// Test slider function in TFT_eWidget library
// Test slide right/left function in CST library


// ================= PIN DEFINITIONS =================
#define SDA_PIN 21
#define SCL_PIN 22
#define INT_PIN 38
#define RST_PIN 32

#define ADCPIN A7 // Battery divider input (PIN 35)


// Slide switch will disconnect and re-connect boost converter through hardware
#define SHAKE_PIN GPIO_NUM_26   // Shake switch pin

// ================= MISC. DEFINITIONS =================
#define BUFFER_SIZE 50

// ================= GLOBAL OBJECTS =================
TFT_eSPI tft = TFT_eSPI();
CST816S touch(SDA_PIN, SCL_PIN, RST_PIN, INT_PIN);
TFT_eSPI_Button btnUnlock, btnBack, btnSample, btnProfile, btnSettings, btnConfirm;
TFT_eSPI_Button btnPrevField, btnNextField, btnMinus, btnPlus;
ESP32Time rtc(0);
//MAX30105 particleSensor;

// Swiping function - testing
const char* menuItems[] = {"PROFILE", "SAMPLE", "SETTINGS"};
uint16_t menuColors[] = {TFT_PINK, TFT_SKYBLUE, TFT_VIOLET}; 
const int MENU_COUNT = 3; // Update if History menu is added
int menuIndex = 0;

// Debounce / lockout for gestures so one swipe doesn't advance 5 times
unsigned long lastGestureMs = 0;
const unsigned long GESTURE_LOCKOUT_MS = 250;

// One big "card" button (tap to open)
TFT_eSPI_Button btnMenuCard;


TwoWire I2C_2 = TwoWire(1);  // Second I2C bus for MAX30102

// ================= SCREEN STATE =================
enum ScreenState {
  HOME_SCREEN,
  MENU_SCREEN,
  PROFILE_SCREEN,
  SAMPLE_SCREEN,
  SETTINGS_SCREEN
};
ScreenState currentScreen = HOME_SCREEN;

unsigned long lastSampletime = 0;
MAX30105 particleSensor;
// Defines the arrays for later calculations, can be changed, however these have to be very large to be able to hold samples, will affect storage.
// Changed to 5000 from 3000
float irarray[5000];           // raw IR values
int arraysize[5000];           // index array
float heartbeattime[5000];
unsigned long timearray[5000]; // timestamps
unsigned long P2PTime_Array[5000];//This is for the p2p time
int i = 0; // counts the numbers of samples VERY VERY VERY IMPORTANT
unsigned long initial;
bool startRequested = false;
// ------------ Globals for history in the future as well as inputs into the AI model
float FIRST_DERIVMAX = 0; // The first derivative max
float FIRST_DERIV = 0;    // The first derivative for the loop
float mean = 0;            // basic mean for skewness
float stddev = 0;          // basic std deviation for skewness
float skewness = 0;        // skewness
float iqr = 0;             // Interquartile Range
float teager = 0;          // Teager energy for calculating mac
float teager_max = 0;      // This is the mac teager energy
float meanP2Pinterval=0;   // Finds the mean P2P interval
float stddevTime=0; // Standard deviation for P2P interval

int h=0;
int SAMPLES;
#define SAMPLING_FREQUENCY 100 // this will need to change depending on the setup
float entropy; 
double *vReal = NULL;
double *vImag = NULL;
ArduinoFFT<double> *FFT = NULL;
//----------------------This will be for heartrate globals im sure this can be all done better but this is how I know how to do it
int p=0;; // this is anouther counter
const byte RATE_SIZE = 4; //Increase this for more averaging. 4 is good.
byte rates[RATE_SIZE]; //Array of heart rates
byte rateSpot = 0;
long lastBeat = 0; //Time at which the last beat occurred

float beatsPerMinute;
int beatAvg;

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

// ================= SETTINGS VARIABLES =================
String settings_fields[] = {"Hour", "Minute", "Month", "Day", "Year"};
// [0]=hour (0-23), [1]=minute (0-59), [2]=month (1-12), [3]=day (1-31), [4]=year (e.g. 2025)
int settings_values[] = {0, 0, 1, 1, 2025};

int settings_fieldIndex = 0;
const int SETTINGS_NUM_FIELDS = 5;
bool settings_loaded = false;

// ================= WAKEUP VARIABLES =================
RTC_DATA_ATTR int bootCount = 0;

unsigned long lastTouch = 0;
const unsigned long SLEEP_TIMEOUT = 1200000; // 60 seconds

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
  pinMode((int)SHAKE_PIN, INPUT_PULLDOWN);  // Set GPIO to be always high, wake when shaken to low
}

int print_wakeup_reason(){
  // tells you why board exited deep sleep
  esp_sleep_wakeup_cause_t wakeup_reason;

  int reason = 0;
  wakeup_reason = esp_sleep_get_wakeup_cause();

  switch(wakeup_reason) {
    case ESP_SLEEP_WAKEUP_EXT0 : Serial.println("Wakeup caused by external signal using RTC_IO"); reason = 2; break;
    case ESP_SLEEP_WAKEUP_EXT1 : Serial.println("Wakeup caused by external signal using RTC_CNTL"); reason = 3; break;
    case ESP_SLEEP_WAKEUP_TIMER : Serial.println("Wakeup caused by timer"); reason = 4; break;
    case ESP_SLEEP_WAKEUP_TOUCHPAD : Serial.println("Wakeup caused by touchpad"); reason = 5; break;
    case ESP_SLEEP_WAKEUP_ULP : Serial.println("Wakeup caused by ULP program"); reason = 6; break;
    default : Serial.printf("Wakeup was not caused by deep sleep: %d\n",wakeup_reason); reason = -1; break;
  }
  return reason;
}

void sleep_param() {
  if (millis() - lastTouch < SLEEP_TIMEOUT) return;

  Serial.println("Going to sleep...");

  // Turn off display & touch power // IDK MAYBE ADD THIS BACK IS POWER IS DRAINING TOO MUCH BUT WAS CAUSING ISSUES
  // digitalWrite(25, LOW);

  // Turn off MAX30102 // ADD BACK WHEN SENSOR IS ATTACHED FOR TESTING - IF ISSUES ARISE, DELETE THIS
  //particleSensor.shutDown();

  delay(100);
  esp_deep_sleep_start();
}

bool max30102_active = false;

void enableMAX30102() {
  if (max30102_active) return;

  Serial.println("Enabling MAX30102...");

  // Start I2C bus (safe to call multiple times)
  I2C_2.begin(7, 8);

  if (!particleSensor.begin(I2C_2, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found!");
    return;
  }

  // MAX30102 setup: LED current, sample rate = 100 HZ pulse width has to be edited, have yet to find a good compromise
  particleSensor.setup();//particleSensor.setup(60, 1, 2, 100, 411, 4096); // this is fine for now, it is important to note that I have not yet tuned this completley
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.setPulseAmplitudeRed(0x1F);

  // Clear buffers so old data doesn't leak
  // memset(ppg_red_buffer, 0, sizeof(ppg_red_buffer));
  // memset(ppg_ir_buffer, 0, sizeof(ppg_ir_buffer));
  // ppg_index = 0;
  // buffer_filled = false;

  max30102_active = true;
}

void disableMAX30102() {
  if (!max30102_active) return;

  Serial.println("Disabling MAX30102...");
  particleSensor.shutDown();
  max30102_active = false;
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

// Manually drawing Icons, may change to icons stored on flash later
void drawMenuIcon(int idx, int cx, int cy, int size, uint16_t color) {
  // Simple built-in icons (no bitmaps needed)
  // idx: 0=Profile, 1=Sample, 2=Settings

  if (idx == 0) {
    // Profile icon: head + shoulders
    tft.fillCircle(cx, cy - size/5, size/4, color);
    tft.fillRoundRect(cx - size/3, cy, 2*size/3, size/3, 6, color);
  }
  else if (idx == 1) {
    // Sample icon: droplet (simple teardrop-ish)
    tft.fillCircle(cx, cy, size/3, color);
    tft.fillTriangle(cx, cy - size/2,
                     cx - size/3, cy,
                     cx + size/3, cy,
                     color);
  }
  else if (idx == 2) {
    // Settings icon: gear-ish (circle + spokes)
    int rOuter = size / 3;        // outer radius
    int rInner = rOuter - 6;      // ring thickness (tweak 5-8)
    int toothL = 8;               // tooth length (outward)
    int toothW = 6;               // tooth width

    // Outer ring (filled)
    tft.fillCircle(cx, cy, rOuter, color);
    // Hollow it out to make a ring
    tft.fillCircle(cx, cy, rInner, TFT_VIOLET);    // assumes card interior is drawn on colored bg? (see note below)

    // Teeth: 8 teeth (N, NE, E, SE, S, SW, W, NW)
    // Top
    tft.fillRoundRect(cx - toothW/2, cy - rOuter - toothL + 2, toothW, toothL, 2, color);
    // Bottom
    tft.fillRoundRect(cx - toothW/2, cy + rOuter - 2, toothW, toothL, 2, color);
    // Left
    tft.fillRoundRect(cx - rOuter - toothL + 2, cy - toothW/2, toothL, toothW, 2, color);
    // Right
    tft.fillRoundRect(cx + rOuter - 2, cy - toothW/2, toothL, toothW, 2, color);

    // Diagonals
    // ----- Diagonal teeth that ACTUALLY touch the ring -----
    int knob = 8;  // size of diagonal tooth block

    // radius to the circle edge at 45 degrees
    int rDiag = (int)(rOuter * 0.7071f) - 1;  // r / sqrt(2)

    // NE
    tft.fillRoundRect(cx + rDiag - knob/2, cy - rDiag - knob/2,
                      knob, knob, 2, color);

    // SE
    tft.fillRoundRect(cx + rDiag - knob/2, cy + rDiag - knob/2,
                      knob, knob, 2, color);

    // SW
    tft.fillRoundRect(cx - rDiag - knob/2, cy + rDiag - knob/2,
                      knob, knob, 2, color);

    // NW
    tft.fillRoundRect(cx - rDiag - knob/2, cy - rDiag - knob/2,
                      knob, knob, 2, color);



    // Center hub outline (optional, makes it pop)
    tft.drawCircle(cx, cy, rInner - 2, color);
    tft.drawCircle(cx, cy, (rInner - 2) / 2, color);
  }
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

// Menu with all three buttons
void displayMenu() {
  currentScreen = MENU_SCREEN;
  tft.fillScreen(TFT_BLACK);

  btnSample.initButton(&tft, 60, 120, 90, 40, TFT_WHITE, TFT_PINK, TFT_WHITE, "Sample", 2);
  btnSample.drawButton();

  btnProfile.initButton(&tft, 180, 120, 90, 40, TFT_WHITE, TFT_SKYBLUE, TFT_WHITE, "Profile", 2);
  btnProfile.drawButton();

  btnSettings.initButton(&tft, 120, 180, 90, 40, TFT_WHITE, TFT_YELLOW, TFT_WHITE, "Settings", 2);
  btnSettings.drawButton();

  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30, TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);
}

// Menu with carosel functions
void drawMenuDots(int activeIdx) {
  int y = 205;
  int spacing = 14;
  int startX = (tft.width() / 2) - ((MENU_COUNT - 1) * spacing / 2);

  for (int i = 0; i < MENU_COUNT; i++) {
    int x = startX + i * spacing;
    // filled circle for active, hollow for others
    if (i == activeIdx) tft.fillCircle(x, y, 4, TFT_WHITE);
    else tft.drawCircle(x, y, 4, TFT_WHITE);
  }
}

void drawMenuCarousel() {
  currentScreen = MENU_SCREEN;
  tft.fillScreen(TFT_BLACK);

  // Back button at top
  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30, TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);

  // Location and size of each button
  int cardW = 110;
  int cardH = 110;
  int cardX = tft.width()/2;
  int cardY = 120;

  uint16_t fill = menuColors[menuIndex];
  uint16_t outline = TFT_WHITE;

  // Make the card tappable (invisible button over it)
  btnMenuCard.initButton(&tft, cardX, cardY, cardW - 20, cardH - 20,
                         TFT_BLACK, TFT_BLACK, TFT_BLACK, "", 1);
  //btnMenuCard.drawButton();

  // Filled rounded square
  tft.fillRoundRect(cardX - cardW/2, cardY - cardH/2, cardW, cardH, 16, fill);
  tft.drawRoundRect(cardX - cardW/2, cardY - cardH/2, cardW, cardH, 16, outline);

  // Icon (slightly toward top)
  int iconCx = cardX;
  int iconCy = cardY - 5; // change this value to move location of icon up or down
  drawMenuIcon(menuIndex, iconCx, iconCy, 80, TFT_WHITE);

  // Label near bottom of square
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, fill);
  tft.setTextSize(2);
  tft.drawString(menuItems[menuIndex], cardX, cardY + 45);

  // Dots indicator
  drawMenuDots(menuIndex);
}

void openSelectedMenuItem() {
  Serial.print("Opening: ");
  Serial.println(menuItems[menuIndex]);

  if (menuIndex == 0) displayProfileMenu();
  else if (menuIndex == 1) displaySampleMenu();
  else if (menuIndex == 2) displaySettingsMenu();
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
  
  enableMAX30102(); // turn sensor on 

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(3);
  tft.drawString("Measuring...", tft.width()/2, tft.height()/2);

  // Run the sampling process
  runSampling();

    tft.fillScreen(TFT_BLACK);
    tft.setTextDatum(MC_DATUM);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setTextSize(2);
    tft.drawString("Done", tft.width()/2, 120);
    for (int j = 0; j < i; j++) {
        Serial.print(arraysize[j]);
        Serial.print(",");
        Serial.println(irarray[j]);
      }
      Serial.println("All data printed.");
      POWEROF2();
      FIRST_DERIV_MAX();
      SKEWNESS_CALC();
      FFT_SPECTRAL();
      IQR_CALC();
      TEAGER_ENERGY();
      TIME_STD_DEV();
      Serial.print("First_derixmax: ");
      Serial.print(FIRST_DERIVMAX);
      Serial.print(" ");
      Serial.print("Skewness: ");
      Serial.print(skewness);
      Serial.print(" ");
      Serial.print("iqr: ");
      Serial.print(iqr);
      Serial.print(" ");
      Serial.print("Teager_max: ");
      Serial.print(teager_max);
      Serial.print(" ");
      Serial.print("beatAvg: ");
      Serial.print(beatAvg);
      Serial.print(" ");
      Serial.print("stdDevTime: ");
      Serial.print(stddevTime);
      Serial.print(" ");
      Serial.print("entropy: ");
      Serial.print(entropy);
      Serial.print(" ");
      Serial.print("h: ");
      Serial.print(h);

  // Back button
  btnBack.initButton(&tft, tft.width()/2, 30,
                     50, 30,
                     TFT_BLACK, TFT_BLACK, TFT_WHITE,
                     "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);
}

String getSettingsDisplayValue(int idx) {
  // Hour - formats with leading 0 (8 -> 08)
  if (idx == 0) return (settings_values[0] < 10 ? "0" : "") + String(settings_values[0]);
  // Minute 
  if (idx == 1) return (settings_values[1] < 10 ? "0" : "") + String(settings_values[1]);
  return String(settings_values[idx]);
}

void displaySettingsMenu() {
  currentScreen = SETTINGS_SCREEN;
  tft.fillScreen(TFT_BLACK);

  // MAYBE CHANGE THIS TO ADD PM and AM ???
  // Retrieve current time
  if (!settings_loaded) {
    settings_values[0] = rtc.getHour();       // 0-23
    settings_values[1] = rtc.getMinute();     // 0-59
    settings_values[2] = rtc.getMonth() + 1;  // 1-12
    settings_values[3] = rtc.getDay();        // 1-31
    settings_values[4] = rtc.getYear();       // e.g. 2026
    settings_loaded = true;
  }

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);

  // FIELD LABEL (top middle)
  tft.setTextSize(2);
  tft.drawString(settings_fields[settings_fieldIndex], tft.width()/2, 70);

  // FIELD VALUE (center)
  tft.setTextSize(3);
  tft.drawString(getSettingsDisplayValue(settings_fieldIndex), tft.width()/2, 130);

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
                        TFT_WHITE, TFT_VIOLET, TFT_WHITE,
                        "Save", 2);
  btnConfirm.drawButton();
}

// Calculate the first derivative max 
void FIRST_DERIV_MAX() {
  for (int j = 1; j < i; j++) {
    FIRST_DERIV = (irarray[j] - irarray[j - 1]) / ((timearray[j] - timearray[j - 1]) / 1000.0); // time is converted here in seconds
    if (FIRST_DERIV > FIRST_DERIVMAX) {
      FIRST_DERIVMAX = FIRST_DERIV; // Makes the first derivative max if the first derivative is larger then the one before
    }
  }
}

// Calculate the skewness
void SKEWNESS_CALC() {
  // calculate mean
  mean = 0;
  for (int j = 0; j < i; j++) {
    mean += irarray[j];
  }
  mean = mean / (i + 0.0); // finds the mean

  // Calculate STD Deviation
  stddev = 0;
  for (int j = 0; j < i; j++) {
    stddev += sq((irarray[j] - mean));
  }
  stddev = sqrt(stddev / (i + 0.0));

  // finds skewness
  skewness = 0;
  for (int j = 0; j < i; j++) {
    float temp = (irarray[j] - mean) / stddev;
    skewness += temp * temp * temp;
  }
  skewness = (i + 0.0) / ((i - 1.0) * (i - 2.0)) * skewness;
}

// Calculate interquartile range
void IQR_CALC() {
  static float temp[3000];
  for (int j = 0; j < i; j++) {
    temp[j] = irarray[j];
  }
  for (int a = 0; a < i - 1; a++) {
    for (int b = a + 1; b < i; b++) {
      if (temp[b] < temp[a]) {
        float t = temp[a];
        temp[a] = temp[b];
        temp[b] = t;
      }
    }
  }
  int q1_index = (int)(0.25 * (i - 1));
  int q3_index = (int)(0.75 * (i - 1));

  float Q1 = temp[q1_index];
  float Q3 = temp[q3_index];
  iqr = Q3 - Q1;
}

// Find the maximum Teager energy
void TEAGER_ENERGY() {
  teager_max = 0;
  for (int j = 1; j < i - 1; j++) {
    teager = irarray[j] * irarray[j] - irarray[j - 1] * irarray[j + 1];
    if (teager > teager_max) {
      teager_max = teager;
    }
  }
}

void POWEROF2() {
  h = 1;
  while ((h << 1) <= i) {
    h<<= 1;
  }
}
void TIME_STD_DEV() {
  // calculate mean
  meanP2Pinterval = 0;
  for (int j = 0; j < p; j++) {
    meanP2Pinterval += P2PTime_Array[j];
  }
  meanP2Pinterval = meanP2Pinterval / (p + 0.0); // finds the mean

  // Calculate STD Deviation
  stddevTime = 0;
  for (int j = 0; j < p; j++) {
    stddevTime += sq((P2PTime_Array[j] -  meanP2Pinterval));
  }
  stddevTime = sqrt(stddev / (p + 0.0));
}
void FFT_SPECTRAL(){
SAMPLES=h;          // Must be a power of 2
  vReal = new double[SAMPLES];  // dynamically allocate
  vImag = new double[SAMPLES];
  if(FFT != NULL) delete FFT;  // clean up old one
  FFT = new ArduinoFFT<double>(vReal, vImag, SAMPLES, SAMPLING_FREQUENCY);
for(int j=0; j<SAMPLES; j++){
  vReal[j]=irarray[j];
  vImag[j]=0;
}
  FFT->windowing(FFTWindow::Hamming, FFTDirection::Forward);
  FFT->compute(FFTDirection::Forward);
  FFT->complexToMagnitude();
   entropy = calculateSpectralEntropy();

   delete FFT;
  FFT = NULL;
}
float calculateSpectralEntropy() {
  float sum = 0;
  for(int i=1; i<SAMPLES/2; i++) {
    sum += vReal[i];
  }

   entropy = 0;
  for(int l=1; l<SAMPLES/2; l++) {
    float k = vReal[l] / sum;  // Normalize to probability
    if(k > 0) {
      entropy -= k * log(k);
    }
  }
    
  return entropy;
}

// ================= SENSOR SAMPLING FUNCTION =================
int runSampling() {
  startRequested = true;
  i = 0;
  initial = millis();
  Serial.println("Starting now");

  // reset values (leave as-is)
  FIRST_DERIVMAX = 0;
  FIRST_DERIV = 0;
  mean = 0;
  stddev = 0;
  skewness = 0;
  iqr = 0;
  teager = 0;
  teager_max = 0;
  p = 0;
  lastBeat = 0;
  beatAvg = 0;
  h = 0;

  // run for 30 seconds (or until array fills)
  while ((millis() - initial) < 30000 && i < 3000) {
    particleSensor.check();

    while (particleSensor.available() && i < 3000) {
      uint32_t ir = particleSensor.getIR();

      irarray[i] = ir;
      timearray[i] = millis();

      if (checkForBeat(ir) == true) {
        long delta = millis() - lastBeat;
        lastBeat = millis();

        if (p < 3000) {
          P2PTime_Array[p] = delta / 1000; // seconds (keeping his behavior)
          p++;
        }

        beatsPerMinute = 60 / (delta / 1000.0);
        if (beatsPerMinute < 255 && beatsPerMinute > 20) {
          rates[rateSpot++] = (byte)beatsPerMinute;
          rateSpot %= RATE_SIZE;

          beatAvg = 0;
          for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
          beatAvg /= RATE_SIZE;
        }
      }

      arraysize[i] = i;
      i++;

      particleSensor.nextSample();
    }

    delay(1); // yield so UI/WDT doesn't freak out
  }

  startRequested = false;
  Serial.println("Full range done");
  return 0;
}


// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(200);
  print_wakeup_reason();

  bootCount++;
  Serial.printf("Boot number: %d\n", bootCount);
  pinsetup();
  esp_sleep_enable_ext1_wakeup((1ULL << SHAKE_PIN), ESP_EXT1_WAKEUP_ANY_HIGH);

  pinMode(25, OUTPUT);
  digitalWrite(25, HIGH);
  delay(50);
  tft.init();
  tft.setRotation(0);

  Wire.begin(SDA_PIN, SCL_PIN);
  touch.begin();

  // Don't initialize MAX30102 at setup, wait until sampling screen is entered
  // I2C_2.begin(7, 8);
  // if (!particleSensor.begin(I2C_2, I2C_SPEED_STANDARD)) {
  //   Serial.println("MAX30102 not found. Check connections!");
  //   while (1);
  // }


  // // Configure sensor settings to default
  // particleSensor.setup();
  // particleSensor.setPulseAmplitudeRed(0x0F);
  // particleSensor.setPulseAmplitudeIR(0x0F);

  // zero buffers
  // memset(ppg_red_buffer, 0, sizeof(ppg_red_buffer));
  // memset(ppg_ir_buffer, 0, sizeof(ppg_ir_buffer));

  // Optional: customize parameters
  // particleSensor.setup(60, 4, 2, 100, 400, 4096); 
  // (sampleRate, ledBrightness, pulseWidth, adcRange, sampleAverage)

  if (!LittleFS.begin()) {
    Serial.println("An error has occurred while mounting LittleFS");
    delay(1000);
    return;
  }
  Serial.println("LittleFS mounted successfully.");

  if (bootCount == 1) {
    rtc.setTime(0, 8, 1, 1, 12, 2025);  // Second, Minute, Hour, Day, Month, Year
  }

  loadProfileData();  // Test reading saved data

  displayHome();
  lastTouch = millis();
}

// ================= LOOP =================
void loop() {
  String gest = "";
  if (touch.available()) {
    lastTouch = millis();
    gest = touch.gesture();
    uint16_t x = touch.data.x, y = touch.data.y;
    int mappedX = map(x, 0, 240, 0, tft.width());
    int mappedY = map(y, 0, 240, 0, tft.height());

    if (currentScreen == HOME_SCREEN)
      btnUnlock.press(btnUnlock.contains(mappedX, mappedY));
    else if (currentScreen == MENU_SCREEN) {
      btnBack.press(btnBack.contains(mappedX, mappedY));
      // btnProfile.press(btnProfile.contains(mappedX, mappedY));
      // btnSample.press(btnSample.contains(mappedX, mappedY));
      // btnSettings.press(btnSettings.contains(mappedX, mappedY));
      btnMenuCard.press(btnMenuCard.contains(mappedX, mappedY));
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
    else if (currentScreen == SETTINGS_SCREEN) {
      btnPrevField.press(btnPrevField.contains(mappedX, mappedY));
      btnNextField.press(btnNextField.contains(mappedX, mappedY));
      btnMinus.press(btnMinus.contains(mappedX, mappedY));
      btnPlus.press(btnPlus.contains(mappedX, mappedY));
      btnBack.press(btnBack.contains(mappedX, mappedY));
      btnConfirm.press(btnConfirm.contains(mappedX, mappedY));
    }
  } 
  else {
    btnUnlock.press(false);
    btnBack.press(false);
    // btnProfile.press(false);
    // btnSample.press(false);
    // btnSettings.press(false);
    btnConfirm.press(false);
    btnPrevField.press(false);
    btnNextField.press(false);
    btnMinus.press(false);
    btnPlus.press(false);
    btnMenuCard.press(false);
  }

  // === ACTIONS BY SCREEN ===
  if (currentScreen == HOME_SCREEN) {
    if (btnUnlock.justReleased()) drawMenuCarousel();
  }
    else if (currentScreen == MENU_SCREEN) {

      if (btnBack.justReleased()) {
        displayHome();
        return;
      }

      unsigned long now = millis();

      // ---- Handle swipes FIRST ----
      if (gest == "SWIPE LEFT" || gest == "SWIPE RIGHT") {
        // lockout so swipe doesn't also count as a tap
        if (now - lastGestureMs > GESTURE_LOCKOUT_MS) {
          if (gest == "SWIPE LEFT") {
            menuIndex = (menuIndex + 1) % MENU_COUNT;
            Serial.println("Swipe Left -> next menu item");
            drawMenuCarousel();
          } else {
            menuIndex = (menuIndex - 1 + MENU_COUNT) % MENU_COUNT;
            Serial.println("Swipe Right -> previous menu item");
            drawMenuCarousel();
          }
          lastGestureMs = now;
        }
        return; // IMPORTANT: don't allow "tap to open" on same touch
      }

      // ---- Only allow tap if we did NOT recently swipe ----
      if (btnMenuCard.justReleased()) {
        if (now - lastGestureMs > GESTURE_LOCKOUT_MS) {
          openSelectedMenuItem();
        }
        return;
      }
    }
  else if (currentScreen == PROFILE_SCREEN) {

    if (btnBack.justReleased()) {
      loadProfileData();
      drawMenuCarousel();
    }

    if (btnConfirm.justReleased()) {
      writeProfileData();
      drawMenuCarousel(); 
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
      disableMAX30102();
      drawMenuCarousel();
    }
  }

  else if (currentScreen == SETTINGS_SCREEN) {

    if (btnBack.justReleased()) {
      settings_loaded = false;
      drawMenuCarousel();
    }

    if (btnConfirm.justReleased()) {
      int hour  = settings_values[0];
      int minute= settings_values[1];
      int month = settings_values[2];
      int day   = settings_values[3];
      int year  = settings_values[4];

      // seconds, minutes, hours, day, month, year
      rtc.setTime(0, minute, hour, day, month, year);

      settings_loaded = false;
      drawMenuCarousel(); 
    }

    if (btnPrevField.justReleased()) {
      settings_fieldIndex = (settings_fieldIndex - 1 + SETTINGS_NUM_FIELDS) % SETTINGS_NUM_FIELDS;
      displaySettingsMenu();
    }

    if (btnNextField.justReleased()) {
      settings_fieldIndex = (settings_fieldIndex + 1) % SETTINGS_NUM_FIELDS;
      displaySettingsMenu();
    }

    if (btnMinus.justReleased()) {
      switch (settings_fieldIndex) {
        case 0: settings_values[0] = (settings_values[0] - 1 + 24) % 24; break; // hour 0-23
        case 1: settings_values[1] = (settings_values[1] - 1 + 60) % 60; break; // minute 0-59
        case 2: settings_values[2] = (settings_values[2] - 1 < 1) ? 12 : settings_values[2] - 1; break; // month 1-12
        case 3: settings_values[3] = (settings_values[3] - 1 < 1) ? 31 : settings_values[3] - 1; break; // day
        case 4: settings_values[4] = settings_values[4] - 1; break; // year
      }

      displaySettingsMenu();
    }

    if (btnPlus.justReleased()) {
      switch (settings_fieldIndex) {
        case 0: settings_values[0] = (settings_values[0] + 1) % 24; break; // hour 0-23
        case 1: settings_values[1] = (settings_values[1] + 1) % 60; break; // minute 0-59
        case 2: settings_values[2] = (settings_values[2] + 1 > 12) ? 1 : settings_values[2] + 1; break; // month
        case 3: settings_values[3] = (settings_values[3] + 1 > 31) ? 1 : settings_values[3] + 1; break; // day
        case 4: settings_values[4] = settings_values[4] + 1; break; // year
      }
    displaySettingsMenu();
  }
}

  sleep_param();  // checks for light sleep via timer
  delay(15);

}
