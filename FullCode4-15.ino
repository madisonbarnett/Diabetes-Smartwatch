// Screen
#include <Wire.h>
#include <TFT_eSPI.h>
#include <TFT_eWidget.h>
#include <CST816S.h>

// Operating System
#include <LittleFS.h>
using namespace fs;
#include <ESP32Time.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// Sleep
#include <esp_sleep.h>
#include <driver/rtc_io.h>

// MAX30102 Sensor
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include "Arduino.h"
#include "arduinoFFT.h"
#include "math.h"

// TFLite
#include <TensorFlowLite_ESP32.h>
#include "mlp.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ================= PIN DEFINITIONS =================
#define SDA_PIN 21
#define SCL_PIN 22
#define INT_PIN 38
#define RST_PIN 32
#define ADCPIN A7
#define SHAKE_PIN 37

// ================= GLOBAL OBJECTS =================
TFT_eSPI tft = TFT_eSPI();
CST816S touch(SDA_PIN, SCL_PIN, RST_PIN, INT_PIN);
TFT_eSPI_Button btnUnlock, btnBack, btnConfirm;
TFT_eSPI_Button btnPrevField, btnNextField, btnMinus, btnPlus;

ESP32Time rtc(0);

const char* menuItems[] = {"PROFILE", "SAMPLE", "SETTINGS", "HISTORY"};
uint16_t menuColors[] = {TFT_PINK, TFT_SKYBLUE, TFT_VIOLET, TFT_ORANGE};
const int MENU_COUNT = 4;
int menuIndex = 0;

// History Menu Globals
int historyCount = 0;
float historyAvg = 0;
float historyMin = 0;
float historyMax = 0;
int historyDays = 0;
int historyHours = 0;
int historyMinutes = 0;

float graphGlucose[100];
unsigned long graphEpochs[100];
int graphCount;
int historyView = 0;   // 0 = summary, 1 = graph, 2 = clear button

unsigned long lastGestureMs = 0;
const unsigned long GESTURE_LOCKOUT_MS = 250;

bool ignoreMenuBackOnce = false;

TFT_eSPI_Button btnMenuCard;
TwoWire I2C_2 = TwoWire(1);

// ================= SCREEN STATE =================
enum ScreenState {
  HOME_SCREEN,
  MENU_SCREEN,
  PROFILE_SCREEN,
  SAMPLE_SCREEN,
  SETTINGS_SCREEN,
  HISTORY_SCREEN
};
ScreenState currentScreen = HOME_SCREEN;

MAX30105 particleSensor;

const int SAMPLE_RATE_HZ = 100;
const int MAX_SAMPLES = 1500;
const unsigned long CAPTURE_DURATION_MS = 15000;
const int MIN_PEAK_DISTANCE = (int)(SAMPLE_RATE_HZ * 0.4f);

unsigned long initial = 0;
bool startRequested = false;

// Arrays
float irarray[MAX_SAMPLES];
int arraysize[MAX_SAMPLES];
unsigned long timearray[MAX_SAMPLES];
float P2PTime_Array[MAX_SAMPLES];

static float norm_ir[MAX_SAMPLES];
static float sort_buf[MAX_SAMPLES];

// PSD buffer for FFT spectral entropy (max bins = MAX_SAMPLES/2+1)
static float psd_buf[MAX_SAMPLES / 2 + 1];

int i = 0;

// Feature globals
float FIRST_DERIVMAX = 0;
float mean = 0;
float stddev = 0;
float skewness = 0;
float iqr = 0;
float teager_max = 0;
float meanP2Pinterval = 0;
float stddevTime = 0;
float spectral_entropy = 0;
float entropy = 0;   // alias written at end of FFT_SPECTRAL_ENTROPY
float ppg_freq = 0;

// Heart rate globals
int p = 0;
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute;
int beatAvg;

// ================= AI GLOBALS =================
constexpr int kTensorArenaSize = 55 * 1024;
uint8_t* tensor_arena = nullptr;

namespace {
  const tflite::Model* tfl_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* tfl_input = nullptr;
  TfLiteTensor* tfl_output = nullptr;
  bool model_ready = false;
}

float ai_result = -1.0f;

// ================= PROFILE VARIABLES =================
String fields[] = {"Age", "Sex", "Height", "Weight", "Diagnosis"};
int values[] = {21, 0, 65, 130, 1};
int fieldIndex = 0;
const int NUM_FIELDS = 5;

// ================= SETTINGS VARIABLES =================
String settings_fields[] = {"Hour", "Minute", "Month", "Day", "Year"};
int settings_values[] = {0, 0, 1, 1, 2025};
int settings_fieldIndex = 0;
const int SETTINGS_NUM_FIELDS = 5;
bool settings_loaded = false;

// ================= WAKEUP VARIABLES =================
RTC_DATA_ATTR int bootCount = 0;
unsigned long lastTouch = 0;
const unsigned long SLEEP_TIMEOUT = 45000;

// ================= FUNCTION PROTOTYPES =================
void displayHome();
void drawMenuCarousel();
void displayProfileMenu();
void displaySampleMenu();
void displaySettingsMenu();
void displayHistoryMenu();
void displayHistoryGraph();
void loadHistoryGraph72h();
void openSelectedMenuItem();
String getDisplayValue(int idx);
bool runSampling();
bool sampleBackPressed();
void drawSamplingUI();
void updateSamplingProgress();
void displayHistoryClearMenu();
void clearHistoryData();

// ================= FILE FUNCTIONS =================

void writeProfileData() {
  File file = LittleFS.open("/profile.txt", "w");
  if (!file) { Serial.println("Failed to open profile file for writing"); return; }
  String line = "";
  for (int i = 0; i < NUM_FIELDS; i++) {
    line += String(values[i]);
    if (i < NUM_FIELDS - 1) line += ",";
  }
  file.println(line);
  file.close();
}

void loadProfileData() {
  File file = LittleFS.open("/profile.txt", "r");
  if (!file) { Serial.println("No saved profile data found"); return; }
  while (file.available()) {
    String line = file.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;
    int start = 0, idx = 0;
    while (idx < NUM_FIELDS && start < (int)line.length()) {
      int comma = line.indexOf(',', start);
      if (comma == -1) comma = line.length();
      values[idx++] = line.substring(start, comma).toInt();
      start = comma + 1;
    }
    file.close();
  }
}

void writeHistoryData(float predictedGlucose) {
  File file = LittleFS.open("/history.txt", "a");
  if (!file) { Serial.println("Failed to open history file for appending"); return; }
  
  String timestamp = rtc.getTime("%Y-%m-%d %H:%M:%S");

  String line = "";
  line += timestamp;
  line += ",";
  line += String(predictedGlucose, 2);

  line += "," + String(values[0]);
  line += "," + String(values[3] * 0.453592f);
  line += "," + String(values[2] * 2.54f);
  line += "," + String(values[4]);
  line += "," + String(meanP2Pinterval, 6);
  line += "," + String(stddev, 6);
  line += "," + String(teager_max, 6);
  line += "," + String(skewness, 6);
  line += "," + String(iqr, 6);
  line += "," + String(entropy, 6);
  line += "," + String(FIRST_DERIVMAX, 6);
  line += "," + String(stddevTime, 6);

  file.println(line);
  file.close();

  Serial.println("History entry saved:");
  Serial.println(line);
}

void summarizeHistoryData() {
  File file = LittleFS.open("/history.txt", "r");
  if (!file) {
    Serial.println("No saved history data found");
    historyCount = 0;
    return;
  }

  historyCount = 0;
  float sum = 0.0;
  unsigned long oldestEpoch = 0;
  unsigned long newestEpoch = 0;

  while (file.available()) {
    String line = file.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;

    int firstComma = line.indexOf(',');
    if (firstComma == -1) continue;

    String timestamp = line.substring(0, firstComma);

    int secondComma = line.indexOf(',', firstComma + 1);
    String glucoseStr = (secondComma == -1)
        ? line.substring(firstComma + 1)
        : line.substring(firstComma + 1, secondComma);

    float glucose = glucoseStr.toFloat();

    int year   = timestamp.substring(0, 4).toInt();
    int month  = timestamp.substring(5, 7).toInt();
    int day    = timestamp.substring(8, 10).toInt();
    int hour   = timestamp.substring(11, 13).toInt();
    int minute = timestamp.substring(14, 16).toInt();
    int second = timestamp.substring(17, 19).toInt();

    struct tm t;
    t.tm_year = year - 1900;
    t.tm_mon  = month - 1;
    t.tm_mday = day;
    t.tm_hour = hour;
    t.tm_min  = minute;
    t.tm_sec  = second;
    t.tm_isdst = -1;

    unsigned long epoch = mktime(&t);

    if (historyCount == 0) {
      historyMin = glucose;
      historyMax = glucose;
      oldestEpoch = epoch;
      newestEpoch = epoch;
    } else {
      if (glucose < historyMin) historyMin = glucose;
      if (glucose > historyMax) historyMax = glucose;
      if (epoch < oldestEpoch) oldestEpoch = epoch;
      if (epoch > newestEpoch) newestEpoch = epoch;
    }

    sum += glucose;
    historyCount++;
  }

  file.close();

  if (historyCount == 0) return;

  historyAvg = sum / historyCount;

  unsigned long span = newestEpoch - oldestEpoch;
  historyDays = span / 86400;
  historyHours = (span % 86400) / 3600;
  historyMinutes = (span % 3600) / 60;
}

void loadHistoryGraph72h() {
  File file = LittleFS.open("/history.txt", "r");
  if (!file) {
    graphCount = 0;
    return;
  }

  graphCount = 0;

  // Get current RTC time as the same format used in history file
  String nowStr = rtc.getTime("%Y-%m-%d %H:%M:%S");

  int year   = nowStr.substring(0, 4).toInt();
  int month  = nowStr.substring(5, 7).toInt();
  int day    = nowStr.substring(8, 10).toInt();
  int hour   = nowStr.substring(11, 13).toInt();
  int minute = nowStr.substring(14, 16).toInt();
  int second = nowStr.substring(17, 19).toInt();

  struct tm nowTm = {};
  nowTm.tm_year = year - 1900;
  nowTm.tm_mon  = month - 1;
  nowTm.tm_mday = day;
  nowTm.tm_hour = hour;
  nowTm.tm_min  = minute;
  nowTm.tm_sec  = second;
  nowTm.tm_isdst = -1;

  unsigned long nowEpoch = mktime(&nowTm);
  unsigned long cutoffEpoch = nowEpoch - (72UL * 3600UL);

  while (file.available()) {
    String line = file.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;

    int firstComma = line.indexOf(',');
    if (firstComma == -1) continue;

    String timestamp = line.substring(0, firstComma);

    int secondComma = line.indexOf(',', firstComma + 1);
    String glucoseStr = (secondComma == -1)
        ? line.substring(firstComma + 1)
        : line.substring(firstComma + 1, secondComma);

    float glucose = glucoseStr.toFloat();

    int year   = timestamp.substring(0, 4).toInt();
    int month  = timestamp.substring(5, 7).toInt();
    int day    = timestamp.substring(8, 10).toInt();
    int hour   = timestamp.substring(11, 13).toInt();
    int minute = timestamp.substring(14, 16).toInt();
    int second = timestamp.substring(17, 19).toInt();

    struct tm t = {};
    t.tm_year = year - 1900;
    t.tm_mon  = month - 1;
    t.tm_mday = day;
    t.tm_hour = hour;
    t.tm_min  = minute;
    t.tm_sec  = second;
    t.tm_isdst = -1;

    unsigned long sampleEpoch = mktime(&t);

    if (sampleEpoch >= cutoffEpoch && sampleEpoch <= nowEpoch) {
      if (graphCount < 100) {
        graphEpochs[graphCount] = sampleEpoch;
        graphGlucose[graphCount] = glucose;
        graphCount++;
      }
    }
  }

  file.close();
}

void displayHistoryGraph() {
  currentScreen = HISTORY_SCREEN;
  historyView = 1;
  tft.fillScreen(TFT_BLACK);

  loadHistoryGraph72h();

  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextDatum(MC_DATUM);
  tft.setTextSize(1);
  tft.drawString("Last 72 Hours", tft.width()/2, 30);

  if (graphCount < 2) {
    tft.setTextDatum(MC_DATUM);
    tft.drawString("Not enough data", tft.width()/2, 120);
    return;
  }

  const int plotW = 140;
  const int plotH = 110;
  const int plotX = ((240 - plotW) / 2) + 5;
  const int plotY = 55;

  float minVal = graphGlucose[0];
  float maxVal = graphGlucose[0];

  for (int i = 1; i < graphCount; i++) {
    if (graphGlucose[i] < minVal) minVal = graphGlucose[i];
    if (graphGlucose[i] > maxVal) maxVal = graphGlucose[i];
  }

  if (maxVal <= minVal) maxVal = minVal + 1.0f;

  unsigned long newestEpoch = graphEpochs[graphCount - 1];
  unsigned long oldestEpoch = newestEpoch - (72UL * 3600UL);

  tft.drawRect(plotX, plotY, plotW, plotH, TFT_WHITE);

  for (int i = 0; i < graphCount - 1; i++) {
    int x1 = plotX + (int)(((graphEpochs[i] - oldestEpoch) * (plotW - 1)) / (72UL * 3600UL));
    int x2 = plotX + (int)(((graphEpochs[i + 1] - oldestEpoch) * (plotW - 1)) / (72UL * 3600UL));

    int y1 = plotY + plotH - 1 -
             (int)(((graphGlucose[i] - minVal) / (maxVal - minVal)) * (plotH - 1));
    int y2 = plotY + plotH - 1 -
             (int)(((graphGlucose[i + 1] - minVal) / (maxVal - minVal)) * (plotH - 1));

    tft.drawLine(x1, y1, x2, y2, TFT_GREEN);
    tft.fillCircle(x1, y1, 2, TFT_GREEN);
  }

  int lastX = plotX + (int)(((graphEpochs[graphCount - 1] - oldestEpoch) * (plotW - 1)) / (72UL * 3600UL));
  int lastY = plotY + plotH - 1 -
              (int)(((graphGlucose[graphCount - 1] - minVal) / (maxVal - minVal)) * (plotH - 1));
  tft.fillCircle(lastX, lastY, 2, TFT_GREEN);

  tft.setTextSize(1);
  tft.setTextDatum(MR_DATUM);
  tft.drawString(String(maxVal, 1), plotX - 4, plotY + 4);
  tft.drawString(String(minVal, 1), plotX - 4, plotY + plotH - 4);

  tft.setTextDatum(TL_DATUM);
  tft.drawString("-72h", plotX, plotY + plotH + 10);

  tft.setTextDatum(TR_DATUM);
  tft.drawString("Now", plotX + plotW, plotY + plotH + 10);
}

void clearHistoryData() {
  if (LittleFS.exists("/history.txt")) {
    LittleFS.remove("/history.txt");
    Serial.println("History file cleared");
  } else {
    Serial.println("No history file to clear");
  }

  historyCount = 0;
  historyAvg = 0;
  historyMin = 0;
  historyMax = 0;
  historyDays = 0;
  historyHours = 0;
  historyMinutes = 0;
  graphCount = 0;
}

void displayHistoryClearMenu() {
  currentScreen = HISTORY_SCREEN;
  historyView = 2;
  tft.fillScreen(TFT_BLACK);

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);

  tft.setTextSize(2);
  tft.drawString("Clear History?", tft.width()/2, 95);

  tft.setTextSize(1);
  tft.drawString("This will erase all", tft.width()/2, 125);
  tft.drawString("saved glucose data", tft.width()/2, 140);

  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30,
                     TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);

  btnConfirm.initButton(&tft, tft.width()/2, 190, 90, 40,
                        TFT_WHITE, TFT_RED, TFT_WHITE, "Confirm", 2);
  btnConfirm.drawButton();
}

// ================= SLEEP =================
void pinsetup() {
  pinMode((int)SHAKE_PIN, INPUT_PULLDOWN);
}

int print_wakeup_reason() {
  esp_sleep_wakeup_cause_t wakeup_reason;
  int reason = 0;
  wakeup_reason = esp_sleep_get_wakeup_cause();

  switch(wakeup_reason) {
    case ESP_SLEEP_WAKEUP_EXT0 : Serial.println("Wakeup caused by external signal using RTC_IO"); reason = 2; break;
    case ESP_SLEEP_WAKEUP_EXT1 : Serial.println("Wakeup caused by external signal using RTC_CNTL"); reason = 3; break;
    case ESP_SLEEP_WAKEUP_TIMER : Serial.println("Wakeup caused by timer"); reason = 4; break;
    case ESP_SLEEP_WAKEUP_TOUCHPAD : Serial.println("Wakeup caused by touchpad"); reason = 5; break;
    case ESP_SLEEP_WAKEUP_ULP : Serial.println("Wakeup caused by ULP program"); reason = 6; break;
    default : Serial.printf("Wakeup was not caused by deep sleep: %d\n", wakeup_reason); reason = -1; break;
  }
  return reason;
}

void sleep_param() {
  if (millis() - lastTouch < SLEEP_TIMEOUT) return;
  Serial.println("Going to sleep...");
  digitalWrite(25, LOW);
  delay(100);
  esp_deep_sleep_start();
}

// ================= SENSOR =================

bool max30102_active = false;

void enableMAX30102() {
  if (max30102_active) return;
  I2C_2.begin(7, 8);
  if (!particleSensor.begin(I2C_2, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found!");
    return;
  }
  particleSensor.setup(0x1F, 1, 2, 100, 411, 4096);
  max30102_active = true;
}

void disableMAX30102() {
  if (!max30102_active) return;
  particleSensor.shutDown();
  max30102_active = false;
}

// ================= AI =================

void setupAI() {
  tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  if (tensor_arena == nullptr) { Serial.println("Failed to allocate tensor arena"); return; }

  tflite::InitializeTarget();

  tfl_model = tflite::GetModel(model_weights_mlp_15s_vitaldb_int8_tflite);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  tfl_input  = interpreter->input(0);
  tfl_output = interpreter->output(0);
  model_ready = true;
  Serial.println("AI model ready");
}

float runInference() {
  if (!model_ready) return -1.0f;

  float in_scale      = tfl_input->params.scale;
  int   in_zero_point = tfl_input->params.zero_point;

  tfl_input->data.int8[0]  = (int8_t)(((float)values[0]               / in_scale) + in_zero_point);  // Age
  tfl_input->data.int8[1]  = (int8_t)(((float)values[3] * 0.45359237f / in_scale) + in_zero_point);  // Weight kg
  tfl_input->data.int8[2]  = (int8_t)(((float)values[2] * 2.54f       / in_scale) + in_zero_point);  // Height cm
  tfl_input->data.int8[3]  = (int8_t)(((float)values[4]               / in_scale) + in_zero_point);  // PreOpDm
  tfl_input->data.int8[4]  = (int8_t)((meanP2Pinterval                / in_scale) + in_zero_point);  // PPGMEANINTERVAL
  tfl_input->data.int8[5]  = (int8_t)((stddev                         / in_scale) + in_zero_point);  // PPGSTD
  tfl_input->data.int8[6]  = (int8_t)((teager_max                     / in_scale) + in_zero_point);  // PPGTEAGER
  tfl_input->data.int8[7]  = (int8_t)((skewness                       / in_scale) + in_zero_point);  // PPGSKEW
  tfl_input->data.int8[8]  = (int8_t)((iqr                            / in_scale) + in_zero_point);  // PPGIQR
  tfl_input->data.int8[9]  = (int8_t)((entropy                        / in_scale) + in_zero_point);  // PPGENTROPY
  tfl_input->data.int8[10] = (int8_t)((FIRST_DERIVMAX                 / in_scale) + in_zero_point);  // FIRST_DERIV_MAX
  tfl_input->data.int8[11] = (int8_t)((stddevTime                     / in_scale) + in_zero_point);  // PPGSTDPP

  if (interpreter->Invoke() != kTfLiteOk) { Serial.println("Invoke() failed"); return -1.0f; }

  int8_t raw = tfl_output->data.int8[0];
  float result = (raw - tfl_output->params.zero_point) * tfl_output->params.scale;
  Serial.print("AI result: ");
  Serial.println(result);
  return result;
}

// ================= FEATURE FUNCTIONS =================

float percentile_interp(float *arr, int n, float pct) {
  float idx = pct * (float)(n - 1);
  int lo = (int)idx;
  int hi = lo + 1;
  if (hi >= n) return arr[n - 1];
  float frac = idx - (float)lo;
  return arr[lo] + frac * (arr[hi] - arr[lo]);
}

void sort_copy(float *src, float *dst, int n) {
  for (int j = 0; j < n; j++) dst[j] = src[j];
  for (int a = 0; a < n - 1; a++) {
    for (int b = a + 1; b < n; b++) {
      if (dst[b] < dst[a]) {
        float t = dst[a];
        dst[a] = dst[b];
        dst[b] = t;
      }
    }
  }
}

void resetFeatures() {
  i = 0;
  p = 0;

  FIRST_DERIVMAX = 0;
  mean = 0;
  stddev = 0;
  skewness = 0;
  iqr = 0;
  teager_max = 0;
  meanP2Pinterval = 0;
  stddevTime = 0;
  spectral_entropy = 0;
  entropy = 0;
  ppg_freq = 0;
}

void NORMALIZE_SIGNAL() {
  mean = 0;
  for (int j = 0; j < i; j++) mean += irarray[j];
  mean /= (float)i;

  stddev = 0;
  for (int j = 0; j < i; j++) {
    float d = irarray[j] - mean;
    stddev += d * d;
  }
  stddev = sqrt(stddev / (float)i);

  if (stddev == 0) {
    for (int j = 0; j < i; j++) norm_ir[j] = 0;
  } else {
    for (int j = 0; j < i; j++) norm_ir[j] = (irarray[j] - mean) / stddev;
  }
}

void FIRST_DERIV_MAX() {
  if (i < 2) {
    FIRST_DERIVMAX = 0;
    return;
  }

  FIRST_DERIVMAX = norm_ir[1] - norm_ir[0];
  for (int j = 1; j < i; j++) {
    float deriv = norm_ir[j] - norm_ir[j - 1];
    if (deriv > FIRST_DERIVMAX) FIRST_DERIVMAX = deriv;
  }
}

void SKEWNESS_CALC() {
  if (i < 1) {
    skewness = 0;
    return;
  }

  float norm_mean = 0;
  for (int j = 0; j < i; j++) norm_mean += norm_ir[j];
  norm_mean /= (float)i;

  float norm_std = 0;
  for (int j = 0; j < i; j++) {
    float d = norm_ir[j] - norm_mean;
    norm_std += d * d;
  }
  norm_std = sqrt(norm_std / (float)i);

  if (norm_std == 0) {
    skewness = 0;
    return;
  }

  skewness = 0;
  for (int j = 0; j < i; j++) {
    float z = (norm_ir[j] - norm_mean) / norm_std;
    skewness += z * z * z;
  }
  skewness /= (float)i;
}

void IQR_CALC() {
  if (i < 1) {
    iqr = 0;
    return;
  }

  sort_copy(norm_ir, sort_buf, i);
  float q1 = percentile_interp(sort_buf, i, 0.25f);
  float q3 = percentile_interp(sort_buf, i, 0.75f);
  iqr = q3 - q1;
}

void TEAGER_ENERGY() {
  if (i < 3) {
    teager_max = 0;
    return;
  }

  float sum = 0;
  for (int j = 1; j < i - 1; j++) {
    float t = norm_ir[j] * norm_ir[j] - norm_ir[j - 1] * norm_ir[j + 1];
    sum += fabsf(t);
  }

  teager_max = sum / (float)(i - 2);
}

void FFT_SPECTRAL_ENTROPY() {
  if (i < 2) {
    spectral_entropy = 0;
    entropy = 0;
    return;
  }

  int k_min = (int)(0.5f * (float)i / (float)SAMPLE_RATE_HZ);
  int k_max = (int)(4.0f * (float)i / (float)SAMPLE_RATE_HZ) + 1;
  if (k_max > i / 2 + 1) k_max = i / 2 + 1;

  int n_bins = k_max - k_min;

  for (int k = k_min; k < k_max; k++) {
    float real = 0.0f, imag = 0.0f;

    for (int t = 0; t < i; t++) {
      float ang = 2.0f * PI * (float)k * (float)t / (float)i;
      real += norm_ir[t] * cosf(ang);
      imag -= norm_ir[t] * sinf(ang);
    }

    psd_buf[k - k_min] = real * real + imag * imag;
  }

  float psd_sum = 0;
  for (int k = 0; k < n_bins; k++) psd_sum += psd_buf[k];

  spectral_entropy = 0;
  if (psd_sum > 0) {
    for (int k = 0; k < n_bins; k++) {
      float pwr = psd_buf[k] / psd_sum;
      if (pwr > 0) spectral_entropy -= pwr * logf(pwr);
    }
  }

  entropy = spectral_entropy;  // keep entropy in sync
}

void PEAK_INTERVAL_FEATURES() {
  meanP2Pinterval = 0;
  stddevTime = 0;
  ppg_freq = 0;
  p = 0;

  if (i < 3) return;

  static int peaks[MAX_SAMPLES];
  int peakCount = 0;

  for (int j = 1; j < i - 1; j++) {
    bool localPeak = (norm_ir[j] > norm_ir[j - 1]) && (norm_ir[j] >= norm_ir[j + 1]);
    bool heightOk = (norm_ir[j] > 0);

    if (!(localPeak && heightOk)) continue;

    if (peakCount == 0) {
      peaks[peakCount++] = j;
    } else {
      int prev = peaks[peakCount - 1];
      if ((j - prev) >= MIN_PEAK_DISTANCE) {
        peaks[peakCount++] = j;
      } else if (norm_ir[j] > norm_ir[prev]) {
        peaks[peakCount - 1] = j;
      }
    }
  }

  if (peakCount < 2) return;

  for (int j = 1; j < peakCount; j++) {
    int sampleDelta = peaks[j] - peaks[j - 1];
    P2PTime_Array[p++] = (float)sampleDelta / (float)SAMPLE_RATE_HZ;
  }

  if (p == 0) return;

  for (int j = 0; j < p; j++) meanP2Pinterval += P2PTime_Array[j];
  meanP2Pinterval /= (float)p;

  for (int j = 0; j < p; j++) {
    float d = P2PTime_Array[j] - meanP2Pinterval;
    stddevTime += d * d;
  }
  stddevTime = sqrt(stddevTime / (float)p);

  if (meanP2Pinterval > 0) ppg_freq = 1.0f / meanP2Pinterval;
}

void PROCESS_AND_PRINT() {
  NORMALIZE_SIGNAL();
  FIRST_DERIV_MAX();
  SKEWNESS_CALC();
  IQR_CALC();
  TEAGER_ENERGY();
  FFT_SPECTRAL_ENTROPY();
  PEAK_INTERVAL_FEATURES();
}

// ================= SAMPLING =================

bool sampleBackPressed() {
  if (!touch.available()) return false;

  lastTouch = millis();

  uint16_t x = touch.data.x;
  uint16_t y = touch.data.y;

  int mappedX = map(x, 0, 240, 0, tft.width());
  int mappedY = map(y, 0, 240, 0, tft.height());

  return btnBack.contains(mappedX, mappedY);
}

void drawSamplingUI() {
  tft.fillScreen(TFT_BLACK);

  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30,
                     TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(2);
  tft.drawString("Measuring...", tft.width()/2, 80);

  int barX = 25, barY = 125, barW = 190, barH = 20;
  tft.drawRoundRect(barX, barY, barW, barH, 6, TFT_WHITE);

  tft.setTextSize(1);
  tft.drawString("0%", tft.width()/2, 165);
}

void updateSamplingProgress() {
  static int lastFillW = -1;
  static int lastPct = -1;

  int barX = 25, barY = 125, barW = 190, barH = 20;

  unsigned long elapsedMs = millis() - initial;
  if (elapsedMs > CAPTURE_DURATION_MS) elapsedMs = CAPTURE_DURATION_MS;

  int pct = (elapsedMs * 100UL) / CAPTURE_DURATION_MS;
  int fillW = ((barW - 4) * pct) / 100;

  if (fillW != lastFillW) {
    if (lastFillW > 0) tft.fillRect(barX + 2, barY + 2, lastFillW, barH - 4, TFT_BLACK);
    if (fillW > 0)     tft.fillRect(barX + 2, barY + 2, fillW,     barH - 4, TFT_GREEN);
    lastFillW = fillW;
  }

  if (pct != lastPct) {
    tft.fillRect(90, 158, 60, 14, TFT_BLACK);
    tft.setTextDatum(MC_DATUM);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setTextSize(1);
    tft.drawString(String(pct) + "%", tft.width()/2, 165);
    lastPct = pct;
  }
}

bool runSampling() {
  i = 0;
  p = 0;
  lastBeat = 0;
  beatAvg = 0;
  rateSpot = 0;

  FIRST_DERIVMAX = 0;
  mean = 0;
  stddev = 0;
  skewness = 0;
  iqr = 0;
  teager_max = 0;
  meanP2Pinterval = 0;
  stddevTime = 0;
  spectral_entropy = 0;
  entropy = 0;

  initial = millis();
  unsigned long lastProgressUpdate = 0;

  drawSamplingUI();
  updateSamplingProgress();

  while ((millis() - initial) < CAPTURE_DURATION_MS && i < 1500) {
    if (sampleBackPressed()) {
      Serial.println("Sampling canceled by user");
      return false;
    }

    particleSensor.check();

    while (particleSensor.available() && i < 1500) {
      uint32_t ir = particleSensor.getIR();
      unsigned long nowMs = millis();

      irarray[i] = ir;
      timearray[i] = nowMs;

      if (checkForBeat(ir)) {
        long delta = nowMs - lastBeat;
        lastBeat = nowMs;

        if (delta > 0) {
          if (p < 1500) {
            P2PTime_Array[p] = delta;
            p++;
          }

          beatsPerMinute = 60.0f / (delta / 1000.0f);

          if (beatsPerMinute < 255 && beatsPerMinute > 20) {
            rates[rateSpot++] = (byte)beatsPerMinute;
            rateSpot %= RATE_SIZE;

            beatAvg = 0;
            for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
            beatAvg /= RATE_SIZE;
          }
        }
      }

      arraysize[i] = i;
      i++;
      particleSensor.nextSample();

      if (nowMs - lastProgressUpdate >= 100) {
        updateSamplingProgress();
        lastProgressUpdate = nowMs;
      }

      if (sampleBackPressed()) {
        Serial.println("Sampling canceled by user");
        return false;
      }
    }

    if (millis() - lastProgressUpdate >= 100) {
      updateSamplingProgress();
      lastProgressUpdate = millis();
    }

    delay(1);
  }

  updateSamplingProgress();
  return true;
}

// ================= DISPLAY FUNCTIONS =================

void drawBackArrow(int cx, int cy) {
  int startX = cx + 10, endX = cx - 10;
  tft.drawLine(startX, cy, endX, cy, TFT_WHITE);
  tft.drawLine(endX, cy, endX + 10, cy - 10, TFT_WHITE);
  tft.drawLine(endX, cy, endX + 10, cy + 10, TFT_WHITE);
}

void drawMenuIcon(int idx, int cx, int cy, int size, uint16_t color) {
  if (idx == 0) {
    tft.fillCircle(cx, cy - size / 5, size / 4, color);
    tft.fillRoundRect(cx - size / 3, cy, 2 * size / 3, size / 3, 6, color);
  }
  else if (idx == 1) {
    tft.fillCircle(cx, cy, size / 3, color);
    tft.fillTriangle(cx, cy - size / 2, cx - size / 3, cy, cx + size / 3, cy, color);
  }
  else if (idx == 2) {
    int rOuter = size / 3, rInner = rOuter - 6, toothL = 8, toothW = 6;
    tft.fillCircle(cx, cy, rOuter, color);
    tft.fillCircle(cx, cy, rInner, TFT_VIOLET);
    tft.fillRoundRect(cx - toothW/2, cy - rOuter - toothL + 2, toothW, toothL, 2, color);
    tft.fillRoundRect(cx - toothW/2, cy + rOuter - 2,           toothW, toothL, 2, color);
    tft.fillRoundRect(cx - rOuter - toothL + 2, cy - toothW/2,  toothL, toothW, 2, color);
    tft.fillRoundRect(cx + rOuter - 2,           cy - toothW/2,  toothL, toothW, 2, color);
    int knob = 8, rDiag = (int)(rOuter * 0.7071f) - 1;
    tft.fillRoundRect(cx + rDiag - knob/2, cy - rDiag - knob/2, knob, knob, 2, color);
    tft.fillRoundRect(cx + rDiag - knob/2, cy + rDiag - knob/2, knob, knob, 2, color);
    tft.fillRoundRect(cx - rDiag - knob/2, cy + rDiag - knob/2, knob, knob, 2, color);
    tft.fillRoundRect(cx - rDiag - knob/2, cy - rDiag - knob/2, knob, knob, 2, color);
    tft.drawCircle(cx, cy, rInner - 2, color);
    tft.drawCircle(cx, cy, (rInner - 2) / 2, color);
  }
  else if (idx == 3) {
    int rOuter = size / 3;
    int rInner = rOuter - 6;
    tft.fillCircle(cx, cy, rOuter, color);
    tft.fillCircle(cx, cy, rInner, TFT_ORANGE);
    tft.drawLine(cx, cy, cx, cy - rOuter / 2, color);
    tft.drawLine(cx, cy, cx + rOuter / 2 - 1, cy, color);
    tft.fillCircle(cx, cy, 2, color);
  }
}

void displayHome() {
  currentScreen = HOME_SCREEN;
  tft.fillScreen(TFT_BLACK);
  btnUnlock.initButton(&tft, tft.width()/2, tft.height()/2,
      tft.width(), tft.height(), TFT_BLACK, TFT_BLACK, TFT_BLACK, "", 1);
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

void drawMenuDots(int activeIdx) {
  int y = 205, spacing = 14;
  int startX = (tft.width()/2) - ((MENU_COUNT-1)*spacing/2);
  for (int i = 0; i < MENU_COUNT; i++) {
    int x = startX + i * spacing;
    if (i == activeIdx) tft.fillCircle(x, y, 4, TFT_WHITE);
    else tft.drawCircle(x, y, 4, TFT_WHITE);
  }
}

void drawMenuCarousel() {
  currentScreen = MENU_SCREEN;
  tft.fillScreen(TFT_BLACK);
  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30, TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);
  int cardW = 110, cardH = 110, cardX = tft.width()/2, cardY = 120;
  uint16_t fill = menuColors[menuIndex];
  btnMenuCard.initButton(&tft, cardX, cardY, cardW-20, cardH-20, TFT_BLACK, TFT_BLACK, TFT_BLACK, "", 1);
  tft.fillRoundRect(cardX-cardW/2, cardY-cardH/2, cardW, cardH, 16, fill);
  tft.drawRoundRect(cardX-cardW/2, cardY-cardH/2, cardW, cardH, 16, TFT_WHITE);
  drawMenuIcon(menuIndex, cardX, cardY-5, 80, TFT_WHITE);
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, fill);
  tft.setTextSize(2);
  tft.drawString(menuItems[menuIndex], cardX, cardY+45);
  drawMenuDots(menuIndex);
}

void openSelectedMenuItem() {
  if (menuIndex == 0) displayProfileMenu();
  else if (menuIndex == 1) displaySampleMenu();
  else if (menuIndex == 2) displaySettingsMenu();
  else if (menuIndex == 3) displayHistoryMenu();
}

void displayProfileMenu() {
  currentScreen = PROFILE_SCREEN;
  tft.fillScreen(TFT_BLACK);
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(2);
  tft.drawString(fields[fieldIndex], tft.width()/2, 70);
  tft.setTextSize(3);
  tft.drawString(getDisplayValue(fieldIndex), tft.width()/2, 130);
  btnPrevField.initButton(&tft, 45, 70, 40, 40, TFT_WHITE, TFT_BLACK, TFT_WHITE, "<", 2);
  btnNextField.initButton(&tft, 195, 70, 40, 40, TFT_WHITE, TFT_BLACK, TFT_WHITE, ">", 2);
  btnPrevField.drawButton(); btnNextField.drawButton();
  btnMinus.initButton(&tft, 45, 130, 50, 45, TFT_WHITE, TFT_BLACK, TFT_WHITE, "-", 3);
  btnPlus.initButton(&tft, 195, 130, 50, 45, TFT_WHITE, TFT_BLACK, TFT_WHITE, "+", 3);
  btnMinus.drawButton(); btnPlus.drawButton();
  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30, TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);
  btnConfirm.initButton(&tft, tft.width()/2, 190, 70, 40, TFT_WHITE, TFT_PINK, TFT_WHITE, "Save", 2);
  btnConfirm.drawButton();
}

String getDisplayValue(int idx) {
  if (idx == 1) return (values[1] == 0) ? "F" : "M";
  if (idx == 4) {
    if (values[4] == 1) return "T1";
    if (values[4] == 2) return "T2";
    return "None";
  }
  return String(values[idx]);
}

void displaySampleMenu() {
  currentScreen = SAMPLE_SCREEN;
  enableMAX30102();

  tft.fillScreen(TFT_BLACK);
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(2);
  tft.drawString("Place finger", tft.width()/2, 100);
  tft.drawString("on sensor", tft.width()/2, 130);

  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30,
                     TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);

  int cx = tft.width() / 2;
  int cy = 160;
  int startX = cx - 15, endX = cx + 15;
  tft.drawLine(startX, cy, endX, cy, TFT_WHITE);
  tft.drawLine(endX, cy, endX - 12, cy - 12, TFT_WHITE);
  tft.drawLine(endX, cy, endX - 12, cy + 12, TFT_WHITE);

  unsigned long startWait = millis();
  while (millis() - startWait < 3000) {
    if (sampleBackPressed()) {
      disableMAX30102();
      drawMenuCarousel();
      ignoreMenuBackOnce = true;
      return;
    }
    delay(10);
  }

  bool completed = runSampling();

  if (!completed) {
    disableMAX30102();
    drawMenuCarousel();
    ignoreMenuBackOnce = true;
    return;
  }

  Serial.print("Samples collected: ");
  Serial.println(i);

  if (i < 10) {
    tft.fillScreen(TFT_BLACK);
    tft.setTextDatum(MC_DATUM);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setTextSize(2);
    tft.drawString("Not enough data", tft.width()/2, 110);
    btnBack.initButton(&tft, tft.width()/2, 30, 50, 30,
                       TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
    btnBack.drawButton();
    drawBackArrow(tft.width()/2, 30);
    return;
  }

  PROCESS_AND_PRINT();
  ai_result = runInference();

  tft.fillScreen(TFT_BLACK);
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(2);
  tft.drawString("Result:", tft.width()/2, 100);
  tft.setTextSize(3);

  if (ai_result < 0) {
    tft.drawString("Error", tft.width()/2, 140);
  } else {
    tft.drawString(String(ai_result, 4), tft.width()/2, 140);
    writeHistoryData(ai_result);
  }

  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30,
                     TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);
}

String getSettingsDisplayValue(int idx) {
  if (idx == 0) return (settings_values[0] < 10 ? "0" : "") + String(settings_values[0]);
  if (idx == 1) return (settings_values[1] < 10 ? "0" : "") + String(settings_values[1]);
  return String(settings_values[idx]);
}

void displaySettingsMenu() {
  currentScreen = SETTINGS_SCREEN;
  tft.fillScreen(TFT_BLACK);
  if (!settings_loaded) {
    settings_values[0] = rtc.getHour();
    settings_values[1] = rtc.getMinute();
    settings_values[2] = rtc.getMonth() + 1;
    settings_values[3] = rtc.getDay();
    settings_values[4] = rtc.getYear();
    settings_loaded = true;
  }
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(2);
  tft.drawString(settings_fields[settings_fieldIndex], tft.width()/2, 70);
  tft.setTextSize(3);
  tft.drawString(getSettingsDisplayValue(settings_fieldIndex), tft.width()/2, 130);
  btnPrevField.initButton(&tft, 45, 70, 40, 40, TFT_WHITE, TFT_BLACK, TFT_WHITE, "<", 2);
  btnNextField.initButton(&tft, 195, 70, 40, 40, TFT_WHITE, TFT_BLACK, TFT_WHITE, ">", 2);
  btnPrevField.drawButton(); btnNextField.drawButton();
  btnMinus.initButton(&tft, 45, 130, 50, 45, TFT_WHITE, TFT_BLACK, TFT_WHITE, "-", 3);
  btnPlus.initButton(&tft, 195, 130, 50, 45, TFT_WHITE, TFT_BLACK, TFT_WHITE, "+", 3);
  btnMinus.drawButton(); btnPlus.drawButton();
  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30, TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton(); drawBackArrow(tft.width()/2, 30);
  btnConfirm.initButton(&tft, tft.width()/2, 190, 70, 40, TFT_WHITE, TFT_VIOLET, TFT_WHITE, "Save", 2);
  btnConfirm.drawButton();
}

void displayHistoryMenu() {
  currentScreen = HISTORY_SCREEN;
  historyView = 0;
  tft.fillScreen(TFT_BLACK);
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(1);

  summarizeHistoryData();

  if (historyCount == 0) {
    tft.setTextSize(2);
    tft.drawString("No Data", tft.width()/2, 120);
  } else {
    tft.setTextSize(2);
    tft.drawString("Samples: " + String(historyCount), tft.width()/2, 90);
    tft.drawString("Avg: " + String(historyAvg, 1), tft.width()/2, 110);
    tft.drawString("Min: " + String(historyMin, 1), tft.width()/2, 130);
    tft.drawString("Max: " + String(historyMax, 1), tft.width()/2, 150);
    tft.drawString("Span: " + String(historyDays) + "d " + String(historyHours) + "h", tft.width()/2, 170);
  }

  btnBack.initButton(&tft, tft.width()/2, 30, 50, 30, TFT_BLACK, TFT_BLACK, TFT_WHITE, "", 1);
  btnBack.drawButton();
  drawBackArrow(tft.width()/2, 30);
}

// ================= SETUP =================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);
  
  delay(500);
  print_wakeup_reason();
  Serial.println("Step 1");

  bootCount++;
  Serial.println("Step 2");
  pinsetup();
  esp_sleep_enable_ext1_wakeup((1ULL << SHAKE_PIN), ESP_EXT1_WAKEUP_ANY_HIGH);

  pinMode(25, OUTPUT);
  digitalWrite(25, HIGH);
  Serial.println("Step 3");
  delay(50);
  tft.init();
  Serial.println("Step 4 - tft init done");
  tft.setRotation(0);
  Wire.begin(SDA_PIN, SCL_PIN);
  Serial.println("Step 5");

  touch.begin();
  Serial.println("Step 6");
  if (!LittleFS.begin(true)) {
    Serial.println("LittleFS mount failed");
    delay(1000);
    return;
  }

  if (bootCount == 1) rtc.setTime(0, 8, 1, 1, 12, 2025);

  loadProfileData();
  setupAI();
  Serial.println("Step 7 - AI ready");
  displayHome();
  Serial.println("Step 8 - display home done");
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
      btnMenuCard.press(btnMenuCard.contains(mappedX, mappedY));
    } else if (currentScreen == PROFILE_SCREEN) {
      btnPrevField.press(btnPrevField.contains(mappedX, mappedY));
      btnNextField.press(btnNextField.contains(mappedX, mappedY));
      btnMinus.press(btnMinus.contains(mappedX, mappedY));
      btnPlus.press(btnPlus.contains(mappedX, mappedY));
      btnBack.press(btnBack.contains(mappedX, mappedY));
      btnConfirm.press(btnConfirm.contains(mappedX, mappedY));
    } else if (currentScreen == SAMPLE_SCREEN) {
      btnBack.press(btnBack.contains(mappedX, mappedY));
    } else if (currentScreen == SETTINGS_SCREEN) {
      btnPrevField.press(btnPrevField.contains(mappedX, mappedY));
      btnNextField.press(btnNextField.contains(mappedX, mappedY));
      btnMinus.press(btnMinus.contains(mappedX, mappedY));
      btnPlus.press(btnPlus.contains(mappedX, mappedY));
      btnBack.press(btnBack.contains(mappedX, mappedY));
      btnConfirm.press(btnConfirm.contains(mappedX, mappedY));
    } else if (currentScreen == HISTORY_SCREEN) {
      btnBack.press(btnBack.contains(mappedX, mappedY));
      btnConfirm.press(btnConfirm.contains(mappedX, mappedY));
    }

  } else {
    btnUnlock.press(false);
    btnBack.press(false);
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

  } else if (currentScreen == MENU_SCREEN) {
    if (btnBack.justReleased()) {
      if (ignoreMenuBackOnce) {
        ignoreMenuBackOnce = false;
      } else {
        displayHome();
        return;
      }
    }

    unsigned long now = millis();
    if (gest == "SWIPE LEFT" || gest == "SWIPE RIGHT") {
      if (now - lastGestureMs > GESTURE_LOCKOUT_MS) {
        menuIndex = (gest == "SWIPE LEFT")
            ? (menuIndex + 1) % MENU_COUNT
            : (menuIndex - 1 + MENU_COUNT) % MENU_COUNT;
        drawMenuCarousel();
        lastGestureMs = now;
      }
      return;
    }
    if (btnMenuCard.justReleased()) {
      if (now - lastGestureMs > GESTURE_LOCKOUT_MS) openSelectedMenuItem();
      return;
    }

  } else if (currentScreen == PROFILE_SCREEN) {
    if (btnBack.justReleased())    { loadProfileData(); drawMenuCarousel(); }
    if (btnConfirm.justReleased()) { writeProfileData(); drawMenuCarousel(); }
    if (btnPrevField.justReleased()) { fieldIndex = (fieldIndex - 1 + NUM_FIELDS) % NUM_FIELDS; displayProfileMenu(); }
    if (btnNextField.justReleased()) { fieldIndex = (fieldIndex + 1) % NUM_FIELDS; displayProfileMenu(); }
    if (btnMinus.justReleased()) {
      if (fieldIndex == 1) values[1] = !values[1];
      else if (fieldIndex == 4) values[4] = (values[4] - 1 + 3) % 3;
      else values[fieldIndex]--;
      displayProfileMenu();
    }
    if (btnPlus.justReleased()) {
      if (fieldIndex == 1) values[1] = !values[1];
      else if (fieldIndex == 4) values[4] = (values[4] + 1) % 3;
      else values[fieldIndex]++;
      displayProfileMenu();
    }

  } else if (currentScreen == SAMPLE_SCREEN) {
    if (btnBack.justReleased()) { disableMAX30102(); drawMenuCarousel(); }

  } else if (currentScreen == SETTINGS_SCREEN) {
    if (btnBack.justReleased()) { settings_loaded = false; drawMenuCarousel(); }
    if (btnConfirm.justReleased()) {
      rtc.setTime(0, settings_values[1], settings_values[0],
                  settings_values[3], settings_values[2], settings_values[4]);
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
        case 0: settings_values[0] = (settings_values[0] - 1 + 24) % 24; break;
        case 1: settings_values[1] = (settings_values[1] - 1 + 60) % 60; break;
        case 2: settings_values[2] = (settings_values[2] - 1 < 1) ? 12 : settings_values[2] - 1; break;
        case 3: settings_values[3] = (settings_values[3] - 1 < 1) ? 31 : settings_values[3] - 1; break;
        case 4: settings_values[4]--; break;
      }
      displaySettingsMenu();
    }
    if (btnPlus.justReleased()) {
      switch (settings_fieldIndex) {
        case 0: settings_values[0] = (settings_values[0] + 1) % 24; break;
        case 1: settings_values[1] = (settings_values[1] + 1) % 60; break;
        case 2: settings_values[2] = (settings_values[2] + 1 > 12) ? 1 : settings_values[2] + 1; break;
        case 3: settings_values[3] = (settings_values[3] + 1 > 31) ? 1 : settings_values[3] + 1; break;
        case 4: settings_values[4]++; break;
      }
      displaySettingsMenu();
    }

  } else if (currentScreen == HISTORY_SCREEN) {
    unsigned long now = millis();

    if ((gest == "SWIPE LEFT" || gest == "SWIPE RIGHT") &&
        (now - lastGestureMs > GESTURE_LOCKOUT_MS)) {

      if (gest == "SWIPE LEFT") {
        if (historyView == 0) displayHistoryGraph();
        else if (historyView == 1) displayHistoryClearMenu();
        lastGestureMs = now;
        return;
      } else if (gest == "SWIPE RIGHT") {
        if (historyView == 2) displayHistoryGraph();
        else if (historyView == 1) displayHistoryMenu();
        lastGestureMs = now;
        return;
      }
    }

    if (btnBack.justReleased()) { drawMenuCarousel(); return; }

    if (historyView == 2 && btnConfirm.justReleased()) {
      clearHistoryData();
      displayHistoryMenu();
      return;
    }
  }

  sleep_param();
  delay(15);
}
