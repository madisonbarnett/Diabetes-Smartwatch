#include <Wire.h>
#include "MAX30105.h"
#include "Arduino.h"
#include "math.h"

MAX30105 particleSensor;

const int SAMPLE_RATE_HZ = 100;
const int MAX_SAMPLES = 1500;
const unsigned long CAPTURE_DURATION_MS = 15000;
const int MIN_PEAK_DISTANCE = (int)(SAMPLE_RATE_HZ * 0.4f);

unsigned long initial = 0;
bool startRequested = false;

float irarray[MAX_SAMPLES];
int arraysize[MAX_SAMPLES];
unsigned long timearray[MAX_SAMPLES];
float P2PTime_Array[MAX_SAMPLES];

static float norm_ir[MAX_SAMPLES];
static float sort_buf[MAX_SAMPLES];

int i = 0;
int p = 0;

float FIRST_DERIVMAX = 0;
float mean = 0;
float stddev = 0;
float skewness = 0;
float iqr = 0;
float teager_max = 0;
float meanP2Pinterval = 0;
float stddevTime = 0;
float spectral_entropy = 0;
float ppg_freq = 0;

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
    return;
  }

  int n_bins = i / 2 + 1;
  float *psd = new float[n_bins];
  if (!psd) {
    spectral_entropy = 0;
    return;
  }

  for (int k = 0; k < n_bins; k++) {
    double real = 0.0;
    double imag = 0.0;

    for (int t = 0; t < i; t++) {
      double ang = 2.0 * PI * (double)k * (double)t / (double)i;
      real += (double)norm_ir[t] * cos(ang);
      imag -= (double)norm_ir[t] * sin(ang);
    }

    psd[k] = (float)(real * real + imag * imag);
  }

  float psd_sum = 0;
  for (int k = 0; k < n_bins; k++) psd_sum += psd[k];

  spectral_entropy = 0;
  if (psd_sum > 0) {
    for (int k = 0; k < n_bins; k++) {
      float pwr = psd[k] / psd_sum;
      if (pwr > 0) spectral_entropy -= pwr * log(pwr);
    }
  }

  delete[] psd;
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

  Serial.print(stddev, 12);
  Serial.print(",");
  Serial.print(meanP2Pinterval, 12);
  Serial.print(",");
  Serial.print(stddevTime, 12);
  Serial.print(",");
  Serial.print(teager_max, 12);
  Serial.print(",");
  Serial.print(skewness, 12);
  Serial.print(",");
  Serial.print(iqr, 12);
  Serial.print(",");
  Serial.print(spectral_entropy, 12);
  Serial.print(",");
  Serial.print(FIRST_DERIVMAX, 12);
  Serial.print(",");
  Serial.println(ppg_freq, 12);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found");
    while (1);
  }

  particleSensor.setup();
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.setPulseAmplitudeRed(0x1F);

  Serial.println("Type S to start, type F to end");
  Serial.println("ppg_std,ppg_mean_pp_interval_s,ppg_std_pp_interval_s,ppg_teager_energy,ppg_skew,ppg_iqr,ppg_spectral_entropy,ppg_first_deriv_max,ppg_freq");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == 's' || c == 'S') {
      resetFeatures();
      startRequested = true;
      initial = millis();
      Serial.println("Starting now");
    }

    if (c == 'f' || c == 'F') {
      if (startRequested && i > 0) {
        startRequested = false;
        PROCESS_AND_PRINT();
      }
    }
  }

  particleSensor.check();

  if (!startRequested) {
    while (particleSensor.available()) {
      particleSensor.nextSample();
    }
    return;
  }

  while (particleSensor.available() && startRequested) {
    if (i < MAX_SAMPLES) {
      irarray[i] = (float)particleSensor.getIR();
      timearray[i] = millis();
      arraysize[i] = i;
      i++;
    }

    particleSensor.nextSample();

    if ((millis() - initial >= CAPTURE_DURATION_MS) || (i >= MAX_SAMPLES)) {
      startRequested = false;
      Serial.println("Full range done");
      PROCESS_AND_PRINT();
      break;
    }
  }
}
