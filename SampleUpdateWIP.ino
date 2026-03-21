#include <Wire.h>
#include "MAX30105.h"
#include "Arduino.h"
#include "heartRate.h"
#include "arduinoFFT.h"
#include "math.h"

unsigned long lastSampletime = 0;
MAX30105 particleSensor;
// Defines the arrays for later calculations, can be changed, however these have to be very large to be able to hold samples, will affect storage.
float irarray[3000];           // raw IR values
int arraysize[3000];           // index array
float heartbeattime[3000];
unsigned long timearray[3000]; // timestamps
unsigned long P2PTime_Array[3000]; // This is for the p2p time
int i = 0; // counts the numbers of samples VERY VERY VERY IMPORTANT
unsigned long initial;
bool startRequested = false;

// ------------ Globals for history in the future as well as inputs into the AI model
float FIRST_DERIVMAX = 0; // The first derivative max
float FIRST_DERIV = 0;    // The first derivative for the loop
float mean = 0;            // basic mean for normalization
float stddev = 0;          // basic std deviation for normalization
float skewness = 0;        // skewness
float iqr = 0;             // Interquartile Range
float teager = 0;          // Teager energy
float teager_max = 0;      // Mean absolute Teager energy (renamed semantically, variable kept for compatibility)
float meanP2Pinterval = 0; // Finds the mean P2P interval
float stddevTime = 0;      // Standard deviation for P2P interval

// Normalized signal buffer — shared across all feature functions
static float norm_ir[3000];

// ------------- FFT globals
int h = 0;
int SAMPLES;
#define SAMPLING_FREQUENCY 100
float entropy;
double *vReal = NULL;
double *vImag = NULL;
ArduinoFFT<double> *FFT = NULL;

// ---------------------- Heart rate globals
int p = 0;
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;

float beatsPerMinute;
int beatAvg;

// --------------------- Initial Setup
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
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 's' || c == 'S') {
      startRequested = true;
      i = 0;
      initial = millis();
      Serial.println("Starting now");
      // Reset all values
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
    }
    if (c == 'f' || c == 'F') {
      startRequested = false;
      // Print all captured samples
      for (int j = 0; j < i; j++) {
        Serial.print(arraysize[j]);
        Serial.print(",");
        Serial.println(irarray[j]);
      }
      Serial.println("All data printed.");

      // Call order: POWEROF2 and NORMALIZE_SIGNAL must come first
      POWEROF2();
      NORMALIZE_SIGNAL();   // z-score normalize into norm_ir[] before all features
      FIRST_DERIV_MAX();
      SKEWNESS_CALC();
      FFT_SPECTRAL();
      IQR_CALC();
      TEAGER_ENERGY();
      TIME_STD_DEV();

      Serial.print(FIRST_DERIVMAX);
      Serial.print(" ");
      Serial.print(skewness);
      Serial.print(" ");
      Serial.print(iqr);
      Serial.print(" ");
      Serial.print(teager_max);
      Serial.print(" ");
      Serial.print(beatAvg);
      Serial.print(" ");
      Serial.print(stddevTime);
      Serial.print(" ");
      Serial.print(entropy);
      Serial.print(" ");
      Serial.print(h);
    }
  }

  particleSensor.check();

  if (!startRequested) {
    while (particleSensor.available()) {
      particleSensor.nextSample();
    }
    return;
  }

  // Capture samples
  if (startRequested && particleSensor.available()) {
    uint32_t ir = particleSensor.getIR();
    irarray[i] = ir;
    timearray[i] = millis();

    if (checkForBeat(ir) == true) {
      long delta = millis() - lastBeat;
      lastBeat = millis();
      P2PTime_Array[p] = delta / 1000; // seconds
      p++;
      beatsPerMinute = 60 / (delta / 1000.0);
      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;
        beatAvg = 0;
        for (byte x = 0; x < RATE_SIZE; x++)
          beatAvg += rates[x];
        beatAvg /= RATE_SIZE;
      }
    }

    arraysize[i] = i;
    i++;
    particleSensor.nextSample();

    // Stop automatically after 30 seconds
    if (millis() - initial >= 30000) {
      startRequested = false;
      Serial.println("Full range done");
    }
  }
}

// =========================================================
// NORMALIZE_SIGNAL
// Z-score normalizes irarray[] into norm_ir[]
// Must be called before all other feature functions.
// Matches Python: ppg_series_clean = (x - mean) / std
// =========================================================
void NORMALIZE_SIGNAL() {
  // Mean
  mean = 0;
  for (int j = 0; j < i; j++) mean += irarray[j];
  mean /= (float)i;

  // Population std dev (N denominator — matches numpy.std default)
  stddev = 0;
  for (int j = 0; j < i; j++) stddev += sq(irarray[j] - mean);
  stddev = sqrt(stddev / (float)i);

  // Populate normalized buffer
  if (stddev == 0) {
    for (int j = 0; j < i; j++) norm_ir[j] = 0;
  } else {
    for (int j = 0; j < i; j++) norm_ir[j] = (irarray[j] - mean) / stddev;
  }
}

// =========================================================
// FIRST_DERIV_MAX
// Per-sample diff on z-scored signal — no time division.
// Matches Python: derivative = np.diff(ppg_series_clean)
//                 features['ppg_first_deriv_max'] = np.max(derivative)
// =========================================================
void FIRST_DERIV_MAX() {
  FIRST_DERIVMAX = -1e30;
  for (int j = 1; j < i; j++) {
    float deriv = norm_ir[j] - norm_ir[j - 1];
    if (deriv > FIRST_DERIVMAX) FIRST_DERIVMAX = deriv;
  }
}

// =========================================================
// SKEWNESS_CALC
// Adjusted Fisher-Pearson coefficient on z-scored signal.
// Matches Python: scipy.stats.skew() (default bias=False)
// =========================================================
void SKEWNESS_CALC() {
  // Compute mean and std of normalized signal (should be ~0 and ~1)
  float norm_mean = 0;
  for (int j = 0; j < i; j++) norm_mean += norm_ir[j];
  norm_mean /= (float)i;

  float norm_std = 0;
  for (int j = 0; j < i; j++) norm_std += sq(norm_ir[j] - norm_mean);
  norm_std = sqrt(norm_std / (float)i);

  skewness = 0;
  if (norm_std == 0) return;

  for (int j = 0; j < i; j++) {
    float temp = (norm_ir[j] - norm_mean) / norm_std;
    skewness += temp * temp * temp;
  }
  // Adjusted Fisher-Pearson — same formula as scipy.stats.skew()
  skewness = ((float)i / ((float)(i - 1) * (float)(i - 2))) * skewness;
}

// =========================================================
// IQR_CALC
// Linear interpolation percentile on z-scored signal.
// Matches Python: np.percentile(ppg_series_clean, 75)
//                 - np.percentile(ppg_series_clean, 25)
// =========================================================
void IQR_CALC() {
  static float temp[3000];
  for (int j = 0; j < i; j++) temp[j] = norm_ir[j]; // use normalized signal

  // Sort ascending
  for (int a = 0; a < i - 1; a++) {
    for (int b = a + 1; b < i; b++) {
      if (temp[b] < temp[a]) {
        float t = temp[a]; temp[a] = temp[b]; temp[b] = t;
      }
    }
  }

  // Linear interpolation — matches numpy.percentile default
  auto percentile_interp = [&](float pct) -> float {
    float idx = pct * (float)(i - 1);
    int lo = (int)idx;
    int hi = lo + 1;
    if (hi >= i) return temp[i - 1];
    float frac = idx - (float)lo;
    return temp[lo] + frac * (temp[hi] - temp[lo]);
  };

  float Q1 = percentile_interp(0.25f);
  float Q3 = percentile_interp(0.75f);
  iqr = Q3 - Q1;
}

// =========================================================
// TEAGER_ENERGY
// Mean of absolute Teager values on z-scored signal.
// Matches Python: energy = x[1:-1]**2 - x[:-2] * x[2:]
//                 return np.mean(np.abs(energy))
// Note: variable teager_max kept for serial output compatibility
// but now holds the MEAN, not the max.
// =========================================================
void TEAGER_ENERGY() {
  if (i < 3) { teager_max = 0; return; }
  float sum = 0;
  int count = 0;
  for (int j = 1; j < i - 1; j++) {
    float t = norm_ir[j] * norm_ir[j] - norm_ir[j - 1] * norm_ir[j + 1];
    sum += fabsf(t);
    count++;
  }
  teager_max = (count > 0) ? sum / (float)count : 0;
}

// =========================================================
// POWEROF2
// Finds largest power of 2 <= i for FFT sizing. Unchanged.
// =========================================================
void POWEROF2() {
  h = 1;
  while ((h << 1) <= i) {
    h <<= 1;
  }
}

// =========================================================
// FFT_SPECTRAL
// Power spectrum (squared magnitudes), no windowing, includes DC.
// Matches Python: psd = np.abs(np.fft.rfft(x))**2
//                 psd_norm = psd / np.sum(psd)
//                 entropy = -sum(psd_norm * log(psd_norm))
// Changes from original:
//   - No Hamming windowing (Python doesn't apply one)
//   - Squared magnitudes (power) not raw magnitudes
//   - Starts at index 0 to include DC, matching rfft output
//   - Input is norm_ir[] (z-scored signal)
// =========================================================
void FFT_SPECTRAL() {
  SAMPLES = h;
  vReal = new double[SAMPLES];
  vImag = new double[SAMPLES];

  // Load normalized signal — no windowing to match Python rfft
  for (int j = 0; j < SAMPLES; j++) {
    vReal[j] = (j < i) ? norm_ir[j] : 0.0;
    vImag[j] = 0.0;
  }

  if (FFT != NULL) { delete FFT; FFT = NULL; }
  FFT = new ArduinoFFT<double>(vReal, vImag, SAMPLES, SAMPLING_FREQUENCY);

  FFT->compute(FFTDirection::Forward); // no windowing call
  FFT->complexToMagnitude();           // vReal[j] = magnitude

  // Square to get power spectrum; use first half + DC (rfft equivalent)
  int n_bins = SAMPLES / 2 + 1;
  float psd_sum = 0;
  for (int j = 0; j < n_bins; j++) {
    vReal[j] = vReal[j] * vReal[j]; // power
    psd_sum += vReal[j];
  }

  // Spectral entropy
  entropy = 0;
  if (psd_sum > 0) {
    for (int l = 0; l < n_bins; l++) {
      float k = vReal[l] / psd_sum;
      if (k > 0) entropy -= k * log(k);
    }
  }

  delete FFT; FFT = NULL;
  delete[] vReal; vReal = NULL;
  delete[] vImag; vImag = NULL;
}

// =========================================================
// TIME_STD_DEV
// BUGFIX: original used wrong variable (stddev) in sqrt.
// Now correctly accumulates into stddevTime before sqrt.
// Matches Python: np.std(pp_intervals) / fs
// =========================================================
void TIME_STD_DEV() {
  if (p == 0) { stddevTime = 0; return; }

  meanP2Pinterval = 0;
  for (int j = 0; j < p; j++) meanP2Pinterval += P2PTime_Array[j];
  meanP2Pinterval /= (float)p;

  stddevTime = 0;
  for (int j = 0; j < p; j++) stddevTime += sq(P2PTime_Array[j] - meanP2Pinterval);
  stddevTime = sqrt(stddevTime / (float)p); // FIXED: was sqrt(stddev / p)
}
