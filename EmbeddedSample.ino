#include <Wire.h>
#include "MAX30105.h"
#include "Arduino.h"
#include "heartRate.h"
#include "arduinoFFT.h"
unsigned long lastSampletime = 0;
MAX30105 particleSensor;
// Defines the arrays for later calculations, can be changed, however these have to be very large to be able to hold samples, will affect storage.
float irarray[3000];           // raw IR values
int arraysize[3000];           // index array
float heartbeattime[3000];
unsigned long timearray[3000]; // timestamps
unsigned long P2PTime_Array[3000];//This is for the p2p time
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
// ------------- All I know is pain and suffering globals for stupid FFT
//still need to setup

//----------------------This will be for heartrate globals im sure this can be all done better but this is how I know how to do it
int p=0;; // this is anouther counter
const byte RATE_SIZE = 4; //Increase this for more averaging. 4 is good.
byte rates[RATE_SIZE]; //Array of heart rates
byte rateSpot = 0;
long lastBeat = 0; //Time at which the last beat occurred

float beatsPerMinute;
int beatAvg;
// --------------------- Initial Setup
void setup() {
  Serial.begin(115200); // high baud for fast dumping it can be set to the 115 baud rate, was just testing how it would affect sample rate
  Wire.begin();

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found");
    while (1);
  }

  // MAX30102 setup: LED current, sample rate = 100 HZ pulse width has to be edited, have yet to find a good compromise
  particleSensor.setup();//particleSensor.setup(60, 1, 2, 100, 411, 4096); // this is fine for now, it is important to note that I have not yet tuned this completley
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.setPulseAmplitudeRed(0x1F);

  Serial.println("Type S to start, type F to end"); //This doesnt even print to begin with lol
}

void loop() {
  // Check for serial input
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 's' || c == 'S') {
      startRequested = true;
      i = 0;
      initial = millis();
      Serial.println("Starting now");
      //-------------------------- Resets all values
      FIRST_DERIVMAX = 0; // The first derivative max
      FIRST_DERIV = 0;    // The first derivative for the loop
      mean = 0;           // basic mean for skewness
      stddev = 0;         // basic std deviation for skewness
      skewness = 0;       // skewness
      iqr = 0;            //interquartile range
      teager = 0;         // Teager energy
      teager_max = 0;     // Teagerr energy max
      p=0;
       lastBeat = 0; //Time at which the last beat occurred
      beatAvg=0;
    }
    if (c == 'f' || c == 'F') {
      startRequested = false;
      // This is basically just debugging and checking samples
      // Print all captured samples
      for (int j = 0; j < i; j++) {
        Serial.print(arraysize[j]);
        Serial.print(",");
        Serial.println(irarray[j]);
      }
      Serial.println("All data printed.");
      POWEROF2();
      FIRST_DERIV_MAX();
      SKEWNESS_CALC();
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

    }
  }
  // This can be its own seperate function idk how that affects the timing tho
  particleSensor.check();

  if (!startRequested) {
    // Skip any pending samples in FIFO
    while (particleSensor.available()) {
      particleSensor.nextSample();
    }
    return;
  }

  // Capture samples
  if (startRequested && particleSensor.available()) {
    uint32_t ir = particleSensor.getIR(); // raw IR
    // Mean p2p
    irarray[i] = ir;
    timearray[i] = millis();
    if(checkForBeat(ir)==true){
      long delta=millis()-lastBeat;
      lastBeat=millis();
      P2PTime_Array[p]=delta/1000; //seconds
      p++; // this needs to be reset to 0 for each sample
      beatsPerMinute=60/(delta/1000.0);
      if(beatsPerMinute<255 && beatsPerMinute >20){
        rates[rateSpot++]=(byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;
        //take average of readings
        beatAvg=0;
        for(byte x=0; x<RATE_SIZE; x++)
          beatAvg += rates[x];
          beatAvg /= RATE_SIZE;
      }
      Serial.print(beatAvg); 
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

//-------------------------------------------------------------------------
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
  p = 1;
  while ((p << 1) <= i) {
    p <<= 1;
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
    stddevTime += sq((P2PTime_Array[j] - mean));
  }
  stddevTime = sqrt(stddev / (p + 0.0));
}





