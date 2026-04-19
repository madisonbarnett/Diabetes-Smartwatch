# Diabetes-Smartwatch-Embedded
This is the current code for the embedded side of the Non-Invasive Glucose Monitoring smartwatch

# Current Full Code for the Smartwatch
- `FinalCodeV1\DiabetesSmartwatch.ino`: This is the latest full code for the Smartwatch

# Current Sample Implementation
- `max30102_live_validation`: This is the current sample code for the smartwatch
Important note: This code uses 59 frequency bins from .5-4Hz for the DFT. A larger number of frequency bins will increase the processing time.

# Implementation
- All code was run through the Arduino IDE
- To flash the code, the GUI or AI model testing should be run with the file: `mlp.h`
- The screen needs to be configured on an individual basis
  - The order is as follows to configure the screen
    - Documents -> Arduino-> Librariers -> TFT_eSPI -> User Setup uses GC9A01 Driver
    - Pins must be configured in the user setup using the wiring diagram
    - Example: `FinalCodeV1\User_Setup.h`

# On-Device AI testing 
- `AI_Test`: Has the basic implementation of any AI model; for different models will need to change the model header
- For testing processed features or whatever method should be input, the shape of the model needs to be taken into account

#TFLite Micro
- Tflite Micro has a limited number of libraries that can be implemented. They are shown below:
<img width="540" height="1158" alt="image" src="https://github.com/user-attachments/assets/3e9ff629-a17c-43e4-98e9-96b7159f1d66" />
