// Edited from https://github.com/tensorflow/tflite-micro-arduino-examples/blob/main/examples/hello_world/hello_world.ino

#include <TensorFlowLite_ESP32.h>

#include "mlp.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
constexpr int kTensorArenaSize = 80 * 1024;
uint8_t* tensor_arena = nullptr;

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

bool model_ready = false;
} // namespace

void setup() {
  Serial.begin(115200);
  Serial.print("Free heap: ");
  Serial.println(ESP.getFreeHeap());
  tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
if (tensor_arena == nullptr) {
  Serial.println("Failed to allocate tensor arena");
  return;
}
  tflite::InitializeTarget();

  // Map the model into a usable data structure.
  model = tflite::GetModel(model_weights_mlp_15s_vitaldb_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema version mismatch: ");
    Serial.print(model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in all operation implementations.
  static tflite::AllOpsResolver resolver;

  // Set up error reporter.
  static tflite::MicroErrorReporter micro_error_reporter;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }
input = interpreter->input(0);
output = interpreter->output(0);
model_ready = true;
  model_ready = true;
}

void loop() {
    if (!model_ready) return;
  
  static bool printed = false;
  
  float in_scale = input->params.scale;
  int in_zero_point = input->params.zero_point;
  float standardscalar[]={5.87907639e+01,6.11347983e+01,1.62614850e+02,1.24806867e-01,8.45733021e-01,8.22931007,1.14888411e-03,8.24479980e-01,1.40127322e+00,2.94353759,7.33692993e-02,4.29523154e-02};
  float standardmean[]={1.51789513e+01,1.22301296e+01,1.07160379e+01,3.30499793e-01,1.65342310e-01,4.27223254e+00,7.51964371e-04,4.61487307e-01,2.92132266e-01,2.73570936e-01,3.53447231e-02,7.01600395e-02};
 

  input->data.int8[0]  =(int8_t)((((77.0-standardmean[0])/standardscalar[0])/ in_scale)+in_zero_point);  // Age repplace the float number with the global variable in sample
  input->data.int8[1]  =(int8_t)((((7.5-standardmean[1])/standardscalar[1])/in_scale)+in_zero_point);  // Weight
  input->data.int8[2]  =(int8_t)((((160.2-standardmean[2])/standardscalar[2])/in_scale)+in_zero_point);  // Height
  input->data.int8[3]  =(int8_t)((((0.0-standardmean[3])/standardscalar[3])/in_scale)+in_zero_point);  // PreOpDm
  input->data.int8[4]  =(int8_t)((((0.6551818181818181-standardmean[4])/standardscalar[4])/in_scale)+in_zero_point);  // PPGMEANINTERVAL
  input->data.int8[5]  =(int8_t)((((11.933197357447153-standardmean[5])/standardscalar[5])/in_scale)+in_zero_point);  // PPGSTD
  input->data.int8[6]  =(int8_t)((((-0.3698673265974342-standardmean[6])/standardscalar[6])/in_scale)+in_zero_point);  // PPGTEAGER
  input->data.int8[7]  =(int8_t)((((2.586104761259092-standardmean[7])/standardscalar[7])/in_scale)+in_zero_point);  // PPGSKEW
  input->data.int8[8]  =(int8_t)((((0.0009382611054101664-standardmean[8])/standardscalar[8])/in_scale)+in_zero_point);  // PPGIQR
  input->data.int8[9]  =(int8_t)((((8.922658299525736-standardmean[9])/standardscalar[9])/in_scale)+in_zero_point);  // PPGENTROPY
  input->data.int8[10] =(int8_t)((((0.049861116768918606-standardmean[10])/standardscalar[10])/in_scale)+in_zero_point); //first deriv max
  input->data.int8[11] = (int8_t)((((0.004292026048768319-standardmean[11])/standardscalar[11])/in_scale)+in_zero_point);// PPGSTDPP
  
  char c = Serial.read();
  if (c == 'S' || c == 's') {
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke() failed");
      return;
    }

    float scale = output->params.scale;
    int zero_point = output->params.zero_point;

    // Read and dequantize
    int8_t raw = output->data.int8[0];
    float result = (raw - zero_point) * scale;


    Serial.println(result);
    Serial.println((int)input->type);
  }
}