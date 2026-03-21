// Edited from https://github.com/tensorflow/tflite-micro-arduino-examples/blob/main/examples/hello_world/hello_world.ino

#include <TensorFlowLite_ESP32.h>

#include "mlp_hist.h"
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
  model = tflite::GetModel(model_weights_mlp_hist_15s_vitaldb_int8_tflite);
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

  input->data.int8[0]  = (int8_t)((1.0f  / in_scale) + in_zero_point);  // Age repplace the float number with the global variable in sample
  input->data.int8[1]  = (int8_t)((2.0f  / in_scale) + in_zero_point);  // Weight
  input->data.int8[2]  = (int8_t)((3.0f  / in_scale) + in_zero_point);  // Height
  input->data.int8[3]  = (int8_t)((4.0f  / in_scale) + in_zero_point);  // PreOpDm
  input->data.int8[4]  = (int8_t)((5.0f  / in_scale) + in_zero_point);  // PPGMEANINTERVAL
  input->data.int8[5]  = (int8_t)((6.0f  / in_scale) + in_zero_point);  // PPGSTD
  input->data.int8[6]  = (int8_t)((7.0f  / in_scale) + in_zero_point);  // PPGTEAGER
  input->data.int8[7]  = (int8_t)((8.0f  / in_scale) + in_zero_point);  // PPGSKEW
  input->data.int8[8]  = (int8_t)((9.0f  / in_scale) + in_zero_point);  // PPGIQR
  input->data.int8[9]  = (int8_t)((10.0f / in_scale) + in_zero_point);  // PPGENTROPY
  input->data.int8[10] = (int8_t)((11.0f / in_scale) + in_zero_point);  // PPGFIRSTDERIVMAX
  input->data.int8[11] = (int8_t)((12.0f / in_scale) + in_zero_point);  // PPGSTDPP
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