# Program to convert trained model (saved in pkl file) to compact C file

import joblib
import emlearn
import time

# Start tracking program time
start_time = time.time()

# Load model
# Latest model pkl file size: ~72 MB
MODEL_NAME = 'rf_glucose_model'
model = joblib.load(f'./model_weights/{MODEL_NAME}.pkl')
print(f"Model '{MODEL_NAME}' loaded!")

# Convert to compact C code
c_code = emlearn.convert(model, method='inline')
print("Model converted to compact C code!")

# Save to C header file
c_code.save(name=MODEL_NAME, file=f'./model_weights/{MODEL_NAME}.h')

print("Compact C code generated in model weights folder!")

# Print total program time
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_hours = int(elapsed_time / 3600)
elapsed_minutes = int((elapsed_time % 3600) / 60)
elapsed_seconds = int(elapsed_time % 60)

print(f"Total program time: {elapsed_hours} hours, {elapsed_minutes} minutes, {elapsed_seconds} seconds")

# Latest C file size: ~963 MB
# Program time: 8 min 31 sec