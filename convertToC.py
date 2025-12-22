import joblib
import emlearn
import time

# Start tracking program time
start_time = time.time()

# Load model
model = joblib.load('./model_weights/small_rf_glucose_model.pkl')
print("Model loaded!")

# Convert to compact C code
c_code = emlearn.convert(model, method='inline')
print("Model converted to compact C code!")

# Save to header + source files
c_code.save(name='small_rf_glucose', file='./model_weights/small_rf_glucose.h')

print("Compact C code generated in model weights folder!")

# Print total program time
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_hours = int(elapsed_time / 3600)
elapsed_minutes = int((elapsed_time % 3600) / 60)
elapsed_seconds = int(elapsed_time % 60)

print(f"Total program time: {elapsed_hours} hours, {elapsed_minutes} minutes, {elapsed_seconds} seconds")