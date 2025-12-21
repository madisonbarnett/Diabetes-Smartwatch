import joblib
import emlearn

# Load your best original model
model = joblib.load('./model_weights/rf_glucose_model.pkl')

# Convert using the efficient loadable method
c_code = emlearn.convert(model, method='inline')

# Save to header + source files
c_code.save(name='rf_glucose', file='./model_weights/rf_glucose.h')

print("Compact C code generated in model weights folder!")