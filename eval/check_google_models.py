# check_google_models.py
import google.generativeai as genai

# --- PASTE YOUR GOOGLE API KEY HERE ---
GOOGLE_API_KEY = "xxx" 

print("Configuring API with your key...")
genai.configure(api_key=GOOGLE_API_KEY)
print("Configuration complete.")
print("\n--- Requesting available models from Google ---")

try:
    model_list = []
    for model in genai.list_models():
        # We only care about models that support content generation
        if 'generateContent' in model.supported_generation_methods:
            model_list.append(model.name)

    print("\n--- SUCCESS: Found the following models available for your key ---")
    for name in sorted(model_list):
        print(name)

except Exception as e:
    print("\n--- FAILED to get model list ---")
    print("The specific error is:")
    print(e)
