import google.generativeai as genai

genai.configure(api_key="AIzaSyDi0NBS-Oz7uPkjlqykS_9JoSKIAYwNOEo")

model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content("Say hello")
print(response.text)
