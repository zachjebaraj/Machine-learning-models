import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random

# Step 1: Generate Synthetic Data
def generate_user_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        avg_screen_time = round(random.uniform(1, 10), 2)  # hours per day
        unlocks = random.randint(10, 150)  # unlocks per day
        session_duration = round(random.uniform(2, 20), 2)  # average minutes per session
        night_usage_ratio = round(random.uniform(0, 1), 2)  # 0 = none, 1 = always
        social_media_time = round(random.uniform(0.5, avg_screen_time), 2)
        notification_response_time = round(random.uniform(1, 20), 2)  # in seconds

        # Simple logic to label addiction risk
        if avg_screen_time > 6 or unlocks > 100 or night_usage_ratio > 0.7:
            label = "High"
        elif avg_screen_time > 3 or unlocks > 50:
            label = "Medium"
        else:
            label = "Low"

        data.append([
            avg_screen_time, unlocks, session_duration, night_usage_ratio,
            social_media_time, notification_response_time, label
        ])

    return pd.DataFrame(data, columns=[
        "avg_screen_time", "unlocks", "session_duration", "night_usage_ratio",
        "social_media_time", "notification_response_time", "addiction_level"
    ])

# Step 2: Train the ML Model
df = generate_user_data()
X = df.drop("addiction_level", axis=1)
y = df["addiction_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Evaluate the Model
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Step 4: Predict & Control Logic
def assess_user(input_data):
    df_input = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(df_input)[0]
    print(f"\n⚠️ Addiction Risk Level: {prediction}")

    if prediction == "High":
        print("💡 Suggestion: You may need a digital detox. Try turning off notifications or using app blockers.")
    elif prediction == "Medium":
        print("✅ Suggestion: Consider setting screen time limits and taking regular breaks.")
    else:
        print("🎉 Great job! Keep maintaining a healthy tech-life balance.")

# Example User Input
user_input = {
    "avg_screen_time": 7.5,
    "unlocks": 120,
    "session_duration": 15,
    "night_usage_ratio": 0.85,
    "social_media_time": 4.2,
    "notification_response_time": 5
}

assess_user(user_input)
