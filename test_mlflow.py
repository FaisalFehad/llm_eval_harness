import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

print("Loading model...")
model = mlflow.pyfunc.load_model("models:/v15-student-6bit@production")

print("Running inference...")
result = model.predict(pd.DataFrame([{
    "title": "Senior Node.js Developer",
    "location": "London, UK",
    "jd_text": "We need a Senior Node.js Developer to build APIs with Express and PostgreSQL. Docker and AWS experience required. Salary: 75000 to 95000 GBP.",
}]))

print(result["response"].iloc[0])
