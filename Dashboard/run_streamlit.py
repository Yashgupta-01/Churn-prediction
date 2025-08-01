import subprocess
from pyngrok import ngrok
import time

# Kill any existing tunnels (prevention)
ngrok.kill()

# Set port
port = 8501

# Start ngrok tunnel
public_url = ngrok.connect(port)
print("🔗 Streamlit App Public URL:", public_url)

# Wait for tunnel to settle
time.sleep(2)

# Start Streamlit
subprocess.run(["streamlit", "run", "/content/drive/MyDrive/Telecom churn prediction/dashboard/app.py", "--server.port", str(port)])
