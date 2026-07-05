"""Load test for the fraud-scoring API.

    pip install locust
    locust -f serving/locustfile.py --host http://localhost:7860
    # or against the live Space:
    # locust -f serving/locustfile.py --host https://leonardasvekrikas-source-fraud-detection-demo.hf.space

Open http://localhost:8089, set users + spawn rate, and watch p50/p95/p99 + RPS.
"""

from locust import HttpUser, between, task

# A real fraudulent transaction (also in docs/sample_transactions.json).
FRAUD_PAYLOAD = {
    "features": {
        "Time": 406.0, "V1": -2.312227, "V2": 1.951992, "V3": -1.609851, "V4": 3.997906,
        "V5": -0.522188, "V6": -1.426545, "V7": -2.537387, "V8": 1.391657, "V9": -2.770089,
        "V10": -2.772272, "V11": 3.202033, "V12": -2.899907, "V13": -0.595222, "V14": -4.289254,
        "V15": 0.389724, "V16": -1.140747, "V17": -2.830056, "V18": -0.016822, "V19": 0.416956,
        "V20": 0.126911, "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198,
        "V25": 0.044519, "V26": 0.17784, "V27": 0.261145, "V28": -0.143276, "Amount": 0.0,
    },
    "top_k": 6,
}


class FraudScoringUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def score(self):
        self.client.post("/score", json=FRAUD_PAYLOAD, name="/score")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
