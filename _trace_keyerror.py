import traceback
from chatbot import process_user_query, init_session_metrics

cache = {}
metrics = init_session_metrics()

queries = [
    "My stomach burns after spicy food",
    "I'm getting fever",
]

for q in queries:
    print("\nQUERY:", q)
    try:
        result = process_user_query(q, "Ananya", cache, metrics)
        print("STATUS:", result.get("status"))
        print("MESSAGE PREVIEW:", str(result.get("message", ""))[:140])
    except Exception:
        traceback.print_exc()
