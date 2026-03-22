import app
from chatbot import init_session_metrics

history = [{"role": "assistant", "content": "hi"}]
cache = {}
metrics = init_session_metrics()

for q in ["I am getting fever", "I'm getting fever"]:
    try:
        out = app.submit_query(q, "Ananya", history, cache, metrics)
        print("QUERY:", q)
        print("STATUS OK, LAST BOT MSG:", out[2][-1]["content"][:180])
    except Exception as exc:
        print("QUERY:", q)
        print("ERROR:", repr(exc))
