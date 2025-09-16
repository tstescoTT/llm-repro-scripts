# Repro script usage



## repro_concurrent_llm_requests.py

```bash
curl -L -o repro_concurrent_llm_requests.py https://raw.githubusercontent.com/tstescoTT/llm-repro-scripts/main/repro_concurrent_llm_requests.py
export API_TOKEN="my-api-key"
# run 1x prompt for trace capture
python3 repro_concurrent_llm_requests.py --concurrency 1 --loops 1

# run with quick timeouts to abort on vLLM server
python3 repro_concurrent_llm_requests.py --concurrency 32 --loops 3 --timeout 0.1
```
