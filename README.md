# Repro script usage



## repro_concurrent_llm_requests.py

```bash
curl -L -o repro_concurrent_llm_requests.py https://raw.githubusercontent.com/tstescoTT/llm-repro-scripts/refs/heads/main/repro_concurrent_llm_requests.py
export API_TOKEN="my-api-key"

# run with quick timeouts to abort on vLLM server
python3 repro_concurrent_llm_requests.py --concurrency 32 --loops 3 --timeout 15
```




## text_repro.py

Download and run in tt-metal project:
```bash
cd tt-metal/models/demos/llama3_70b_galaxy/demo
curl -L -o text_repro.py https://raw.githubusercontent.com/tstescoTT/llm-repro-scripts/refs/heads/main/text_repro.py
curl -L -o sample_prompts/repro_prompt_isl_2500.json https://raw.githubusercontent.com/tstescoTT/llm-repro-scripts/refs/heads/main/sample_prompts/repro_prompt_isl_2500.json

pytest models/demos/llama3_70b_galaxy/demo/text_repro.py -k repro-isl-2500
```