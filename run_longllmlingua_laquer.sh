# model_name="Qwen3-VL-8B-Instruct"
# model_name="minimax-m2.5"
# model_name="qwen3vl"
model_name="deepseek-chat"
# base_url="http://localhost:8000/v1"
base_url="https://models.sjtu.edu.cn/api/v1"
# api_key="sk-Of0btqIp3GW0Jv65PhkqbQ" # openclaw
api_key="sk-tYfudtgpHIqCKpxWisXgtA" # sjtu

export OPENAI_API_KEY=$api_key
export OPENAI_BASE_URL=$base_url

python scripts/filter_extraction_trace_longllmlingua_laquer.py \
  --trace-json /home/liuqiaoan/Documents/SimpleMem/deepseek-chat/locomo10_sample_0_extraction_trace.json \
  --output-json /home/liuqiaoan/Documents/SimpleMem/deepseek-chat/locomo10_sample_0_longllmlingua_filtered.json \
  --compressor-model-name /mnt/sdb/liuqiaoan/gpt2-dolly \
  --compressor-device-map cuda \
  --top-k 1 \
  --turn-window-k 3 \
  --condition-in-question after \
  --condition-text "Please focus on facts related to this memory entry." \
  --condition-placement prepend \
  --model $model_name \
  # --first-stage-filter fine_topk_by_contrastive_ppl
  # --api-key $api_key \
  # --base-url $base_url \
