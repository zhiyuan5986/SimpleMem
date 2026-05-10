model_name="minimax-m2.5"
model_name="deepseek-chat"
# base_url="http://localhost:8000/v1"
base_url="https://models.sjtu.edu.cn/api/v1"
# api_key="sk-Of0btqIp3GW0Jv65PhkqbQ" # openclaw
api_key="sk-tYfudtgpHIqCKpxWisXgtA" # sjtu

export OPENAI_API_KEY=$api_key
export OPENAI_BASE_URL=$base_url

python consolidate_locomo10.py \
  --logs-dir ./deepseek-chat_all-MiniLM-L6-v2 \
  --db-search-roots ./lancedb_data_deepseek-chat_all-MiniLM-L6-v2 \
  --compressor-model-name /mnt/sdb/liuqiaoan/gpt2-dolly \
  --compressor-device-map cuda \
  --top-k 3 \
  --turn-window-k 3 \
  --condition-in-question after \
  --condition-text "Please focus on facts related to this memory entry." \
  --condition-placement prepend \
  --model $model_name \