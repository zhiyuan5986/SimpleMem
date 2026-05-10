export HF_ENDPOINT=https://hf-mirror.com
model_name="deepseek-chat"
# base_url="http://localhost:8000/v1"
base_url="https://models.sjtu.edu.cn/api/v1"
# api_key="sk-Of0btqIp3GW0Jv65PhkqbQ" # openclaw
api_key="sk-tYfudtgpHIqCKpxWisXgtA" # sjtu
datetime=$(date +%Y%m%d%H%M%S)
# cp -r ./lancedb_data_deepseek-chat_all-MiniLM-L6-v2 ./lancedb_data_deepseek-chat_all-MiniLM-L6-v2-copy
python test_locomo10_dualview_qa.py --dataset test_ref/locomo10.json \
    --num-samples 10 \
    --db-path ./lancedb_data_deepseek-chat_all-MiniLM-L6-v2 \
    --result-file "./deepseek-chat_all-MiniLM-L6-v2/locomo10_dualview_results_$datetime.json" \
    --llm-api-key $api_key \
    --llm-model $model_name \
    --llm-base-url $base_url \
    --embedding-model /mnt/sdb/liuqiaoan/all-MiniLM-L6-v2 \
    --no-answer-generation \
    --semantic-top-k 25 \
    --keyword-top-k 5 \
    --no-enable-planning \
    --no-enable-reflection \
    --no-enable-parallel-retrieval \
    --question-processing-mode parallel \
    --question-workers 8 \
    --keyword-extraction-mode llm \
    --raw-semantic-top-k 25 \
    --raw-keyword-top-k 5 \
    --mem-sem-weight 0.65 \
    --mem-lex-weight 0.35 \
    --raw-sem-weight 0.45 \
    --raw-lex-weight 0.55 \
    --final-mem-weight 1 \
    --final-raw-weight 0 \
    --final-agree-weight 0
    # --mem-sem-weight 0.65 \
    # --mem-lex-weight 0.35 \
    # --raw-sem-weight 0.45 \
    # --raw-lex-weight 0.55 \
    # --final-mem-weight 0.45 \
    # --final-raw-weight 0.45 \
    # --final-agree-weight 0.1


