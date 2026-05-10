export HF_ENDPOINT=https://hf-mirror.com
top_k=20
python test_locomo10_token_cost.py --input-json deepseek-chat/locomo10_dualview_results_20260501153336.json \
                                   --top-k $top_k \
                                   --output-json deepseek-chat/locomo10_dualview_results_20260501153336_token_cost_top$top_k.json \
                                   --db-path ./lancedb_data_deepseek-chat_all-MiniLM-L6-v2 \
                                   --embedding-model /mnt/sdb/liuqiaoan/all-MiniLM-L6-v2 \
                                   --categories 1,2,3,4