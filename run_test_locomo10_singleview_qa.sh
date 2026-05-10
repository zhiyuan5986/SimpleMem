datetime=$(date +%Y%m%d%H%M%S)
python test_locomo10_singleview_qa.py --parallel-questions \
    --dataset test_ref/locomo10.json \
    --num-samples 10 \
    --db-path ./lancedb_data_deepseek-chat_all-MiniLM-L6-v2 \
    --result-file "./deepseek-chat_all-MiniLM-L6-v2/locomo10_singleview_results_$datetime.json" \
    --semantic-top-k 25 \
    --keyword-top-k 5 \
    --structured-top-k 5

