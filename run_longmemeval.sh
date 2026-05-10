export hf_endpoint=https://hf-mirror.com
export cuda_visible_devices=2
python test_longmemeval500.py --parallel-questions \
                              --dataset /mnt/sdb/liuqiaoan/longmemeval-cleaned/longmemeval_s_cleaned.json \
                              --num-samples 500 \