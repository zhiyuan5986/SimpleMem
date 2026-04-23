export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2
model_name="Qwen3-VL-8B-Instruct"
# model_name="minimax-m2.5"
model_name="qwen3vl"
base_url="https://models.sjtu.edu.cn/api/v1"
# base_url="http://localhost:8000/v1"
api_key="sk-Of0btqIp3GW0Jv65PhkqbQ" # openclaw
api_key="sk-tYfudtgpHIqCKpxWisXgtA" # sjtu

python -m scripts.compare_locomo_support_answering --answer-model $model_name \
                                                    --answer-base-url $base_url \
                                                    --answer-api-key $api_key \
                                                    --judge-model $model_name \
                                                    --judge-base-url $base_url \
                                                    --judge-api-key $api_key \
                                                    --resume-per-sample \
                                                    --analysis-json "outputs/$model_name/locomo_support_analysis.json" \
                                                    --output-json "outputs/$model_name/locomo_support_answer_compare.json"
