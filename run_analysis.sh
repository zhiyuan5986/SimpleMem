export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2
model_name="Qwen3-VL-8B-Instruct"
# model_name="minimax-m2.5"
# model_name="qwen3vl"
base_url="https://models.sjtu.edu.cn/api/v1"
base_url="http://localhost:8000/v1"
api_key="sk-Of0btqIp3GW0Jv65PhkqbQ" # openclaw
api_key="sk-tYfudtgpHIqCKpxWisXgtA" # sjtu

python scripts/analyze_locomo_supporting_entries.py --model $model_name \
                                                    --base-url $base_url \
                                                    --api-key $api_key \
                                                    --resume-per-sample \
                                                    --entries-dir "./$model_name" \
                                                    --output-json "outputs/$model_name/locomo_support_analysis.json" \
                                                    --disable-reflection \
                                                    --disable-planning

