#!/bin/bash
# Entrypoint for BSA eval custom job on OpenWeights.
# Expects env vars: MODEL_KEY, BASE_MODEL, ADAPTER_HF_ID (optional), EVAL_TASKS
set -e

cd bsa

# Install Python dependencies not in the default OW image
pip install pyyaml tqdm python-dotenv pydantic adjustText seaborn pandas matplotlib

# Patch VLLMClient.served_model() to support VLLM_MODEL_OVERRIDE env var.
# This is needed so LoRA adapter queries go to the adapter, not the base model.
python3 -c "
import pathlib
p = pathlib.Path('src/inference_vllm.py')
code = p.read_text()
old = 'def served_model(self) -> str:'
new = '''def served_model(self) -> str:
        import os
        override = os.environ.get(\"VLLM_MODEL_OVERRIDE\")
        if override:
            return override'''
if old in code and 'VLLM_MODEL_OVERRIDE' not in code:
    code = code.replace(old, new)
    p.write_text(code)
    print('Patched VLLMClient.served_model with VLLM_MODEL_OVERRIDE support')
else:
    print('VLLMClient already patched or patch target not found')
"

# Build vLLM command
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $N_GPUS GPU(s)"
VLLM_CMD="vllm serve $BASE_MODEL --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.9 --quantization fp8"
VLLM_CMD="$VLLM_CMD --tensor-parallel-size $N_GPUS"

if [ -n "$ADAPTER_HF_ID" ]; then
    VLLM_CMD="$VLLM_CMD --enable-lora --max-lora-rank 64"
    VLLM_CMD="$VLLM_CMD --lora-modules $MODEL_KEY=$ADAPTER_HF_ID"
    export VLLM_MODEL_OVERRIDE="$MODEL_KEY"
    echo "Starting vLLM with LoRA adapter: $MODEL_KEY=$ADAPTER_HF_ID"
else
    echo "Starting vLLM with base model: $BASE_MODEL"
fi

# Start vLLM in background
echo "Command: $VLLM_CMD"
$VLLM_CMD > /tmp/vllm.log 2>&1 &
VLLM_PID=$!

# Wait for vLLM to be ready (up to 20 minutes)
echo "Waiting for vLLM to start..."
for i in $(seq 1 1200); do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM process died. Log:"
        tail -100 /tmp/vllm.log
        exit 1
    fi
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM ready after ${i}s"
        # Print available models
        curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print('Available models:', [m['id'] for m in d['data']])"
        break
    fi
    sleep 1
done

# Verify vLLM is actually running
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM failed to start within 1200s. Log:"
    tail -200 /tmp/vllm.log
    exit 1
fi

OUTPUT_DIR="output/non_em_experiment"
mkdir -p "$OUTPUT_DIR"

# Determine condition label for this model
CONDITION="$MODEL_KEY"

# Run eval tasks based on EVAL_TASKS env var (comma-separated)
# Available: sycophancy_behavior, myopia_behavior, self_report, em_check

if echo "$EVAL_TASKS" | grep -q "sycophancy_behavior"; then
    echo "=== Running sycophancy behavior eval ==="
    python3 scripts/non_em/eval_sycophancy_behavior.py \
        --base-url http://localhost:8000/v1 \
        --models "$CONDITION" \
        --output-dir "$OUTPUT_DIR" || echo "WARNING: sycophancy behavior eval failed"
fi

if echo "$EVAL_TASKS" | grep -q "myopia_behavior"; then
    echo "=== Running myopia behavior eval ==="
    python3 scripts/non_em/eval_myopia_behavior.py \
        --base-url http://localhost:8000/v1 \
        --models "$CONDITION" \
        --output-dir "$OUTPUT_DIR" || echo "WARNING: myopia behavior eval failed"
fi

if echo "$EVAL_TASKS" | grep -q "self_report"; then
    echo "=== Running self-report probes ==="
    python3 scripts/non_em/eval_self_report.py \
        --base-url http://localhost:8000/v1 \
        --models "$CONDITION" \
        --output-dir "$OUTPUT_DIR" || echo "WARNING: self-report eval failed"
fi

if echo "$EVAL_TASKS" | grep -q "em_check"; then
    echo "=== Running EM check ==="
    python3 scripts/non_em/eval_em_check.py \
        --base-url http://localhost:8000/v1 \
        --models "$CONDITION" \
        --output-dir "$OUTPUT_DIR" || echo "WARNING: EM check failed"
fi

# Copy results to /uploads so OpenWeights collects them
echo "=== Uploading results ==="
mkdir -p /uploads
cp -r "$OUTPUT_DIR"/* /uploads/ 2>/dev/null || true

echo "=== Done: $MODEL_KEY ==="

# Stop vLLM
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
