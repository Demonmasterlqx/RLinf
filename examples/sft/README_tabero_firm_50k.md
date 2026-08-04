# Tabero Firm π0 full-LoRA SFT

This example trains both PI0 LoRA subtrees and the TacField TCN on
`datas/tabero_firm`. Training uses FP32 throughout and exports a merged BF16
checkpoint for T2-VLA.

## Entrypoints

- Config: `config/tabero_firm_sft_full_lora_tacfield_fp32_50k.yaml`
- Launcher: `run_tabero_firm_sft_full_lora_tacfield_fp32_50k.sh`

The launcher places actor ranks 0-6 on physical GPUs 0,1,3,4,5,6,7 and refuses
to start unless those seven GPUs are free. GPU 2 is deliberately excluded from
both placement and the launcher's visible-device list. At least 300 GiB must be
available under the results filesystem and W&B online authentication must
succeed. With a micro-batch of one per GPU and four accumulation passes, the
global batch is 28.
It computes Firm-only normalization statistics inside the run directory and
pins their SHA256 for resume.

Run the two-stage 4-step DCP/resume preflight first:

```bash
cd /data/home/sim6g/code/tabero/RLinf
examples/sft/run_tabero_firm_sft_full_lora_tacfield_fp32_50k.sh preflight
```

Start the formal 50,000-step run:

```bash
examples/sft/run_tabero_firm_sft_full_lora_tacfield_fp32_50k.sh formal
```

Resume from a complete DCP checkpoint without changing its run directory or
normalization statistics:

```bash
TABERO_FIRM_RESUME_DIR=/absolute/run/checkpoints/global_step_5000 \
  examples/sft/run_tabero_firm_sft_full_lora_tacfield_fp32_50k.sh resume
```

Checkpoints are saved every 5,000 optimizer steps. The final launcher stage
merges both adapters, retains the TCN, and writes the BF16 export under
`<run>/exports/step_50000`.

## Evaluation contract

The exported model is `SFT full-LoRA TacField`, not piRL. Evaluate it directly
with the T2-VLA checkpoint server and Tabero client, without RLT/DSRL bundle
flags and without `--send-dsrl-raw-image`. Task 0-6 Firm evaluation uses 50
episodes per task, seed 11, prompt seed 0, `firmly tightly`, HDF5 resets, eight
consecutive success steps, 30 inference chunks, and replan horizon 10. Preserve
each task's server log, client log, JSON, and TXT; derive success rate from the
JSON and cross-check all 50 episode starts and the completion marker.
