# Frozen gated RMSNorm cases

`cases.npz` and `fixture.json` are the portable numeric inputs and float64
references for NormBench. See the [run and comparison guide](../../../README.md).

The captured tensors come from the user's Qwen3.5-0.8B first linear-attention
normalization operation for `gsm8k_256`, originally exported as operation
fingerprint `79ff4efc8bc7d831b7178de5bd3661645addc0f35feba4fb93953624d6eaa5ea`.
They are model-derived numerical activations/weights, not a model checkpoint.
No Hugging Face credentials, responses, or personal data are included.

Synthetic cases use NumPy PCG64 seed 42. The archive deduplicates equal arrays;
`fixture.json` maps logical names to physical arrays and verifies file hashes.
All hosts must use this same fixture, not independently regenerated copies.
