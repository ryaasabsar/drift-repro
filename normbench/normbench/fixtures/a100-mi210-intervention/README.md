# Frozen A100–MI210 intervention inputs

`bundle.json` and `inputs.npz` contain the captured normalization inputs, source
native/trace outputs, exact shared FP32 rsqrt operands, and precomputed FP32
sigmoid/rsqrt references. They were derived from the supplied complete A100 and
MI210 campaigns; source worker fingerprints and environments are recorded.

All data is numeric, loaded without pickle. No model, credentials, original run
folder, or accelerator environment is bundled. See [the experiment guide](../../../INTERVENTIONS.md).

Bundle fingerprint:
`c9d433b5ea0a07a41b7875dad28ac9044a85a1c26e5ca99b55cbf03e5f0802bd`.
