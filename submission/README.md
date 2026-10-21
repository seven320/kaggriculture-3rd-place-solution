# Final submissions

The two final submissions of team DSM. sub126 is sub125 with melon replanting added as the last stage (write-up §3, §7.1).

## Files

`sub126/` holds every file of the submission except the weights, plus `LICENSE_kaggle_environments_Apache-2.0.txt` (the licence
text that the header of `postprocess_c1p_w9p.py` points to; not part of the submitted archive). `sub125/` holds only `main.py`; all other files of sub125 are the same as in sub126, and sub125 has no `melon_replant.py`.

| File | What it is | Author |
|---|---|---|
| `main.py` | entry point: the network, then `_pace` → `_land` → the post-processing stages (order on its last line) | masspeaks (template, `_pace`), shimishige (`_land`), denden (wiring of the stages) |
| `model_tf.py`, `model_ptr_infer.py`, `obs_feat.py`, `cfg.json` | the network (Transformer, 266 tokens) and its features | masspeaks |
| `stock_target.json` | inventory targets used by `_pace` (quantiles per time × item, computed from 90 public games of another team) | masspeaks |
| `postprocess_c1p_w9p.py`, `postprocess_d23s_p1.py`, `collect_plan_d29.py` | wasted-move fixes and sells, the hour-23 barn sell, the last-day collection plan | denden |
| `melon16.py` | MELON16: buys melon seeds at the start of day 16 | masspeaks |
| `melon_replant.py` | melon replanting on days 16–19 (sub126 only) | masspeaks |
| `policy.pt` | gen70 weights, 41.8 MB — **not in git** (see below) | shimishige (training of gen21–70) |

`main.py` of sub125 and sub126 differ in two lines only: sub126 adds `import melon_replant as _MREP` and wraps the return value of
`agent()` in `_MREP.apply(converted, ...)`.

## Differences from the submitted files

All files are byte-identical to what we submitted, except one comment line in each of two files, where we removed the name of a
private computing environment from a tool path:

| File | sha256 of the submitted file | sha256 here |
|---|---|---|
| `sub126/obs_feat.py` | `8d20f20d36836799c3cf21baeba865ddcbd34cdc305090f64cb655f22c740609` | `7fcab25049e7fc9c1b0839b1ee8d899ca10cbd30fa8819abb4a289d234d2123f` |
| `sub126/model_ptr_infer.py` | `4198f8b9c4807b27385126af7d4deb5a0c85fc6150a280baa5c61725acb64e97` | `ab5c73320748c9038574a6bc2b6db552fb5cca3b28c6a90ada3f6dfbfd43026f` |

The three post-processing modules are byte-identical to the copies in `../postprocess/`.

## Weights

| File | Size | sha256 |
|---|---|---|
| gen70 `policy.pt` | 41.8 MB | `76ec119f8513c8b728fbb0501b2839a0fe4556fdba96bbf6b8bd7e93b007f23c` |

The weights are not committed to this repository. They are in the Kaggle Dataset [kaggriculture-dsm-3rd-place-weights](https://www.kaggle.com/datasets/denden12/kaggriculture-dsm-3rd-place-weights) (`gen70/policy.pt`). After downloading,
put `policy.pt` into `sub126/` (and `sub125/` with the other files) and check it with `sha256sum policy.pt`.

To build a runnable submission folder: copy `sub126/` (for sub125, also replace `main.py` with `sub125/main.py` and delete
`melon_replant.py`), add `policy.pt`, and pack the folder as the Kaggle submission archive with `main.py` at its root:

```bash
cp -r submission/sub126 build_sub126 && cp /path/to/gen70/policy.pt build_sub126/
tar -czf sub126.tar.gz -C build_sub126 .
```
