#import "@preview/touying:0.5.5": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Minecraft Mob Detection],
    subtitle: [Real-time multi-class object detection],
    author: [Igor Jakus #h(1em) Jan Pułtorak],
    date: [16.06.2026],
  ),
  config-colors(
    primary: rgb("#3a7d34"),
    primary-light: rgb("#5fa84f"),
    secondary: rgb("#2b2b2b"),
  ),
)

#set text(size: 22pt)

// Results auto-fetched from W&B via: uv run python scripts/fetch_results.py
#let results = json("results.json").runs
#let pct(v) = if v == none { [—] } else { [#calc.round(v * 100, digits: 1)%] }
#let mins(v) = if v == none { [—] } else { [#v min] }

#title-slide()

// ============================================================
== Problem Definition

Given a live gameplay video stream, the model should, *in real time*:

- *Detect* all visible mobs in each frame
- *Classify* each detected mob by type (Creeper, Cow, Pig, Sheep, Chicken)
- *Localize* each detection with a bounding box

// ============================================================
== Formalization — Input & Classes

*1. The input* — each frame is a 3D tensor:
$ I in RR^(H times W times 3) $
where $H$, $W$ are height/width and $3$ are the RGB channels.

#v(0.8em)

*2. The target classes* — the set of mob types:
$ C = {c_0, c_1, c_2, dots, c_K} $
with $K$ mob classes; $c_0$ is the background (no mob).

// ============================================================
== Formalization — Output

*3. The detections* — the model outputs $N$ predictions per frame:
$ hat(Y) = {hat(d)_1, hat(d)_2, dots, hat(d)_N} $

Each detection is a tuple of three components:
$ hat(d)_i = (hat(b)_i, hat(p)_i, hat(s)_i) $
$ hat(b)_i = (hat(x)_i, hat(y)_i, hat(w)_i, hat(h)_i) in RR^4 quad #text(size: 16pt)[(box)] $
$ hat(p)_i = (hat(p)_i^1, dots, hat(p)_i^K) quad #text(size: 16pt)[(class probs)] $
$ hat(s)_i in [0, 1] quad #text(size: 16pt)[(confidence)] $

// ============================================================
== Formalization — Model & Loss

*4. The model function* — parameterized by weights $theta$:
$ f_theta (I) = hat(Y) $

#v(0.8em)

*5. The objective* — a multi-task loss compares $hat(Y)$ to ground truth $Y$:
$ L = lambda_("loc") L_("loc")(b, hat(b)) + lambda_("cls") L_("cls")(c, hat(p)) $

- $L_("loc")$ — bounding-box coordinate error
- $L_("cls")$ — classification error
- $lambda_("loc"), lambda_("cls")$ — weights balancing the two tasks

// ============================================================
== Examples

#align(center)[
  #image("images/sheeps.png", height: 78%)
]

== Examples

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align(center + horizon)[#image("images/example1.png", width: 100%)],
  align(center + horizon)[#image("images/example2.png", width: 100%)],
)

// ============================================================
== Why Minecraft?

- *Controlled visual domain* — consistent block-art style, limited palette, predictable lighting
- *No privacy issues* — fully synthetic data
- *Challenging enough* — scale variation, many classes, varying biomes & lighting
- *Data availability* — existing datasets
- *Fun & engaging* — motivates thorough experimentation

// ============================================================
== Dataset

*Source:* #link("https://universe.roboflow.com/search?q=minecraft")[Minecraft datasets on Roboflow] — community-curated, with labeled bounding boxes.

+ Download from Roboflow; pool all images across original splits
+ Filter to images containing at least one of our 5 target classes
+ Reshuffle (`seed=42`) and split *70% / 15% / 15%* into train / valid / test
+ Training and inference at 640×640 px (`imgsz=640`)

#v(0.6em)
*Classes (5):* pig, chicken, cow, creeper, sheep

// ============================================================
== Approach

*Models* — fine-tuned with Ultralytics on the Minecraft dataset:
- *YOLOv8n* — lightweight single-stage baseline
- *YOLO26m* — modern middle-ground detector
- *RT-DETR-L* (Large) — transformer-based real-time detector
- *RT-DETR-X* (Extra Large) — larger transformer variant

// ============================================================
== Training Infrastructure

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Compute — University HPC*
    - 2× *RTX 3090* (multi-GPU, DDP)
    - Scheduled via *Slurm* (`sbatch`)
  ],
  [
    *Tooling*
    - PyTorch + Ultralytics
    - Tracking: Weights & Biases
  ],
)

// ============================================================
== Real-Time Inference

The trained model runs on *live Minecraft gameplay*, annotating each
frame on the fly:

```bash
python scripts/detect_video.py gameplay.mp4 \
    --model weights/rtdetr-x.pt \
    --conf 0.4 --imgsz 640
```

- Frame-by-frame streaming (memory-efficient generator)
- Bounding boxes + class + confidence drawn per frame

// ============================================================
#focus-slide[
  Demo
]

// ============================================================
== Results

#set text(size: 19pt)
#table(
  columns: (1.2fr, auto, auto, auto, auto, auto),
  inset: 9pt,
  align: (left, center, center, center, center, center),
  stroke: 0.5pt + gray,
  table.header(
    [*Model*], [*mAP\@0.5*], [*mAP\@0.5:0.95*], [*P*], [*R*], [*Train time*],
  ),
  ..results.map(r => (
    [#r.label],
    pct(r.at("map50", default: none)),
    pct(r.at("map5095", default: none)),
    pct(r.at("precision", default: none)),
    pct(r.at("recall", default: none)),
    mins(r.at("runtime_min", default: none)),
  )).flatten()
)

#v(0.4em)
#text(size: 15pt, fill: gray)[
  All runs: 100 epochs, 2× RTX 3090, 640×640 px. P = precision, R = recall.
]

// ============================================================
== Analysis & Discussion

#set text(size: 20pt)

- *RT-DETR-L* — best accuracy (92.2% mAP\@0.5); *YOLO26m* nearly matches it and trains 2× faster — better tradeoff for real-time gameplay
- *YOLOv8n* — smallest model and fastest to train, but lowest recall (81%) — conservative, misses more mobs
- *RT-DETR-X* underperformed despite size — likely smaller batch (8 vs 16) and GPU memory limits
- mAP\@0.5:0.95 (62–69%) lags mAP\@0.5 — *tight bounding boxes* are harder than finding mobs

// ============================================================
== Conclusions

- Built an *end-to-end real-time* Minecraft mob detector (5 classes)
- *RT-DETR-L* was our best model (mAP\@0.5 ≈ 92%); YOLO26m close behind and much faster to train
- Runs live on *gameplay video*

// ============================================================
#focus-slide[
  Thank you! #linebreak()
  Questions?
]
