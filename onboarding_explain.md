# FabricFlow — Onboarding & User Guide

Authoritative reference for FabricAI. Quote sections by §number when citing.

---

## §1. What FabricFlow is

FabricFlow is an AI-assisted system for fabric professionals. It does two
things in one workflow:

1. **Fabric Recognition** — classify a fabric type from one to three
   photographs of the material.
2. **Digital Fabric Passport (DFP / DPP)** — compile supporting
   documentation across five chapters and produce a single passport JSON
   compliant with the forthcoming EU Digital Product Passport (DPP)
   regulation for textiles.

A third module, **FabricAI**, sits over the top as the chat assistant that
explains results, the technology, fabric knowledge, and supply-chain
implications. FabricAI is grounded — it answers only from the CONTEXT
provided to it (this onboarding document plus recent recognition results).

The project is a research collaboration between the **Royal College of Art
(RCA)**, **AiDLab** (Artificial Intelligence in Design Lab), and **The Hong
Kong Polytechnic University**. Project lead: Prof. Elif Yenigun. Domain
partner for DPP requirements: Felicity (supply-chain / sustainability).
Demo cadence: built for RCA exhibitions and trade-show tablet kiosks.

---

## §2. Fabric Recognition — the two-stage cascade

### §2.1 Stage architecture

Recognition is a two-stage cascade, not a flat classifier:

- **Stage 1** decides the high-level construction class:
  KNIT vs WOVEN vs OTHERS.
- **Stage 2** is one of two specialised heads — a knit head with eleven
  sub-categories, or a woven head with eight sub-categories — selected by
  the Stage-1 output.

Why cascaded: knit and woven differ in *construction* first and in
*pattern* second. A flat 20-way classifier confuses subtle knits like Purl
Knit with similar-textured wovens. The cascade respects how a human textile
expert reads a swatch — first the family, then the structure inside it.

Current production model is ConvNeXt (small for Stage 1, tiny for the two
Stage-2 heads). Earlier deployments used ResNet-50; the upgrade reduced
confusion on visually similar pairs like Rib Knit / Jersey and Twill /
Plain.

### §2.2 The 20 fabric classes

WOVEN (8 classes): Twill, Plain Weave, Satin, Dobby, Jacquard, Velvet,
Crepe, Ribbed Poplin.

KNIT (11 classes): Jersey, Rib Knit, Interlock, Tricot, Raschel, Double
Jersey, Cable Knit, Purl Knit, Intarsia, Basket Hopsack, Leno Gauze.

OTHERS (1 class): residual bucket for nonwovens, films, technical
laminates, and any fabric that doesn't fit knit/woven taxonomies.

### §2.3 Photo input — three slots

1. **Required** — fabric surface, flat-lay close-up. This is what the
   classifier reads first.
2. **Recommended** — magnified view (same fabric, zoomed). Resolves
   ambiguity for similar surface patterns (twill vs satin, jersey vs
   interlock).
3. **Optional** — reverse side. Useful for two-faced constructions
   (double cloth, technical knits with different face/back).

Image rules: JPEG or PNG, minimum 224 × 224 pixels, flat-lay, good even
lighting, no watermarks, no duplicates, no garment-finish photography
(buttons, hems, labels). The classifier was trained on close-up texture
crops, not styled product shots.

### §2.4 Output fields

A recognition result has the following fields:

- `fabric_id` — short hex identifier for the run.
- `fullName` — full label, e.g. "1x1 Rib Knit".
- `l1` — Stage-1 class (KNIT / WOVEN / OTHERS).
- `l2` — Stage-2 sub-category (e.g. "Rib Knit").
- `confidence` — float 0–1, the Stage-2 posterior for the chosen class.
- `quality` — image-quality score 0–100. Below 50 the system refuses to
  classify and asks for a re-shoot.
- `consensus` — when multiple photos are supplied, an agreement score
  across the per-photo posteriors. High consensus = all photos point to
  the same class; low consensus = a photo disagrees, typically the cause
  is a poor reverse-side shot or a different fabric crept into one image.

### §2.5 Reading the confidence band

- ≥ 0.85 → **Reliable**: result confirmed, the platform vouches for it.
- 0.65 – 0.85 → **Moderate**: usable, but a sourcing manager should
  cross-check fibre composition before specifying.
- < 0.65 → **Uncertain**: do not specify from this alone. Re-shoot or
  send for human review.

The horizontal band displayed beneath the confidence number maps to these
thresholds.

### §2.6 Why not 100% confident?

A perfectly-trained model still does not give 100% on real photographs
because:

- The class boundaries are continuous (Jersey and Interlock differ by
  stitch count, not by hard rule).
- Lighting changes surface contrast.
- Some fabrics are *blends of constructions* — e.g. a brushed Jersey can
  read as Velour at low resolution.
- Stage-1 confidence above 0.95 with Stage-2 below 0.70 means "we are
  sure it's knit, less sure exactly which knit".

A confidence well below the reliable band is itself useful information —
treat it as a signal to ask for more photos or human review, not a
failure.

---

## §3. Digital Fabric Passport (DFP)

### §3.1 What a DPP is

The EU Digital Product Passport (DPP) regulation will require textile
products in the EU market to carry a structured, verifiable record of
their composition, origin, environmental footprint, and circularity. The
FabricFlow DPP is a *pre-passport* — it captures the fabric-level evidence
that a brand later combines with garment-level data to mint the final
product passport.

### §3.2 Five chapters

Documentation is collected across exactly five chapters. They map onto the
EU DPP information categories.

1. **Profile & Composition** — what the fabric is.
   - 01a TechSpec (construction, weight, width)
   - 01b MaterialList (fibre breakdown, recycled content)
   - 01c ChemicalSubstances (SVHC declaration, dyes, finishes)

2. **Product Journey** — where it came from.
   - 02a Origin (country, mill, batch)
   - 02b ChainOfCustody (Tier 1 → Tier 4 sourcing tiers)
   - 02c ProcessSteps (spin → knit/weave → dye → finish)
   - 06a–06c SupplierKPIs (OTD, AQL, audit grade, capacity)

3. **Care & Circularity** — how it lives and how it ends.
   - 03a CareInstructions (wash / iron / dry-clean ceilings)
   - 03b EndOfLife (technical, biological, mixed loops)
   - 03c DurabilityTests (ISO test pass/fail across 8 axes)

4. **Certifications & Compliance** — what it is verified against.
   - 04a Certifications (GOTS, OEKO-TEX, GRS, RWS, etc.)
   - 04b TestReports (lab report references)
   - 04c Compliance (REACH SVHC, CPSIA, etc.)

5. **Impact** — what it costs the planet.
   - 05a CarbonFootprint (LCA across stages)
   - 05b WaterUsage (litres per kg, recovery rate)
   - 05c Energy (renewable share)

Cluster 06 — supplier KPIs — is filed under Chapter 02 Product Journey,
not Chapter 05 Impact. Supplier performance is part of the *journey* the
fabric took, not its environmental impact.

### §3.3 Filename convention

Bulk uploads are auto-routed by filename suffix. Files matching
`*_NN<letter>_*.{pdf,docx,jpg,png}` route by cluster number `NN` (1–6) and
letter (a / b / c) into the correct chapter sub-section. Example:
`RibKnit_01a_TechSpec.pdf` routes to Chapter 01 Profile & Composition →
sub-section 'a' Technical Specification.

### §3.4 The two reference specimens

The home page exposes two real specimens that demo users can click to
preload the full passport bundle:

- **Rib Knit** — KNIT, 18 documents (3 per chapter × 5 chapters + 3
  supplier KPI docs), built from a Vietnam mill (Hanoi Knit Mills) sourcing
  organic cotton + Elastane (95% / 5%).
- **Twill Weave** — WOVEN, 18 documents, built from a Turkey mill sourcing
  rPET (recycled polyester) + organic cotton.

Both bundles are intentionally complete and clean — they exist to show
the workflow when documentation is fully in order. Real-world bundles are
usually 60–80% complete; the gating logic below handles missing fields
gracefully.

### §3.5 Compile Passport — what happens

When the user clicks **Compile Passport**, the front-end POSTs all
collected files to `/api/build_passport_v2`. The backend parses each file
with a document-type-specific parser (18 parsers in total, one per
sub-section), then assembles a single passport JSON with ~200 fields
mapped onto the canonical Felicity shell.

The compiled passport renders into the canonical v2 shell
(`web/passport_v2.html`), with two views: a **Preview** card (gauge,
origin, supplier, key tags) and the **Full Passport** (six tab panels —
Profile, Journey, Care, Certifications, Impact, Supplier).

---

## §4. Scoring — fabric score & supplier grade

### §4.1 The gate

A passport only gets a numerical fabric/supplier score if four gate
conditions are all met:

1. `compositionTotal == 100` — fibre percentages sum to exactly 100.
2. `svhcPassed == true` — Chemical Substances declaration confirms no
   Substances of Very High Concern above threshold.
3. `originDisclosed == true` — country and mill are stated, not redacted.
4. `hasTestReport == true` — at least one third-party test report is
   attached.

If any of the four fails, the score reads "Gate not met — N/4 conditions
satisfied" and the gauge shows a hairline outline instead of a value. The
shell engine refuses to invent a number from partial evidence.

### §4.2 Fabric score — five clusters (max 100)

When the gate is met, the fabric score sums five clusters:

- **C1 Material & Origin** (max 25) — fibre composition score weighted by
  recycled / organic / virgin shares; origin transparency bonus.
- **C2 Chemistry & Finish** (max 20) — SVHC pass; ZDHC MRSL conformance;
  finishing chemicals declaration completeness.
- **C3 Durability** (max 20) — pass count across 8 ISO durability tests
  (pilling, dimensional stability, colour-fastness, etc.).
- **C4 Care & End-of-Life** (max 20) — care ceiling efficiency + whether
  an EOL loop is declared (technical / biological / mixed).
- **C5 Impact** (max 15) — carbon intensity vs industry average; water
  recovery rate; renewable energy share.

Bands: A ≥ 85, B 75–84, C 65–74, D 55–64, F < 55.

### §4.3 Supplier grade — three pillars (max 100)

- **sA Reliability** (max 35) — On-Time Delivery rate, AQL pass rate.
- **sB Compliance** (max 35) — Social audit grade (SA8000 / BSCI / SLCP);
  audit recency.
- **sC Sustainability** (max 30) — Wastewater recovery, renewable energy,
  ZDHC conformance.

Same band thresholds as fabric score (A / B / C / D / F).

### §4.4 The override

If the per-document supplier KPI files state explicit letter grades, the
shell aggregates those into a weighted average and *overrides* the engine's
computed grade. This is intentional: where the supplier has already
self-certified or been independently audited to a letter grade, the
document-stated grade takes precedence over the engine's heuristic sum.

For the two reference specimens:
- Rib Knit (Hanoi Knit Mills) → **B+** overall.
- Twill Weave → **A** overall.

These match Felicity's expected outputs in the scoring matrix.

---

## §5. Recognition workflow on the demo

1. **Home / §01 Recognition** — the user can either:
   - Drop their own photo into the first slot, or
   - Click one of the two reference specimens (§03 at the bottom of the
     home), which preloads two photos into slots 1 and 2 *and* preloads
     the 18 passport PDFs into the correct chapters of §02. **Files are
     preloaded only — the user must click Analyse and Compile Passport
     themselves.** No automatic analysis runs on specimen click.

2. **Click Analyse** — the Recognition card POSTs to `/api/predict`. On
   success the result card appears (fabric name, confidence, construction,
   sub-category) and the previously-locked §02 Digital Fabric Passport
   section unlocks. The FabricAI assistant FAB also becomes visible at
   this point — recognition is the access gate.

3. **§02 reveals** — the five-chapter dashboard shows all chapter cards
   with their current completion badges. If the user used a specimen, all
   five chapters already show "Complete · N docs".

4. **Click Compile Passport** — the Passport Preview card mounts in an
   iframe below. The card shows the fabric score gauge, origin, supplier
   short-name, supplier grade, on-time delivery, fibre tags, certification
   tags. A View Full Passport button opens the full six-tab passport in a
   new window.

5. **Iterate** — the user can change documents in any chapter and click
   Compile again to rebuild the passport.

---

## §6. FabricAI — how the assistant grounds itself

### §6.1 The contract

Every FabricAI answer is built from three sources only:

- This onboarding document (§1–§7).
- The most recent Fabric Recognition results, exposed by the host server
  at `/api/results/recent`.
- The current page context — when the assistant drawer is open, the
  widget prepends a `CURRENT PAGE CONTEXT` block describing the fabric
  result on screen and the compiled passport fields. This is treated as
  ground truth for the current turn.

There are no tools, no live browsing, no other knowledge channels. The
strict-source rule (in the system prompt) instructs the model to reply
"I don't have that information in my Fabric Recognition knowledge base."
when the answer is not derivable from these three sources.

### §6.2 Topic scopes

Four topic scopes shape what the model is permitted to draw on:

- **Explain Results** — CONTEXT-only. If the result is not in CONTEXT,
  say so.
- **Fabric Knowledge** — CONTEXT plus general textile knowledge allowed.
- **Explain Technology** — CONTEXT-only. Do not speculate about model
  architecture, training data, or implementation details that aren't in
  this onboarding doc.
- **Supply Chain** — CONTEXT plus general supply-chain knowledge allowed.

### §6.3 Role lenses

Five role lenses (Commercial, Retail, Sourcing, Operations,
Sustainability) re-weight which dimensions of the answer to emphasise.
The voice stays neutral — role only changes content priorities. Each
lens ends the answer with a one-line role-flavoured decision frame
(commercial: keep/shelve; retail: front-of-rack/pass; sourcing: next-step
TODO; operations: go/no-go + tolerance to watch; sustainability:
compliance status + missing evidence).

### §6.4 Languages

Six languages are supported (English, 中文, Türkçe, Italiano, Español,
plus a free-form Other). When a non-English language is selected, the
model replies in that language but keeps source-tag names in English so
citations remain machine-parseable.

### §6.5 Citations

Every answer ends with a `Sources:` line. Allowed source tags:

- `Onboarding §X.Y` — for sections of this document.
- `Result <fabric_id>` — for a specific recent result.
- `Page context` — for the in-page passport / recognition snapshot
  injected by the assistant drawer.
- `General textile knowledge` — only when Topic Scope permits.
- `General supply-chain knowledge` — only when Topic Scope permits.

If no source applies, write `Sources: none`.

---

## §7. Privacy & data handling

Photos uploaded for recognition and documents uploaded for the passport
are not retained server-side beyond the lifetime of the user's session.
The FabricAI chat history is stored per-session as an append-only JSONL
log under `backend/data/sessions/` and is deleted when the session is
reset. No data is sent to third parties other than the OpenAI Responses
API call that produces the assistant's reply text.

Never reveal: API endpoint URLs, file paths, environment variables,
credentials, model architectures, training data sources, or any
information that is not in this onboarding document. If asked, decline
politely.

---

*End of onboarding document.*
