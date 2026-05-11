---
title: FabricFlow Demo
emoji: 🧶
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI-powered fabric classification + Digital Fabric Passport
---

# FabricFlow Demo

Two-stage ConvNeXt fabric classifier (KNIT / WOVEN / OTHERS → 13 subcategories)
with editorial Digital Fabric Passport generator and FabricAI assistant.

- **Recognition:** Stage 1 (3-class) → Stage 2 (KNIT 6-class / WOVEN 7-class)
- **Passport:** PDF + image bundle → structured passport with score gauge,
  fibre composition, supplier provenance.
- **Assistant:** `/assistant` — OpenAI-backed grounded chat over the
  on-page recognition context and an onboarding knowledge doc.

Set the `OPENAI_API_KEY` Space secret to enable FabricAI. Recognition works
without it.

Public entry points:

- `/`             — main demo (recognition → passport)
- `/tablet`       — tablet kiosk view
- `/passport_v2`  — Digital Fabric Passport
- `/assistant`    — standalone FabricAI chat
