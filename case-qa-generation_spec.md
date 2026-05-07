---
name: case-qa-generation
description: Generate Q&A pairs from a filled case JSON using disease templates
argument-hint: "cases/e1013"
---

# Case Q&A Generation from Finalized Case JSON

Generate concrete Q&A pairs from a finalized case JSON: **$ARGUMENTS**

---

## Objective

Given a filled case JSON (saved from the Endo EHR app) and the corresponding disease YAML template(s) from `benchmark/templates/`, generate the complete set of Q&A pairs with correct answers for this video case. Output a JSON file ready for benchmark evaluation, with per-question `answer_schema` for structured output / `response_format` use with any LLM provider.

---

## Step-by-Step Process

### Step 1: Parse the Case JSON

Resolve the input path from `$ARGUMENTS`:
- If a **folder** path (e.g., `cases/c1003`), look for `{folder_name}_case_report.json` inside it
- If a **JSON file** path (e.g., `cases/c1003/c1003_case_report.json`), use directly

Extract:

```
case_id      = derived from filename: strip "_case_report.json" suffix (e.g., "c1003")
video        = MetaData.video
procedure    = MetaData.procedureType
seg_frame    = MetaData.segmentationFrame
report       = report  (the main report object)
```

The `MetaData` block contains: `procedureType`, `Age`, `Gender`, `video`, `segmentationFrame`.

### Step 2: Enumerate All Diseases in the Case

Walk the report structure. The structure differs by procedure type:

**Colonoscopy** :
```python
# report = { "Polyp": { "sections": {...}, "sublocations": [], ... }, ... }
diseases_found = []
for disease_name in report:
    diseases_found.append({
        "disease": disease_name,
        "location": None,
        "sublocations": report[disease_name].get("sublocations", []),
        "sections": report[disease_name]["sections"],
        "segmentationFrame": report[disease_name].get("segmentationFrame")
    })
```

**Endoscopy** :
```python
# report = { "Stomach": { "diseases": { "Erosion": {...}, ... } }, ... }
diseases_found = []
for location in report:
    for disease_name in report[location].get("diseases", {}):
        disease_data = report[location]["diseases"][disease_name]
        diseases_found.append({
            "disease": disease_name,
            "location": location,
            "sublocations": disease_data.get("sublocations", []),
            "sections": disease_data["sections"],
            "segmentationFrame": disease_data.get("segmentationFrame")
        })
```

**Multi-location diseases** : If the same disease appears in multiple locations, group them:
```python
# e.g., Erosion in Stomach AND Duodenum → one entry with multiple locations
grouped = {}
for entry in diseases_found:
    key = entry["disease"]
    if key not in grouped:
        grouped[key] = {"disease": key, "locations": []}
    grouped[key]["locations"].append({
        "location": entry["location"],
        "sublocations": entry["sublocations"],
        "sections": entry["sections"]
    })
```

### Step 3: Load Templates

For each disease found, load the matching template from `benchmark/templates/`:

1. **Primary match**: Convert disease name to slug (lowercase, spaces→underscores, special chars removed) and look for `benchmark/templates/{slug}.yaml`
2. **Fallback match**: If not found, scan all YAML templates in `benchmark/templates/` for one whose `csv_disease_name` field matches the JSON disease name exactly. Some templates use a different display name than the CSV (e.g., template `disease: "Barrett's esophagus"` has `csv_disease_name: "Suspected Barrett's esophagus"`).
3. If neither match found, print warning: `WARNING: No template for "{disease_name}" — skipping`

**When a template is matched via `csv_disease_name`**: Use the template's `disease` field (not the JSON disease name) in all generated question stems. This ensures stems say "Barrett's esophagus" instead of "Suspected Barrett's esophagus". Print: `NOTE: "{json_disease_name}" matched via csv_disease_name → using template "{template_disease_name}" ({slug}.yaml)`

### Step 4: Generate Disease Identification Q&A

Disease identification uses a **two-phase structure**: all MCQs first (one per disease), then all binaries (deduplicated across diseases). This ensures harder discrimination questions come first while the LLM's first impression is fresh, and avoids asking the same distractor binary question twice when diseases share distractors.

#### Phase A: DI MCQs (Turns 1..N, one per disease)

Process diseases in order. For each disease:

1. Read `distractors` from the template
2. If distractors available (non-empty list):
   - Pick 2 random distractors from the 4 available
   - Build options: `["Normal", correct_disease, distractor_1, distractor_2]`
   - Shuffle positions 2-4 randomly (position 1 = "Normal" ALWAYS)
   - Assign letters A, B, C, D
   - Record correct answer letter
   - Track which distractors were used in this MCQ (`mcq_picks`)
3. If distractors empty:
   - Output warning: `# WARNING: No distractors available for {disease}. Fill DiseaseComps.csv.`
   - Use placeholder: `["Normal", correct_disease, "TO_BE_FILLED_1", "TO_BE_FILLED_2"]`

**Stem for first disease (Turn 1, video attached):**
```
"Given this {procedure} video clip, what condition would an expert endoscopist suspect among the given options?"
```

**Stem for 2nd+ disease in multi-disease cases (Turn 2+):**
```
"This {procedure} video clip also shows another condition. What additional finding would an expert endoscopist suspect among the given options?"
```

**Single-disease case:** 1 MCQ at Turn 1.
**Multi-disease case:** N MCQs at Turns 1..N.

#### Phase B: DI Binaries (Turns N+1..M, deduplicated, shuffled)

After all MCQs, generate binary questions:

1. **Positive binaries** — One per disease (answer = "Yes"):
   ```json
   {
     "phase": "disease_identification",
     "disease": "{disease_name}",
     "task": "DI",
     "subtask": "DI-C",
     "type": "Binary",
     "stem": "Given this {procedure} video clip, would an expert endoscopist suspect {disease_name}?",
     "options": ["Yes", "No"],
     "correct": "Yes"
   }
   ```

2. **Negative binaries** — Deduplicated across all diseases (answer = "No"):
   - For each disease, compute remaining distractors: `disease.distractors - disease.mcq_picks`
   - Pool all remaining distractors from all diseases into a single set (union)
   - **Deduplication**: Each unique distractor name generates exactly ONE negative binary, regardless of how many diseases share it
   - Generate one binary per unique distractor:
   ```json
   {
     "phase": "disease_identification",
     "task": "DI",
     "subtask": "DI-R",
     "type": "Binary",
     "stem": "Given this {procedure} video clip, would an expert endoscopist suspect {distractor}?",
     "options": ["Yes", "No"],
     "correct": "No"
   }
   ```

3. **Shuffle** all binaries (positive + negative) into random order and assign sequential turns starting at N+1.

**Distractor notes (`distractor_notes` field):** Some templates include a `distractor_notes` map (e.g., `distractor_notes: {Diverticulum: "(Note: Do not consider ...)"}`). When generating DI questions, check if the selected distractor has a note in this map. If so, append the note to the stem of any MCQ or Binary question where that distractor appears as an option. For the diverticulum template itself, the note is already baked into the MCQ and binary_positive stems.

**Example (3-disease case with overlapping distractors):**
```
PHG distractors:     [Caustic Injury, Dieulafoy Lesion, GOO, Fistula]
GP distractors:      [Caustic Injury, GOO, Linitis Plastica, Fistula]
Angio distractors:   [Dieulafoy Lesion, Diverticulum, Fistula, Xanthoma]

MCQ picks:
  PHG MCQ uses:    Dieulafoy Lesion, Fistula    → remaining: Caustic Injury, GOO
  GP MCQ uses:     Linitis Plastica, GOO         → remaining: Caustic Injury, Fistula
  Angio MCQ uses:  Xanthoma, Diverticulum        → remaining: Dieulafoy Lesion, Fistula

Negative binary pool (union of remaining): {Caustic Injury, GOO, Fistula, Dieulafoy Lesion}
  → 4 unique negative binaries (not 6, because deduplication removes repeats)

Total DI binaries: 3 positive + 4 negative = 7
Total DI questions: 3 MCQs + 7 binaries = 10 (vs 12 without dedup)
```

**Single-disease case:** 1 MCQ + 1 positive binary + 2 negative binaries = 4 DI questions (unchanged).

### Step 4b: Generate Non-Disease Finding Q&A

Some entries in the case JSON are **not diseases** but anatomical assessments. They have `disease_identification.applicable: false` in their templates. Currently these are:

- **Hills hiatus** — GE Junction valve flap grading (Grades I–IV). 1 question.
- **Ampulla of Vater / Papilla** — Major papilla assessment (visualization, type, appearance, discharge, biopsy, impression). Multiple questions.
- **Mucosa** — Mucosal descriptor findings (erythematous, edematous, granular, etc.). Variable number of questions depending on which descriptors are filled. **Location-dependent** — see rule 3b below.
- **Normal** — Normal findings at a location. **Special rule:** Only generate if the case has NOTHING else — no diseases AND no other non-disease findings (see rule 7 below). 5 questions. **Location-dependent** — see rule 3b.

**Handling rules:**

1. **No disease identification** — Do NOT generate disease ID questions for these findings.

2. **Turn ordering** — Non-disease finding questions are asked AFTER all disease identification questions but BEFORE the disease reveal and any location/sublocation/detail questions. The `disease_reveal_prefix` moves to the first location/sublocation/detail question after these.

   Example turn sequence:
   - Turn 1: Disease ID MCQ (video attached)
   - Turn 2: Disease ID Binary
   - Turn 3+: **Non-disease finding questions** (no disease reveal — LLM doesn't know the disease yet)
   - Next turn: `disease_reveal_prefix` + first location/sublocation/detail question

3. **Generate questions** from the template's `detail_questions`, using `phase: "non_disease_finding"`. Process conditionals, extract answers, and apply skip rules exactly as in Step 7 (Detail Q&A). Example for Hills hiatus:

```json
{
  "question_number": 3,
  "turn": 3,
  "phase": "non_disease_finding",
  "disease": "Hills hiatus",
  "type": "MCQ",
  "multi_select": false,
  "stem": "How would an expert endoscopist grade the Hill's valve flap at the GE Junction?",
  "options": {"A": "I", "B": "II", "C": "III", "D": "IV"},
  "correct": "{correct_letter}",
  "answer_schema": {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
    "required": ["answer"],
    "additionalProperties": false
  }
}
```

   **3b. Location-dependent non-disease findings (Mucosa):**

   When a template has `location_in_stem: true`, the question stems contain `{location}` placeholders. Step 2 must:

   1. Read the location from the JSON key where the finding appears (e.g., `report["Stomach"].diseases["Mucosa"]`)
   2. If sublocations are available in the JSON, format as `{location} ({sublocation})` (e.g., "Stomach (Antrum - Lesser Curvature)")
   3. Replace `{location}` in the stem with the resolved location string
   4. Only generate questions for sections that are filled in the JSON (presence of a section = the descriptor is present)

   **Value translation:** Some Mucosa sections use "Yes" as a JSON value that should be translated to a descriptive option in the question. Check the template's `note` field — e.g., for Scalloping, JSON "Yes" → display "Diffuse"; for Pathological vascular pattern, JSON "Yes" → display "Localized".

   Example Mucosa question:
   ```json
   {
     "question_number": 4,
     "turn": 4,
     "phase": "non_disease_finding",
     "disease": "Mucosa",
     "location": "Stomach",
     "type": "MCQ",
     "stem": "Regarding the mucosa in the Stomach (Antrum - Lesser Curvature), what is the distribution of the erythematous changes?",
     "options": {"A": "Localised", "B": "Patchy", "C": "Focal", "D": "Generalised"},
     "correct": "A"
   }
   ```

4. **Extract answers from JSON** — Use the `json_path` from the template. The path depends on procedure type:

   **Endoscopy** (location-wrapped):
   - Hills hiatus: `report["GE Junction"].diseases["Hills hiatus"].sections["Hills Grade"].attrs`
   - Ampulla of Vater / Papilla: `report["Duodenum"].diseases["Ampulla of Vater / Papilla"].sections[...].attrs`
   - Mucosa: `report["{location}"].diseases["Mucosa"].sections["{Section}"].attrs`

   **Colonoscopy**:
   - `report["{finding_name}"].sections["{Section}"].attrs` directly

5. **Edge case — non-disease finding only** — If the case has no actual diseases (only non-disease findings), generate just the finding questions starting at Turn 1 with the video attached. No disease reveal prefix needed.

6. **Multiple non-disease findings** — If multiple are present, generate in this order: Hills hiatus → Ampulla of Vater / Papilla → Mucosa. All before the disease reveal.

7. **Normal — special handling:**

   Normal is ONLY generated when the case JSON contains **nothing else** — no diseases AND no other non-disease findings (no Hills hiatus, no Ampulla, no Mucosa). If anything else is present, skip Normal entirely.

   Normal produces exactly 5 questions with the location revealed in the stem (no sublocation). The location is read from the JSON key where Normal appears (e.g., `report["{location}"].diseases["Normal"]`).

   **MCQ** (disease identification with location):
   ```json
   {
     "turn": 1,
     "phase": "non_disease_finding",
     "disease": "Normal",
     "location": "{location}",
     "type": "MCQ",
     "stem": "Given this {procedure_type} video clip showing the {location}, what condition would an expert endoscopist suspect among the given options?",
     "options": {"A": "Polyp", "B": "Normal", "C": "Ulcer", "D": "Angioectasia"},
     "correct": "B"
   }
   ```
   Fixed 4 options: Normal, Angioectasia, Polyp, Ulcer. Randomize order, assign letters.

   **4 binaries** (shuffled into random order for Turns 2–5):
   - Normal/Abnormal binary (correct = "Normal")
   - Angioectasia binary (correct = "No")
   - Polyp binary (correct = "No")
   - Ulcer binary (correct = "No")

   Shuffle all 4 binaries randomly and assign to turns 2–5.

   **Esophagus note:** When the location is Esophagus, append the note `"(Note: Consider Glycogen nodules if present as normal)"` to the MCQ and Normal/Abnormal binary stems only. Do NOT add it to the distractor binaries (Angioectasia, Polyp, Ulcer).

   **Turn numbering:** MCQ is always Turn 1 (video attached). 4 binaries in random order at Turns 2–5. 5 turns total. No `disease_reveal_prefix` is needed since there is no disease to reveal.

### Step 5: Generate Location Q&A (Endoscopy Only)

**Skip if**: template says `location.applicable: false`

Determine the correct organ:
- Map JSON location key to organ: "Esophagus"→"Esophagus", "GE Junction"→"Esophagus" (merged; sublocation = "Lower"), "Stomach"→"Stomach", "Duodenum"→"Duodenum"
- When disease is in GE Junction: location = "Esophagus", sublocation part = "Lower" (treat as lower esophagus)
- If disease appears in multiple locations → check if they map to different organs
  - Same organ (e.g., two stomach locations): answer = that organ
  - Different organs: answer = "More than one organ"

```json
{
  "phase": "location",
  "disease": "{disease_name}",
  "type": "MCQ",
  "multi_select": false,
  "stem": "In which organ is this {disease_name} present? (Note: GE Junction is considered as Lower Esophagus)",
  "options": {"A": "Esophagus", "B": "Stomach", "C": "Duodenum", "D": "More than one organ"},
  "correct": "{correct_letter}",
  "answer_schema": {
    "type": "object",
    "properties": {
      "answer": {"type": "string", "enum": ["A", "B", "C", "D"]}
    },
    "required": ["answer"],
    "additionalProperties": false
  }
}
```
Note: Options are shuffled — randomly assign letters. "correct" records the letter of the right answer.

### Step 6: Generate Sublocation Q&A

After location is revealed, generate sublocation questions for each organ where the disease is found.

#### Extract Sublocation from JSON

The `sublocations` array in the JSON stores:
- Esophagus: `["Lower"]`, `["Upper"]`, etc.
- Stomach (matrix): `["Antrum", "Antrum - Lesser Curvature"]` — region name and "Region - Wall" combinations
- Duodenum (matrix): `["D1 Bulb", "D1 Bulb - Anterior wall"]`

**Parse Stomach/Duodenum sublocations:**
```python
# Separate part from wall
parts = []
walls = {}
for sub in sublocations:
    if " - " in sub:
        part, wall = sub.split(" - ", 1)
        walls.setdefault(part, []).append(wall)
    else:
        parts.append(sub)
# parts = ["Antrum"], walls = {"Antrum": ["Lesser Curvature"]}
```

#### Generate Part Question

Use options from the template's sublocation section for the relevant organ.

```json
{
  "phase": "sublocation_part",
  "disease": "{disease_name}",
  "organ": "Stomach",
  "type": "MCQ",
  "multi_select": false,
  "stem": "The {disease_name} is present in the Stomach. In which part of the stomach is it located?",
  "options": {"A": "Fundus", "B": "Antrum", "C": "Lower Body", "D": "Incisura"},
  "correct": "B",
  "answer_schema": {
    "type": "object",
    "properties": {
      "answer": {"type": "string", "enum": ["A", "B", "C", "D"]}
    },
    "required": ["answer"],
    "additionalProperties": false
  }
}
```
Note: Options from template, shuffled. Correct letter matches JSON sublocation. If the JSON has multiple parts selected (rare), list all as correct.

#### Generate Wall Question (Stomach/Duodenum Only)

Only generate if:
- Part is a non-heading region (Antrum through Fundus for Stomach; D1 Bulb or D2 for Duodenum)
- JSON has a wall value for that part

```json
{
  "phase": "sublocation_wall",
  "disease": "{disease_name}",
  "organ": "Stomach",
  "part": "Antrum",
  "type": "MCQ",
  "multi_select": false,
  "stem": "The {disease_name} is in the Antrum. On which wall is it located?",
  "options": {"A": "Greater Curvature", "B": "Lesser Curvature", "C": "Posterior Wall", "D": "Anterior Wall"},
  "correct": "B",
  "answer_schema": {
    "type": "object",
    "properties": {
      "answer": {"type": "string", "enum": ["A", "B", "C", "D"]}
    },
    "required": ["answer"],
    "additionalProperties": false
  }
}
```

#### Multi-Organ Flow

For diseases in multiple organs, generate sublocation questions for EACH organ:

```json
[
  {"phase": "sublocation_part", "stem": "The {disease} is present in the Stomach. In which part..."},
  {"phase": "sublocation_part", "stem": "The {disease} is also present in the Duodenum. In which part..."}
]
```

### Step 7: Generate Detail Q&A

For each question in the template's `detail_questions`:

#### 7a. Check Conditional

If the question has a `conditional` field, evaluate it against the JSON:

**Parse the raw conditional string:**
- `Section(Name=Value)` → Check if `sections[Section].attrs[Value]` exists and is `true`, OR if `sections[Section].subsections[Name].attrs[Value]` exists and is `true`
- `Subsection(Name=Value)` → Within the parent section, check if `subsections[Name].attrs[Value]` is `true`
- Support `AND` and `OR` operators between conditions

**If condition evaluates to FALSE → SKIP this question entirely.**

**If condition evaluates to TRUE → include the question, AND add an `answer_reveal_prefix` field.** This prefix reveals the correct answer to the parent question so the LLM has context for the conditional follow-up. Format:

`"The {section/subsection} is {answer_value}."`

For example, if Count is conditional on `Section(Number=Multiple)` and Number's answer is "Multiple":
- `"answer_reveal_prefix": "The number of lesions is Multiple."`

The evaluation pipeline prepends this to the stem when sending to the LLM (similar to `disease_reveal_prefix`). This ensures the LLM knows the parent answer before being asked the dependent question.

#### 7b. Check Procedure Scope

If `procedure_scope: all` → always include.

#### 7c. Extract Answer from JSON

Follow the `json_path` from the template. The base path depends on procedure type:

**Endoscopy** (location-wrapped):
- `sections.{Section}.attrs` → `report[location].diseases[disease].sections[Section].attrs`
- `sections.{Section}.subsections.{Sub}.attrs` → `...sections[Section].subsections[Sub].attrs`

**Colonoscopy** :
- `sections.{Section}.attrs` → `report[disease].sections[Section].attrs`
- `sections.{Section}.subsections.{Sub}.attrs` → `report[disease].sections[Section].subsections[Sub].attrs`

The answer is the key(s) where the value is `true`:
```python
attrs = get_by_path(report, json_path)
answers = [key for key, val in attrs.items() if val == True]
```

**If no answer found** (section exists but no attributes selected) → skip this question entirely. Do not generate a Q&A pair. Do not infer "None of the above" as the correct answer from an empty/unanswered section. All unanswered sections in the case JSON should result in skipped questions.

**However**, if "None" or an equivalent option (e.g., "No additional abnormality", "No bleeding") IS explicitly selected in the JSON as a `true` attribute, then it is a valid correct answer and the question should be generated normally.

**Multi-location diseases** (endoscopy only): Use the sections from the PRIMARY location (first in the JSON). If sections differ by location, use the location-specific data.

#### 7d. Generate MCQ

If `MCQ` is in the question's type list:

**Max 4 options rule:** Every MCQ has at most 4 options (A-D). The template contains ALL possible options; Step 2 selects a subset:
1. Include the correct answer(s) from the JSON
2. Fill remaining slots with random wrong options from the template's option list
3. Total = min(4, total_template_options). If the template has ≤4 options, use all of them.
4. **Distractor diversity rule:** When choosing which wrong options to include, avoid picking options that are semantically near-synonyms or near-duplicates of each other. The goal is to maximize the discriminative value of each distractor. Examples of near-pairs to avoid combining: "Round" / "Oval", "Flat" / "Slightly flat", "Mild" / "Moderate", "Irregular" / "Ill-defined". When both members of a near-pair are wrong, pick only one. When one member is the correct answer, the other may still be included (it tests fine-grained discrimination).
5. Shuffle the selected options randomly
6. Assign letters A, B, C, D
7. For single-select: one correct letter
8. For multi-select: list of correct letters

```json
{
  "phase": "detail",
  "disease": "{disease_name}",
  "type": "MCQ",
  "section": "Size",
  "subsection": null,
  "multi_select": false,
  "stem": "Regarding the largest {disease_name} lesion, what size would an expert endoscopist estimate?",
  "options": {"A": ">10 mm", "B": "5-10 mm", "C": "<5 mm"},
  "correct": "C",
  "answer_schema": {
    "type": "object",
    "properties": {
      "answer": {"type": "string", "enum": ["A", "B", "C"]}
    },
    "required": ["answer"],
    "additionalProperties": false
  }
}
```

**For multi-select:**
```json
{
  "phase": "detail",
  "type": "MCQ",
  "multi_select": true,
  "stem": "... (Select all that apply)",
  "options": {"A": "Dilated tortuous capillaries", "B": "Radial / spoke-wheel pattern", "C": "Central feeding vessel", "D": "None of the above"},
  "correct": ["A", "B"],
  "answer_schema": {
    "type": "object",
    "properties": {
      "answer": {"type": "array", "items": {"type": "string", "enum": ["A", "B", "C", "D"]}}
    },
    "required": ["answer"],
    "additionalProperties": false
  }
}
```

#### 7e. Generate Binary

If `Binary` is in the question's type list:

**For single-select questions with yes/no mapping:**
```json
{
  "phase": "detail",
  "type": "Binary",
  "stem": "... is {condition} present?",
  "options": ["Yes", "No"],
  "correct": "Yes",
  "answer_schema": {
    "type": "object",
    "properties": {
      "answer": {"type": "string", "enum": ["Yes", "No"]}
    },
    "required": ["answer"],
    "additionalProperties": false
  }
}
```

**For multi-select questions (per-attribute binaries):**
Generate one Binary question per attribute in the template:
```json
[
  {
    "phase": "detail",
    "type": "Binary",
    "section": "Vascular Architecture",
    "attribute": "Dilated tortuous capillaries",
    "stem": "... does the {disease} show dilated tortuous capillaries?",
    "options": ["Yes", "No"],
    "correct": "Yes",
    "answer_schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": ["Yes", "No"]}}, "required": ["answer"], "additionalProperties": false}
  },
  {
    "phase": "detail",
    "type": "Binary",
    "attribute": "Central feeding vessel",
    "stem": "... does the {disease} show a central feeding vessel?",
    "options": ["Yes", "No"],
    "correct": "No",
    "answer_schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": ["Yes", "No"]}}, "required": ["answer"], "additionalProperties": false}
  }
]
```

### Step 8: Assemble Output

Combine all questions into a JSON object. Every question MUST include `task` and `subtask` fields, and an `answer_schema` field — a JSON Schema object that the evaluation pipeline passes as `response_format` / `response_schema` to the LLM being tested.

**Task/subtask propagation:**

Every question MUST include `"task"` and `"subtask"` fields. Read the `subtask` value from the template:
- Disease identification: `mcq.subtask`, `binary_positive.subtask`, `binary_negatives.subtask`
- Location: `mcq.subtask` (or `subtask` for binary location)
- Sublocation: `part.subtask`, `wall.subtask` (or `part_subtask`/`wall_subtask` for binary sublocation)
- Detail questions: each item's `subtask` field

Derive `task` from the subtask prefix: `subtask.split("-")[0]` (e.g., `"DI-M"` → `"DI"`).

In `metadata`, include `task_distribution` (count per task) and `subtask_distribution` (count per subtask).

**Three answer_schema patterns:**

| Question type | `answer` field in schema |
|---|---|
| MCQ single-select | `{"type": "string", "enum": ["A","B","C","D"]}` — enum matches option letters |
| MCQ multi-select | `{"type": "array", "items": {"type": "string", "enum": ["A","B","C",...]}}` |
| Binary | `{"type": "string", "enum": ["Yes","No"]}` |

All schemas share the same wrapper: `{"type": "object", "properties": {"answer": ...}, "required": ["answer"], "additionalProperties": false}`

**Multi-turn conversation fields:**

Every question MUST include a `turn` field (sequential, starting at 1). The turn structure follows this order:

1. **DI MCQs** (Turns 1..N): One MCQ per disease. Video is attached in Turn 1.
2. **DI Binaries** (Turns N+1..M): All positive + negative binaries, shuffled and deduplicated.
3. **Non-disease finding questions** (if any): Hills hiatus, Ampulla, Mucosa — no reveal prefix.
4. **Per-disease post-DI questions**: For each disease in order:
   - The **first question** for each disease MUST include a `disease_reveal_prefix` field. The evaluation pipeline prepends this to the stem when sending to the LLM.
     - **First disease with segmentation frame** (disease's `segmentationFrame` is not null):
       `"This {procedure} video contains {disease_name} as seen around frame number {disease_segFrame} (frame numbers are on top-left in green font). Following questions are related to {disease_name}."`
     - **First disease without frame:** `"This {procedure} video contains {disease_name}. Following questions are related to {disease_name}."`
     - **2nd+ disease with segmentation frame:**
       `"This {procedure} video also contains {disease_name} as seen around frame number {disease_segFrame} (frame numbers are on top-left in green font). Following questions are related to {disease_name}."`
     - **2nd+ disease without frame:** `"This {procedure} video also contains {disease_name}. Following questions are related to {disease_name}."`
     - **Frame info rule:** Include the frame number whenever the individual disease has a non-null `segmentationFrame` (read from the disease entry in the report, NOT from `MetaData.segmentationFrame`). This applies to both single-disease and multi-disease cases.
   - Then location → sublocation → detail questions for that disease.

**Single-disease example:** 1 MCQ (Turn 1) → 3 binaries (Turns 2-4) → reveal at Turn 5 with first post-DI question.

**Multi-disease example (3 diseases):** 3 MCQs (Turns 1-3) → ~7 binaries (Turns 4-10) → Disease 1 reveal + questions → Disease 2 reveal + questions → Disease 3 reveal + questions.

**Important:** If location is skipped (single-organ disease) and sublocation is also skipped, the reveal prefix goes on the first detail question for that disease.

Include a top-level `conversation_model` object:
```json
"conversation_model": {
  "total_turns": 36,
  "di_mcq_turns": "1-3 (one MCQ per disease, video attached at Turn 1)",
  "di_binary_turns": "4-10 (positive + negative binaries, deduplicated, shuffled)",
  "turn_11_prefix": "This endoscopy video contains {disease_1} as seen around frame number {segFrame_1} (frame numbers are on top-left in green font). Following questions are related to {disease_1}.",
  "turn_17_prefix": "This endoscopy video also contains {disease_2} as seen around frame number {segFrame_2} (frame numbers are on top-left in green font). Following questions are related to {disease_2}.",
  "turn_24_prefix": "This endoscopy video also contains {disease_3} as seen around frame number {segFrame_3} (frame numbers are on top-left in green font). Following questions are related to {disease_3}.",
  "post_identification_turns": "11-36 (one question per turn)"
}
```

For `conversation_model.segmentation_frame`: set for single-disease cases (same as the disease's frame); omit or set null for multi-disease cases.

**Full output structure (2-disease example):**

```json
{
  "case_id": "{case_id}",
  "video": "{video}",
  "procedure_type": "{procedure}",
  "diseases_found": ["{disease_1}", "{disease_2}"],
  "diseases_with_templates": ["{disease_1}", "{disease_2}"],

  "conversation_model": {
    "total_turns": 20,
    "di_mcq_turns": "1-2 (one MCQ per disease, video attached at Turn 1)",
    "di_binary_turns": "3-7 (2 positive + 3 negative binaries, deduplicated, shuffled)",
    "turn_8_prefix": "This colonoscopy video contains {disease_1} as seen around frame number {segFrame_1} (frame numbers are on top-left in green font). Following questions are related to {disease_1}.",
    "turn_14_prefix": "This colonoscopy video also contains {disease_2} as seen around frame number {segFrame_2} (frame numbers are on top-left in green font). Following questions are related to {disease_2}.",
    "post_identification_turns": "8-20 (one question per turn)"
  },

  "questions": [
    {
      "question_number": 1,
      "turn": 1,
      "phase": "disease_identification",
      "disease": "{disease_1}",
      "task": "DI",
      "subtask": "DI-M",
      "type": "MCQ",
      "multi_select": false,
      "stem": "Given this endoscopy video clip, what condition would an expert endoscopist suspect among the given options?",
      "options": {"A": "Normal", "B": "Gastric Polyp", "C": "Adenoma", "D": "Hyperplastic polyp"},
      "correct": "B",
      "answer_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["answer"],
        "additionalProperties": false
      }
    },
    {
      "question_number": 2,
      "turn": 2,
      "phase": "disease_identification",
      "disease": "{disease_2}",
      "task": "DI",
      "subtask": "DI-M",
      "type": "MCQ",
      "multi_select": false,
      "stem": "This endoscopy video clip also shows another condition. What additional finding would an expert endoscopist suspect among the given options?",
      "options": {"A": "Normal", "B": "Erosion", "C": "Fistula", "D": "Xanthoma"},
      "correct": "B",
      "answer_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["answer"],
        "additionalProperties": false
      }
    },
    {
      "question_number": 3,
      "turn": 3,
      "phase": "disease_identification",
      "disease": "{disease_1}",
      "task": "DI",
      "subtask": "DI-C",
      "type": "Binary",
      "stem": "Given this endoscopy video clip, would an expert endoscopist suspect Gastric Polyp?",
      "options": ["Yes", "No"],
      "correct": "Yes",
      "answer_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["Yes", "No"]}},
        "required": ["answer"],
        "additionalProperties": false
      }
    },
    {
      "question_number": 4,
      "turn": 4,
      "phase": "disease_identification",
      "task": "DI",
      "subtask": "DI-R",
      "type": "Binary",
      "stem": "Given this endoscopy video clip, would an expert endoscopist suspect Fistula?",
      "options": ["Yes", "No"],
      "correct": "No",
      "answer_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["Yes", "No"]}},
        "required": ["answer"],
        "additionalProperties": false
      }
    },
    {
      "question_number": 8,
      "turn": 8,
      "disease_reveal_prefix": "This endoscopy video contains {disease_1}. Following questions are related to {disease_1}.",
      "phase": "location",
      "disease": "{disease_1}",
      "task": "AL",
      "subtask": "AL-O",
      "type": "MCQ",
      "multi_select": false,
      "stem": "In which organ is this {disease_1} present? ...",
      "options": {"A": "Esophagus", "B": "Stomach", "C": "Duodenum", "D": "More than one organ"},
      "correct": "B",
      "answer_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  ],

  "skipped_questions": [
    {"disease": "...", "template_id": "...", "reason": "..."}
  ],

  "skipped_diseases": [
    {"disease": "...", "reason": "No template at benchmark/templates/....yaml"}
  ],

  "metadata": {
    "generated_at": "{ISO_timestamp}",
    "json_source": "{json_file_path}",
    "templates_used": ["benchmark/templates/{slug_1}.yaml", "benchmark/templates/{slug_2}.yaml"],
    "total_questions": 20,
    "questions_by_type": {"MCQ": 12, "Binary": 8},
    "questions_by_phase": {
      "disease_identification": 7,
      "location": 1,
      "sublocation": 2,
      "detail": 10
    },
    "task_distribution": {"DI": 7, "AL": 3, "MA": 3, "PF": 1, "QN": 1, "GS": 1, "CF": 2, "CR": 1, "AR": 1},
    "subtask_distribution": {"DI-M": 2, "DI-C": 2, "DI-R": 3, "AL-O": 1, "AL-P": 1, "AL-W": 1, "MA-T": 1, "MA-SZ": 1}
  }
}
```

### Step 9: Write Output

Write to: `benchmark/qa/batch_N/{case_id}.json` (batch_N is specified when invoking, e.g., `batch_3`)

If the output directory doesn't exist, create it.

**IMPORTANT: No Unicode escapes.** Write all characters as actual UTF-8 glyphs, not `\uXXXX` escape sequences. Medical text contains en-dashes (`–`), comparison symbols (`≥`, `≤`), and other non-ASCII characters — these must appear as-is in the output JSON, never as `\u2013`, `\u2265`, etc. When using the Write tool, write the actual characters directly.

### Step 10: Summary

Print a summary:
```
Generated Q&A for case {case_id}:
  Video: {video}
  Procedure: {procedure}
  Diseases: {disease_1}, {disease_2}
  Total questions: 25 (15 MCQ, 10 Binary)
  Tasks: DI(4) AL(3) MA(5) PF(3) QN(1) GS(2) CF(2) CR(2) AR(1)
  Output: benchmark/qa/batch_N/{case_id}.json
```

---

## Edge Cases

### Multi-Disease Cases
- **DI MCQs first**: All disease identification MCQs are generated first (one per disease, sequential turns). 2nd+ MCQ stem: "This video also shows another condition..."
- **DI Binaries after MCQs**: All positive and negative binaries pooled together, deduplicated (no repeated distractor questions), shuffled into random order
- **Post-DI per disease**: Each disease gets its own location, sublocation, and detail sections with a `disease_reveal_prefix` on its first post-DI question
- Questions are numbered sequentially across all phases

### "More Than One Organ" Location
When a disease appears in multiple organs:
1. Location answer = "More than one organ"
2. After reveal: generate sublocation questions for EACH organ separately
3. Flow: "The {disease} is present in {organ1}. In which part..." → wall → "Also present in {organ2}..."

### Missing Data in JSON
If a section exists in the template but has no data in the JSON (no attrs selected), skip that question. Don't generate Q&A with no correct answer.

### Conditional Questions with Unmet Conditions
If the parent condition is not met (e.g., Paris != Ip so Stalk Length is irrelevant), skip entirely. Don't include conditional questions where the condition evaluates to false.

### Diseases Without Templates
Print a warning and skip. The template must exist to generate Q&A.

### Empty Distractors
If DiseaseComps has empty comparison columns for a disease, use placeholders and flag for manual review.

### Free-text Comments (legacy format only)
The finalized JSONs have no `comments` field. If encountered in older JSONs, ignore it — do NOT generate questions from free text.

---

## Conditional Evaluation Reference

The conditional strings in templates follow these patterns:

```
Section(AttributeName=Value)           → sections[Section].attrs[Value] == true
                                         OR sections[Section].subsections[AttributeName].attrs[Value] == true
Subsection(SubsectionName=Value)       → sections[parentSection].subsections[SubsectionName].attrs[Value] == true
X AND Y                                → both X and Y must be true
X OR Y                                 → either X or Y must be true
```

**Parsing algorithm:**
1. Split by ` OR ` (case-sensitive) → array of OR-clauses
2. Each OR-clause: split by ` AND ` → array of AND-conditions
3. Each condition: extract `Section(Name=Value)` or `Subsection(Name=Value)` via regex
4. Evaluate each condition against the JSON
5. Combine: all ANDs must be true; any OR must be true

**Example:**
```
"Section(Type=Ip) OR Section(Type=Isp)"
→ Check if sections["Paris Classification"].attrs["Ip"] == true
  OR sections["Paris Classification"].attrs["Isp"] == true
```

Note: The "Section" in `Section(X=Y)` refers to the parent section name from the EHR, and X is the subsection/attribute name within it. When there's no subsection, X is directly in attrs. When there IS a subsection, X refers to a subsection name and Y to an attr within it.
