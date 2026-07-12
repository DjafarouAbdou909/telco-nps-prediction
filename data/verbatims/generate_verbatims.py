"""
Generates one synthetic "last support interaction" verbatim per customer
in the test set, using the Gemini API, conditioned on a subset of the
customer's tabular features (tenure, contract, services, NPS risk).

Bonus track, section 4.4 of the Artefact challenge brief.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python generate_verbatims.py

Output:
    data/processed/verbatims.csv       — Customer ID, verbatim, sentiment_hint_used
    data/processed/verbatim_prompt.txt — the exact prompt template used (reproducibility)

Reproducibility notes:
    - The NOISE assignment (which customers get a "flipped" tone) is seeded
      with np.random.seed(SEED) and is therefore exactly reproducible.
    - The generated TEXT itself is not guaranteed byte-identical across runs
      even with the same seed, since Gemini's generation is not fully
      deterministic at temperature > 0. This is documented honestly rather
      than claimed as a false guarantee — what IS reproducible is which
      customers get which sentiment_hint and prompt, not the exact wording.
"""

import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from google import genai

SEED = 42
NOISE_FLIP_RATE = 0.15  # fraction of customers whose tone is deliberately mismatched with their true NPS category
MODEL_NAME = "gemini-2.0-flash"
DATA_DIR = Path("data/processed")
OUTPUT_CSV = DATA_DIR / "verbatims.csv"
PROMPT_FILE = DATA_DIR / "verbatim_prompt.txt"

PROMPT_TEMPLATE = """You are simulating a short customer support interaction note for a fictional \
telecom customer. Write 1 to 3 sentences, in the first person, as if the CUSTOMER themself \
wrote a brief note about their most recent interaction with customer support (could be a call \
summary, a chat message, or an app review).

Customer profile:
- Tenure: {tenure} months as a customer
- Contract type: {contract}
- Internet service: {internet_service}
- Monthly charges: ${monthly_charges:.2f}
- Number of services subscribed: {n_services}

Overall tone to convey: {tone_instruction}

Keep it realistic and natural, like something a real person would actually say — not overly \
dramatic, not a perfect textbook example. Do not mention any internal scoring, categories, or \
the word "detractor"/"promoter"/"passive". Just write what the customer would plausibly say.

Output only the verbatim text, nothing else."""

TONE_INSTRUCTIONS = {
    "Detractor": "frustrated or disappointed, showing real dissatisfaction with the service",
    "Passive": "neutral and matter-of-fact, neither particularly happy nor upset",
    "Promoter": "satisfied and positive, genuinely pleased with the service",
}


def build_prompt(row, sentiment_hint):
    return PROMPT_TEMPLATE.format(
        tenure=row["tenure"],
        contract=row["Contract"],
        internet_service=row["InternetService"],
        monthly_charges=row["MonthlyCharges"],
        n_services=row["n_services"],
        tone_instruction=TONE_INSTRUCTIONS[sentiment_hint],
    )


def assign_sentiment_hints(df, seed=SEED, flip_rate=NOISE_FLIP_RATE):
    """Mostly follow the true NPS_Category, but deliberately flip the tone
    for a fraction of customers to introduce realistic noise and
    counter-intuitive cases, as required by section 4.4 of the brief.
    """
    rng = random.Random(seed)
    categories = ["Detractor", "Passive", "Promoter"]
    hints = []
    for true_cat in df["NPS_Category"]:
        if rng.random() < flip_rate:
            hints.append(rng.choice([c for c in categories if c != true_cat]))
        else:
            hints.append(true_cat)
    return hints


def call_gemini(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
            time.sleep(2 ** attempt)


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable before running this script.")

    client = genai.Client(api_key=api_key)

    test = pd.read_csv(DATA_DIR / "telco_nps_test.csv")
    np.random.seed(SEED)

    sentiment_hints = assign_sentiment_hints(test, seed=SEED, flip_rate=NOISE_FLIP_RATE)
    test = test.copy()
    test["sentiment_hint_used"] = sentiment_hints

    n_flipped = (test["sentiment_hint_used"] != test["NPS_Category"]).sum()
    print(f"Generating verbatims for {len(test)} customers "
          f"({n_flipped} with deliberately mismatched tone, ~{NOISE_FLIP_RATE:.0%} noise rate).")

    # Save the exact prompt template for reproducibility, as required by 4.4
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(PROMPT_TEMPLATE)

    results = []
    # Resume support: skip customers already present in an existing output file
    already_done = set()
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        already_done = set(existing["Customer ID"])
        results = existing.to_dict("records")
        print(f"Resuming: {len(already_done)} customers already done.")

    for i, row in test.iterrows():
        cust_id = row["Customer ID"]
        if cust_id in already_done:
            continue

        prompt = build_prompt(row, row["sentiment_hint_used"])
        try:
            verbatim = call_gemini(client, prompt)
        except Exception as e:
            print(f"FAILED for {cust_id}: {e}")
            verbatim = None

        results.append({
            "Customer ID": cust_id,
            "verbatim": verbatim,
            "sentiment_hint_used": row["sentiment_hint_used"],
            "true_NPS_Category": row["NPS_Category"],
        })

        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"  ...{len(results)}/{len(test)} done, progress saved.")

        time.sleep(0.2)  # basic rate limiting

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved {len(results)} verbatims to {OUTPUT_CSV}")
    print(f"Prompt template saved to {PROMPT_FILE}")


if __name__ == "__main__":
    main()