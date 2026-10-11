# Group 02 — Crude Oil Direction Prediction: Does News Sentiment Add Value?

COMPSCI 760, The University of Auckland, Semester 2 2026.

We try to predict whether WTI crude oil closes up or down the next day, and
then check whether news headline sentiment actually adds anything on top of
a price-only baseline. (Short answer: not for tomorrow's direction - though
it does line up nicely with same-day returns, which is what ended up shaping
our whole conclusion.)

Members: Xiaoran Si, Changhua Guo, Zhengxin Yang, Yueming Qu, Xintong Zheng.

## The script that replicates the results

```bash
# Python 3.12+ with: pandas numpy scipy scikit-learn xgboost torch matplotlib
python Xiaoran_sentiment_alignment_and_modeling.py
```

That's the replication entry point. Takes about 8 minutes on a normal CPU,
no GPU needed. It redoes the full experiment: the news-to-trading-day
alignment (no look-ahead), 7 feature blocks x 3 models with expanding-window
walk-forward (train 500 / test 250 / step 250, so 5 folds and an
out-of-sample N of 1,185), 5,000-resample paired bootstrap, the leakage
positive control, the naive baselines, and Figures 1-7. Everything gets
written to `analysis_out/`.

The headline numbers, if you want to know before running it: none of the 18
sentiment/news augmentations beat the price-only baseline on AUC in a
statistically significant way (honest AUCs land between 0.466 and 0.514, and
the best accuracy is 51.48% vs the 52.49% you get by just always guessing
"up"). Sentiment does correlate with same-day returns (r between +0.054 and
+0.084, p < 0.05) but says basically nothing about tomorrow - the news seems
to be priced in within the day. So we ended up rejecting our original
hypothesis for next-day prediction and reframed the whole thing as an
explanation study.

## What's in this folder

| File | Who | What it does |
|---|---|---|
| `Xiaoran_price_pipeline_20200102_20260915.py` | Xiaoran Si | Builds the price dataset: Yahoo first / stooq fallback, cleaning, 7 technical features, QA, two volume patches, `label_up` -> `price_daily_20200102_20260915_labeled.csv` |
| `Xiaoran_sentiment_alignment_and_modeling.py` | Xiaoran Si | The replication script above - alignment, walk-forward, bootstrap, leakage control, all the figures |
| `ZhengxinY_Combined_DeepSeek_then_CrudeBERT_English.py` | Zhengxin Yang | Sentiment extraction: DeepSeek API pipeline plus CrudeBERT inference with the corrected label mapping, aggregated to daily sentiment CSVs |
| `ChanghuaG_modeling_bidirectional_package.zip` | Changhua Guo | Robustness checks: bidirectional (price -> news) VAR/Granger, sentiment decay, joint models, some alternative methods. Unzip it - there's a README, a requirements.txt and a `run_replicate_results.py` inside |

Each script says in its docstring what its author actually did and what was
taken from outside (see the attribution bit below).

## Full replication order

```bash
# (1) the price dataset
python Xiaoran_price_pipeline_20200102_20260915.py
#     -> price_daily_20200102_20260915_labeled.csv

# (2) daily sentiment (DeepSeek + CrudeBERT)
#     set DEEPSEEK_API_KEY in your environment first
python ZhengxinY_Combined_DeepSeek_then_CrudeBERT_English.py
#     -> sentiment_cb_daily_fixed.csv / sentiment_cb_daily_en.csv / sentiment_llm_daily.csv

# (3) the experiment (reads those CSVs from the same folder)
python Xiaoran_sentiment_alignment_and_modeling.py
#     -> analysis_out/ with model_comparison.csv, significance_tests.csv,
#        contemporaneous_vs_next_day.csv, naive_baselines.csv, fold_details.csv, Fig 1-7

# (4) optional robustness package
unzip ChanghuaG_modeling_bidirectional_package.zip && cd ChanghuaG_*
pip install -r requirements.txt
python run_replicate_results.py
```

One practical note on step (2): it only hits the DeepSeek API for dates that
aren't in its SQLite cache, and the cached daily outputs ship with the
package - so you can replicate everything without an API key if you skip
re-running the LLM part.

## Dependencies

Python 3.12+. The experiment itself needs `pandas numpy scipy scikit-learn
xgboost torch matplotlib`. Step (2) additionally wants `requests aiohttp
nest_asyncio tqdm transformers huggingface_hub` and a `DEEPSEEK_API_KEY`
environment variable (we never hard-code it). The bidirectional package has
its own `requirements.txt`.

## Contribution vs things taken from outside

Our own work: the research design (hypothesis, feature blocks, walk-forward
protocol, the leakage positive control, the same-day-vs-next-day diagnostic),
all the code in this folder, the price dataset assembly (including the two
volume patches we checked by hand), the news corpus collection and cleaning,
and every number / table / figure in the deck and report - all of those come
from our own runs.

Used as-is from outside, with credit:

* CrudeBERT pretrained weights - Kaplan, Trung & Van Durme (2023),
  "CrudeBERT: A Transformer Model of Financial Sentiment for Crude Oil
  Markets", Hugging Face `Captain-1337/CrudeBERT` (MIT licence). We run
  inference only, no retraining or fine-tuning, and the label mapping was
  verified rather than changed.
* DeepSeek API - commercial LLM service; the prompts, caching and QA are ours.
* Data - Yahoo Finance (CL=F), stooq, investing.com (the verified volume
  patches), GDELT 2.0 DOC API, EIA Today in Energy, StockTwits, RSS /
  FreeNewsAPI. Headline text belongs to the original publishers.
* Libraries - pandas, numpy, scipy, scikit-learn, xgboost, PyTorch,
  Hugging Face transformers.

We don't quote numbers from papers as if they were our results - literature
figures are cited where they appear (e.g. Kaplan et al. 2023), and every
figure or table in the deck points at the `analysis_out/` file or script
that produced it.

## Who did what

* **Xiaoran Si** - the price data pipeline, the news-price alignment and the
  full modelling experiment (walk-forward, bootstrap, leakage control,
  figures), plus putting the final deck and report together.
* **Changhua Guo** - news corpus collection and cleaning, and the
  bidirectional / decay / Granger robustness package.
* **Zhengxin Yang** - the DeepSeek and CrudeBERT sentiment pipelines,
  including the corrected label mapping and the contaminated-vs-cleaned
  headline comparison.
* **Yueming Qu** - modelling support experiments and cross-checking results.
* **Xintong Zheng** - consolidating results and documentation.

## Caveats we state up front

* GDELT timestamps come in 3-hour buckets, so we can't tell pre-close from
  post-close news - roughly half a day of blur.
* The news channel mix is not stationary: about 6.26 headlines a day in
  2020-2025 vs 24.36 a day after March 2026, and StockTwits only exists from
  October 2025. Any fixed model has to absorb that structural break.
* Daily aggregation throws away intra-day semantics, and the backtest cost
  model is deliberately simple.
* There's one run with a +79% cumulative return (XGBoost + DeepSeek) - it's
  a multiple-comparisons artefact, its AUC is 0.5053 with p = 0.18, and it's
  the best out of 21 runs. Don't read too much into it.
