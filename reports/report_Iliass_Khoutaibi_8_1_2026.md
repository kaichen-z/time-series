### Report by: Iliass Khoutaibi. 07/31/2026 for paper about time series + LLM agents
## Which datasets we will use:

Since the task at hand is one of context extraction, we can utilize the following datasets:

### 1. Dr-CiK: https://huggingface.co/datasets/ServiceNow/Dr-CiK: 

    - Real-world time-series forecasting often depends not only on historical observations but also on external context that must be actively discovered from heterogeneous, noisy information sources.

    - Each task provides:
        - a historical time series (history_timestamps, history_values) and the ground-truth continuation (future_timestamps, future_values) entity / profile metadata and a target description
        - a corpus of Markdown documents — a mix of supporting documents (which contain the evidence needed to forecast) and distractor documents (which do not)
        - ground-truth evidence (gt_evidence) for evaluation.


### 2. Context-is-key: https://huggingface.co/datasets/ServiceNow/context-is-key

    - Original paper, the previous one was the following paper. Can be found on hugging face. Informations are found on the hugging face website.

### 3. GIFT-CTX: https://huggingface.co/datasets/Salesforce/GIFT-CTX

    - Updated version of Context-is-key, however I did not find contextual information in it.

