### Report by: Iliass Khoutaibi. 07/31/2026 for paper about time series + LLM agents

## Dr-CiK Dataset — Retrieval & Context Extraction

**Dataset:** Dr-CiK will be used as the primary dataset. The hardest task is retrieving the important information from the dictionary.

### Plan

- **Time series agents for retrieval**: Use time series agents to retrieve the relevant information — possibly CORAL.
- **Understand existing retrieval approaches**: Look at LLM agents that have already tackled similar (not necessarily identical) retrieval tasks problems.
- **Literature review on retrieval methods**: Survey retrieval methods more broadly, not limited to time series.
- **Other datasets**: Look beyond Dr-CiK at other datasets, though Dr-CiK alone may be sufficient.
- **Context extraction for generation**: Use these agents to extract context for time series generation. This is important because the approach could generalize to other tasks requiring context, such as news retrieval.
- **Presentations**: Prepare presentations of the ideas, concise, no rambling, html.
- **What method ?**: Propose some methods to solve this problem, be critical. Single agent or multi agent ? Probably multi agent. we will need a shared memory: -One that evolves and other fixed one. In Dr-Cik can be improved regarding the recall one. 
- **Take a look at the time series papers**: Predicting y based on x. 
- **Claude Code**: Use it for coding but be careful of the otput.


<!-- Since the task at hand is one of context extraction, we can utilize the following datasets:

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
 -->
