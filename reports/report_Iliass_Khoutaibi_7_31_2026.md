### Report by: Iliass Khoutaibi. 07/31/2026 for paper about time series + LLM agents
## Relevant Agentic time series generation papers


- ## <ins>Nexus : An Agentic Framework for Time Series Forecasting</ins> (https://arxiv.org/pdf/2605.14389)

This paper introduced a new pipeline for Time Series Forcasting, the task is divided into three main parts: 

1. <ins><strong>Contextualization</strong></ins>: An agent is deployed to sumerize the contextual information of the time series, this helps mitigate the problem of  infromation overload and prevents the LLM from losing information
due to long contexts

2. <strong><ins>Micro and Macro reasoning</strong></ins>: 2 Agents are deployed to analyze the  broad picture of the time seris (shape, patterns, trajectory etc.) and the particular  details (Catalysts, short windows of the time series etc.)

3. <strong><ins> Time Series forcasting</strong></ins>: An agent is deployed to take the contextualized information, the micro and macro analyses to generate the time series infromation under certain guidelines fed by the user.


The used datasets are: 

1. Zillow Home forecasts:  https://www.zillow.com/research/data/  
2. Stock Market Prices: https://finance.yahoo.com/quote/ | https://stooq.com/


- ## <ins>Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs</ins> (https://arxiv.org pdf/2604.18576)

Instead of piling raw search results into context, it maintains a structured belief state, a probability plus evidence summary, updated at each step of an iterative search loop, then aggregates 5 trials in logit space and applies hierarchical calibration. Quite complicated paper.

The used datasets are: 

1. AIBQ2 (Metaculus AI Benchmark Tournament, Q2 2025) — https://www.metaculus.com/tournament/aibq2/  
2. ForecastBench — https://www.forecastbench.org/  
3. Polymarket — https://polymarket.com  
4. Manifold — https://manifold.markets  
5. Metaculus — https://metaculus.com  
6. Rand Forecasting Initiative (RFI) — https://randforecastinginitiative.org  
7. Yahoo Finance (yfinance) — https://finance.yahoo.com  
8. FRED (Federal Reserve Economic Data) — https://fred.stlouisfed.org
9. DBnomics — https://db.nomics.world  
10. Wikipedia — https://wikipedia.org  
11. ACLED (Armed Conflict Location & Event Data) — https://acleddata.com

- ## <ins>CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery </ins> (https://arxiv.org pdf/2604.18576) <span style="color: red;">Important paper !</span>

A very recent paper does not deal with time-series generation in particular, but rather with open-ended discovery as a whole. It formulates open-ended discovery as an iterative process consisting of four stages:


For each time step $t$ until termination... 

1. <strong>RETRIEVE</strong>: Construct a working context $\mathcal{M}_t$
2. <strong>PROPOSE</strong>: Generate a candidate solution to the problem $y_{t+1}$ conditionned on the input and the context $\mathcal{M}_t$
3. <strong>EVALUATE</strong>: Obtain a score and feedback on the proposed solution $s_{t+1} ,f_{t+1}=E(x,y_{t+1})$ with $E$ an evaluation metric.
4. <strong>UPDATE</strong>: Incorporate the newly acquired information into the shared persistent memory, producing $\mathcal{M}_{t+1}$ Older papers deployed independant agents with independant memories to generate new discoveries. CORAL proposes using a shared persistent memory from which agents can share their discoveries and solutions. The experiments are run for 3 hours, some heartbeat wakeup calls are also introduced in order to vear the agents from long contexts and bad solutions/reasonings.

No datasets were used, rather open ended optimization and mathematical tasks such as:
Circle Packing (Math) – https://arxiv.org/pdf/2604.01658  
Erdős Minimum Overlap (Math) – https://arxiv.org/pdf/2604.01658  
Signal Processing (Math) – https://arxiv.org/pdf/2604.01658  
Third Autocorrelation Inequalities (Math) – https://arxiv.org/pdf/2604.01658  
Min-Max Minimum Distance 2D/3D (Math) – https://arxiv.org/pdf/2604.01658  
EPLB (Systems) – https://arxiv.org/pdf/2604.01658  
PRISM (Systems) – https://arxiv.org/pdf/2604.01658  
LLM-SQL (Systems) – https://arxiv.org/pdf/2604.01658  
Transaction Scheduling (Systems) – https://arxiv.org/pdf/2604.01658  
Cloudcast (Systems) – https://arxiv.org/pdf/2604.01658  
Kernel Engineering (Stress Test) – https://arxiv.org/pdf/2604.01658  
Polyominoes (Stress Test) – https://arxiv.org/pdf/2604.01658  

- ## <ins>Dr-CiK: A Testbed for Foresight-Driven Agents </ins> (https://arxiv.org/pdf/2605.27904) <span style="color: red;">Important dataset !</span>

A benchmarking paper introducing a novel dataset specifically designed to highlight the challenges of context retrieval for LLM agents. Most existing work assumes that high-quality context is already available, which is not the case for many real-world tasks. This paper emphasizes that more research is needed to retrieve relevant context effectively while avoiding distractors.


The dataset is the paper itself ! 