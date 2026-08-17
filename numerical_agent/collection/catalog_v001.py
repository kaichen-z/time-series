"""Curated, source-grounded authoring helpers for forecast methods v001.

This module is deliberately declarative: it records reviewed primary sources and
the evidence assignment for each migrated method.  It does not infer provenance
from names and it drops legacy constructs for which no reviewed definition was
found.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import MethodCard, SourceRecord
from .seed import CATEGORY_BY_LEGACY_ID


SOURCE_PAYLOADS: tuple[Mapping[str, object], ...] = (
    {
        "source_id": "source_000001",
        "title": "Forecasting: Principles and Practice (3rd ed)",
        "authors": ["Rob J. Hyndman", "George Athanasopoulos"],
        "year": 2021,
        "source_type": "textbook",
        "url": "https://otexts.com/fpp3/",
        "doi": "",
        "isbn": "9780987507136",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000002",
        "title": "StatsForecast model documentation",
        "authors": ["Nixtla"],
        "year": 2026,
        "source_type": "official_docs",
        "url": "https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000003",
        "title": "Statsmodels state space methods documentation",
        "authors": ["Statsmodels developers"],
        "year": 2026,
        "source_type": "official_docs",
        "url": "https://www.statsmodels.org/stable/statespace.html",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000004",
        "title": "The theta model: a decomposition approach to forecasting",
        "authors": ["Vassilis Assimakopoulos", "Konstantinos Nikolopoulos"],
        "year": 2000,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/S0169-2070(00)00066-2",
        "doi": "10.1016/S0169-2070(00)00066-2",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000005",
        "title": "Forecasting and stock control for intermittent demands",
        "authors": ["J. D. Croston"],
        "year": 1972,
        "source_type": "paper",
        "url": "https://doi.org/10.1057/jors.1972.50",
        "doi": "10.1057/jors.1972.50",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000006",
        "title": "The accuracy of intermittent demand estimates",
        "authors": ["John E. Boylan", "Aris A. Syntetos"],
        "year": 2005,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/j.ijforecast.2004.10.001",
        "doi": "10.1016/j.ijforecast.2004.10.001",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000007",
        "title": "Intermittent demand: linking forecasting to inventory obsolescence",
        "authors": ["Ruud H. Teunter", "Aris A. Syntetos", "M. Zied Babai"],
        "year": 2011,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/j.ejor.2010.09.018",
        "doi": "10.1016/j.ejor.2010.09.018",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000008",
        "title": "An aggregate-disaggregate intermittent demand approach",
        "authors": [
            "Konstantinos Nikolopoulos",
            "Aris A. Syntetos",
            "John E. Boylan",
            "Fotios Petropoulos",
            "Vassilis Assimakopoulos",
        ],
        "year": 2011,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/j.ijforecast.2010.09.008",
        "doi": "10.1016/j.ijforecast.2010.09.008",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000009",
        "title": "Improving forecasting by estimating time series structural components across multiple frequencies",
        "authors": ["Nikolaos Kourentzes", "Fotios Petropoulos", "Juan R. Trapero"],
        "year": 2014,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/j.ijforecast.2013.09.006",
        "doi": "10.1016/j.ijforecast.2013.09.006",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000010",
        "title": "STL: A seasonal-trend decomposition procedure based on loess",
        "authors": ["Robert B. Cleveland", "William S. Cleveland", "Jean E. McRae", "Irma Terpenning"],
        "year": 1990,
        "source_type": "paper",
        "url": "https://www.wessa.net/download/stl.pdf",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000011",
        "title": "Forecast combinations: an over 50-year review",
        "authors": ["Xiaoqian Wang", "Rob J. Hyndman", "Feng Li", "Yanfei Kang"],
        "year": 2023,
        "source_type": "paper",
        "url": "https://robjhyndman.com/publications/combinations/",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000012",
        "title": "Forecasting at scale",
        "authors": ["Sean J. Taylor", "Benjamin Letham"],
        "year": 2018,
        "source_type": "paper",
        "url": "https://doi.org/10.1080/00031305.2017.1380080",
        "doi": "10.1080/00031305.2017.1380080",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
)

ADDITIONAL_STATISTICAL_SOURCE_PAYLOADS: tuple[Mapping[str, object], ...] = (
    {
        "source_id": "source_000013",
        "title": "Predicting chaotic time series",
        "authors": ["J. Doyne Farmer", "John J. Sidorowich"],
        "year": 1987,
        "source_type": "paper",
        "url": "https://doi.org/10.1103/PhysRevLett.59.845",
        "doi": "10.1103/PhysRevLett.59.845",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000014",
        "title": "Conformal prediction interval for dynamic time-series",
        "authors": ["Chen Xu", "Yao Xie"],
        "year": 2021,
        "source_type": "paper",
        "url": "https://proceedings.mlr.press/v139/xu21h.html",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000015",
        "title": "Gaussian Processes for Machine Learning",
        "authors": ["Carl Edward Rasmussen", "Christopher K. I. Williams"],
        "year": 2006,
        "source_type": "textbook",
        "url": "https://gaussianprocess.org/gpml/",
        "doi": "",
        "isbn": "9780262182539",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000016",
        "title": "MLForecast documentation",
        "authors": ["Nixtla"],
        "year": 2026,
        "source_type": "official_docs",
        "url": "https://nixtlaverse.nixtla.io/mlforecast/",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000017",
        "title": "DeepAR: Probabilistic forecasting with autoregressive recurrent networks",
        "authors": ["David Salinas", "Valentin Flunkert", "Jan Gasthaus"],
        "year": 2017,
        "source_type": "paper",
        "url": "https://arxiv.org/abs/1704.04110",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000018",
        "title": "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting",
        "authors": ["Boris N. Oreshkin", "Dmitri Carpov", "Nicolas Chapados", "Yoshua Bengio"],
        "year": 2019,
        "source_type": "paper",
        "url": "https://arxiv.org/abs/1905.10437",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000019",
        "title": "N-HiTS: Neural hierarchical interpolation for time series forecasting",
        "authors": ["Cristian Challu", "Kin G. Olivares", "Boris N. Oreshkin", "Federico Garza", "Max Mergenthaler-Canseco", "Artur Dubrawski"],
        "year": 2022,
        "source_type": "paper",
        "url": "https://arxiv.org/abs/2201.12886",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000020",
        "title": "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting",
        "authors": ["Bryan Lim", "Sercan O. Arik", "Nicolas Loeff", "Tomas Pfister"],
        "year": 2019,
        "source_type": "paper",
        "url": "https://arxiv.org/abs/1912.09363",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000021",
        "title": "Joint estimation of model parameters and outlier effects in time series",
        "authors": ["Chung Chen", "Lon-Mu Liu"],
        "year": 1993,
        "source_type": "paper",
        "url": "https://doi.org/10.1080/01621459.1993.10476344",
        "doi": "10.1080/01621459.1993.10476344",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000022",
        "title": "A new approach to the economic analysis of nonstationary time series and the business cycle",
        "authors": ["James D. Hamilton"],
        "year": 1989,
        "source_type": "paper",
        "url": "https://doi.org/10.2307/1912559",
        "doi": "10.2307/1912559",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000023",
        "title": "Threshold Models in Non-linear Time Series Analysis",
        "authors": ["Howell Tong"],
        "year": 1983,
        "source_type": "textbook",
        "url": "https://link.springer.com/book/10.1007/978-1-4684-7888-4",
        "doi": "10.1007/978-1-4684-7888-4",
        "isbn": "9781468478907",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000024",
        "title": "Specification, estimation, and evaluation of smooth transition autoregressive models",
        "authors": ["Timo Terasvirta"],
        "year": 1994,
        "source_type": "paper",
        "url": "https://doi.org/10.1080/01621459.1994.10476462",
        "doi": "10.1080/01621459.1994.10476462",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000025",
        "title": "Forecasting time series with complex seasonal patterns using exponential smoothing",
        "authors": ["Alysha M. De Livera", "Rob J. Hyndman", "Ralph D. Snyder"],
        "year": 2011,
        "source_type": "paper",
        "url": "https://doi.org/10.1198/jasa.2011.tm09771",
        "doi": "10.1198/jasa.2011.tm09771",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000026",
        "title": "Random forests",
        "authors": ["Leo Breiman"],
        "year": 2001,
        "source_type": "paper",
        "url": "https://doi.org/10.1023/A:1010933404324",
        "doi": "10.1023/A:1010933404324",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000027",
        "title": "XGBoost: A scalable tree boosting system",
        "authors": ["Tianqi Chen", "Carlos Guestrin"],
        "year": 2016,
        "source_type": "paper",
        "url": "https://arxiv.org/abs/1603.02754",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000028",
        "title": "LightGBM: A highly efficient gradient boosting decision tree",
        "authors": ["Guolin Ke", "Qi Meng", "Thomas Finley", "Taifeng Wang", "Wei Chen", "Weidong Ma", "Qiwei Ye", "Tie-Yan Liu"],
        "year": 2017,
        "source_type": "paper",
        "url": "https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000029",
        "title": "A tutorial on support vector regression",
        "authors": ["Alex J. Smola", "Bernhard Scholkopf"],
        "year": 2004,
        "source_type": "paper",
        "url": "https://doi.org/10.1023/B:STCO.0000035301.49549.88",
        "doi": "10.1023/B:STCO.0000035301.49549.88",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
)

FOUNDATION_SOURCE_PAYLOADS: tuple[Mapping[str, object], ...] = (
    {
        "source_id": "source_000030", "title": "A decoder-only foundation model for time-series forecasting",
        "authors": ["Abhimanyu Das", "Weihao Kong", "Rajat Sen", "Yichen Zhou"], "year": 2023,
        "source_type": "paper", "url": "https://arxiv.org/abs/2310.10688", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000031", "title": "TimesFM official repository",
        "authors": ["Google Research"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/google-research/timesfm", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
    {
        "source_id": "source_000032", "title": "Chronos: Learning the Language of Time Series",
        "authors": ["Abdul Fatir Ansari", "Lorenzo Stella", "Caner Turkmen", "Xiyuan Zhang", "Pedro Mercado", "Huibin Shen", "Oleksandr Shchur", "Syama Sundar Rangapuram", "Sebastian Pineda Arango", "Shubham Kapoor", "Jasper Zschiegner", "Danielle C. Maddix", "Hao Wang", "Michael W. Mahoney", "Kari Torkkola", "Andrew Gordon Wilson", "Michael Bohlke-Schneider", "Yuyang Wang"],
        "year": 2024, "source_type": "paper", "url": "https://arxiv.org/abs/2403.07815", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000033", "title": "Chronos forecasting official repository",
        "authors": ["Amazon Science"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/amazon-science/chronos-forecasting", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
    {
        "source_id": "source_000034", "title": "Unified Training of Universal Time Series Forecasting Transformers",
        "authors": ["Gerald Woo", "Chenghao Liu", "Akshat Kumar", "Caiming Xiong", "Silvio Savarese", "Doyen Sahoo"],
        "year": 2024, "source_type": "paper", "url": "https://arxiv.org/abs/2402.02592", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000035", "title": "Uni2TS and Moirai official repository",
        "authors": ["Salesforce AI Research"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/SalesforceAIResearch/uni2ts", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
    {
        "source_id": "source_000036", "title": "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting",
        "authors": ["Kashif Rasul", "Arjun Ashok", "Andrew Robert Williams", "Hena Ghonia", "Rishika Bhagwatkar", "Arian Khorasani", "Mohammad Javad Darvishi Bayazi", "George Adamopoulos", "Roland Riachi", "Nadhir Hassen", "Marin Bilos", "Sahil Garg", "Anderson Schneider", "Nicolas Chapados", "Alexandre Drouin", "Valentina Zantedeschi", "Yuriy Nevmyvaka", "Irina Rish"],
        "year": 2023, "source_type": "paper", "url": "https://arxiv.org/abs/2310.08278", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000037", "title": "Lag-Llama model card",
        "authors": ["Time Series Foundation Models"], "year": 2024, "source_type": "model_card",
        "url": "https://huggingface.co/time-series-foundation-models/Lag-Llama", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000038", "title": "MOMENT: A Family of Open Time-series Foundation Models",
        "authors": ["Mononito Goswami", "Konrad Szafer", "Arjun Choudhry", "Yifu Cai", "Shuo Li", "Artur Dubrawski"],
        "year": 2024, "source_type": "paper", "url": "https://arxiv.org/abs/2402.03885", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000039", "title": "MOMENT official repository",
        "authors": ["MOMENT authors"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/moment-timeseries-foundation-model/moment", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
    {
        "source_id": "source_000040", "title": "Tiny Time Mixers: Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series",
        "authors": ["Vijay Ekambaram", "Arindam Jati", "Pankaj Dayama", "Sumanta Mukherjee", "Nam H. Nguyen", "Wesley M. Gifford", "Chandra Reddy", "Jayant Kalagnanam"],
        "year": 2024, "source_type": "paper", "url": "https://arxiv.org/abs/2401.03955", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000041", "title": "Granite Tiny Time Mixer R2 model card",
        "authors": ["IBM Research"], "year": 2025, "source_type": "model_card",
        "url": "https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000042", "title": "Timer: Generative Pre-trained Transformers Are Large Time Series Models",
        "authors": ["Yong Liu", "Haoran Zhang", "Chenyu Li", "Xiangdong Huang", "Jianmin Wang", "Mingsheng Long"],
        "year": 2024, "source_type": "paper", "url": "https://arxiv.org/abs/2402.02368", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000043", "title": "Timer official repository",
        "authors": ["THUML"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/thuml/Large-Time-Series-Model", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
    {
        "source_id": "source_000044", "title": "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts",
        "authors": ["Xiaoming Shi", "Shiyu Wang", "Yuqi Nie", "Dianqi Li", "Zhou Ye", "Qingsong Wen", "Ming Jin"],
        "year": 2024, "source_type": "paper", "url": "https://arxiv.org/abs/2409.16040", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000045", "title": "Time-MoE official repository",
        "authors": ["Time-MoE authors"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/Time-MoE/Time-MoE", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
    {
        "source_id": "source_000046", "title": "ForecastPFN: Synthetically-Trained Zero-Shot Forecasting",
        "authors": ["Samuel Dooley", "Gurnoor Singh Khurana", "Chirag Mohapatra", "Siddartha Naidu", "Colin White"],
        "year": 2023, "source_type": "paper", "url": "https://arxiv.org/abs/2311.01933", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000047", "title": "ForecastPFN official repository",
        "authors": ["Abacus.AI"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/abacusai/ForecastPFN", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
    {
        "source_id": "source_000048", "title": "TimeGPT-1",
        "authors": ["Azul Garza", "Cristian Challu", "Max Mergenthaler-Canseco"],
        "year": 2023, "source_type": "paper", "url": "https://arxiv.org/abs/2310.03589", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000049", "title": "TimeGPT official documentation",
        "authors": ["Nixtla"], "year": 2026, "source_type": "official_docs",
        "url": "https://www.nixtla.io/docs", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000050", "title": "TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting",
        "authors": ["Defu Cao", "Furong Jia", "Sercan O. Arik", "Tomas Pfister", "Yixiang Zheng", "Wen Ye", "Yan Liu"],
        "year": 2023, "source_type": "paper", "url": "https://arxiv.org/abs/2310.04948", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000051", "title": "UniTS: Building a Unified Time Series Model",
        "authors": ["Shanghua Gao", "Teddy Koker", "Owen Queen", "Thomas Hartvigsen", "Theodoros Tsiligkaridis", "Marinka Zitnik"],
        "year": 2024, "source_type": "paper", "url": "https://arxiv.org/abs/2403.00131", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000052", "title": "UniTS official repository",
        "authors": ["UniTS authors"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/mims-harvard/UniTS", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
    {
        "source_id": "source_000053", "title": "Sundial: A Family of Highly Capable Time Series Foundation Models",
        "authors": ["Yong Liu", "Guo Qin", "Zhiyuan Shi", "Zhi Chen", "Caiyin Yang", "Xiangdong Huang", "Jianmin Wang", "Mingsheng Long"],
        "year": 2025, "source_type": "paper", "url": "https://arxiv.org/abs/2502.00816", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000054", "title": "Sundial official repository",
        "authors": ["THUML"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/thuml/Sundial", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
    {
        "source_id": "source_000055", "title": "Toto 2.0: Time Series Forecasting Enters the Scaling Era",
        "authors": ["Emaad Khwaja", "Chris Lettieri", "Gerald Woo", "Eden Belouadah", "Marc Cenac", "Guillaume Jarry", "Enguerrand Paquin", "Xunyi Zhao", "Viktoriya Zhukov", "Othmane Abou-Amal", "Chenghao Liu", "Ameet Talwalkar", "David Asker"],
        "year": 2026, "source_type": "paper", "url": "https://arxiv.org/abs/2605.20119", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000056", "title": "Toto official repository",
        "authors": ["Datadog"], "year": 2026, "source_type": "official_repo",
        "url": "https://github.com/DataDog/toto", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-17", "primary": False, "review_status": "verified",
    },
)

COMBINED_SOURCE_PAYLOADS: tuple[Mapping[str, object], ...] = (
    {
        "source_id": "source_000057", "title": "The Combination of Forecasts",
        "authors": ["J. M. Bates", "C. W. J. Granger"], "year": 1969,
        "source_type": "paper", "url": "https://doi.org/10.1057/jors.1969.103",
        "doi": "10.1057/jors.1969.103", "isbn": "", "retrieved_at": "2026-08-17",
        "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000058", "title": "Stacked generalization",
        "authors": ["David H. Wolpert"], "year": 1992, "source_type": "paper",
        "url": "https://doi.org/10.1016/S0893-6080(05)80023-1",
        "doi": "10.1016/S0893-6080(05)80023-1", "isbn": "", "retrieved_at": "2026-08-17",
        "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000059", "title": "FFORMA: Feature-based forecast model averaging",
        "authors": ["Pablo Montero-Manso", "George Athanasopoulos", "Rob J. Hyndman", "Thiyanga S. Talagala"],
        "year": 2020, "source_type": "paper", "url": "https://doi.org/10.1016/j.ijforecast.2019.02.011",
        "doi": "10.1016/j.ijforecast.2019.02.011", "isbn": "", "retrieved_at": "2026-08-17",
        "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000060", "title": "Time series forecasting using a hybrid ARIMA and neural network model",
        "authors": ["G. Peter Zhang"], "year": 2003, "source_type": "paper",
        "url": "https://doi.org/10.1016/S0925-2312(01)00702-0",
        "doi": "10.1016/S0925-2312(01)00702-0", "isbn": "", "retrieved_at": "2026-08-17",
        "primary": True, "review_status": "verified",
    },
)


DEPTH_EXPANSION_SOURCE_PAYLOADS: tuple[Mapping[str, object], ...] = (
    {
        "source_id": "source_000061", "title": "Bayesian Online Changepoint Detection",
        "authors": ["Ryan Prescott Adams", "David J. C. MacKay"], "year": 2007,
        "source_type": "paper", "url": "https://arxiv.org/abs/0710.3742", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000062", "title": "Optimal detection of changepoints with a linear computational cost",
        "authors": ["Rebecca Killick", "Paul Fearnhead", "Idris A. Eckley"], "year": 2012,
        "source_type": "paper", "url": "https://arxiv.org/abs/1101.1438", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000063", "title": "Forecast reconciliation: A review",
        "authors": ["George Athanasopoulos", "Rob J. Hyndman", "Nikolaos Kourentzes", "Anastasios Panagiotelis"], "year": 2024,
        "source_type": "paper", "url": "https://robjhyndman.com/publications/frreview.html", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000064", "title": "Forecasting with temporal hierarchies",
        "authors": ["George Athanasopoulos", "Rob J. Hyndman", "Nikolaos Kourentzes", "Fotios Petropoulos"], "year": 2017,
        "source_type": "paper", "url": "https://robjhyndman.com/publications/temporal-hierarchies/", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000065", "title": "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization",
        "authors": ["Shanika L. Wickramasuriya", "George Athanasopoulos", "Rob J. Hyndman"], "year": 2019,
        "source_type": "paper", "url": "https://robjhyndman.com/publications/mint/", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000066", "title": "Are Transformers Effective for Time Series Forecasting?",
        "authors": ["Ailing Zeng", "Muxi Chen", "Lei Zhang", "Qiang Xu"], "year": 2022,
        "source_type": "paper", "url": "https://arxiv.org/abs/2205.13504", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000067", "title": "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
        "authors": ["Yuqi Nie", "Nam H. Nguyen", "Phanwadee Sinthong", "Jayant Kalagnanam"], "year": 2022,
        "source_type": "paper", "url": "https://arxiv.org/abs/2211.14730", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000068", "title": "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting",
        "authors": ["Haixu Wu", "Jiehui Xu", "Jianmin Wang", "Mingsheng Long"], "year": 2021,
        "source_type": "paper", "url": "https://arxiv.org/abs/2106.13008", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000069", "title": "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
        "authors": ["Tian Zhou", "Ziqing Ma", "Qingsong Wen", "Xue Wang", "Liang Sun", "Rong Jin"], "year": 2022,
        "source_type": "paper", "url": "https://arxiv.org/abs/2201.12740", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000070", "title": "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
        "authors": ["Yong Liu", "Tengge Hu", "Haoran Zhang", "Haixu Wu", "Shiyu Wang", "Lintao Ma", "Mingsheng Long"], "year": 2023,
        "source_type": "paper", "url": "https://arxiv.org/abs/2310.06625", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000071", "title": "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis",
        "authors": ["Haixu Wu", "Tengge Hu", "Yong Liu", "Hang Zhou", "Jianmin Wang", "Mingsheng Long"], "year": 2022,
        "source_type": "paper", "url": "https://arxiv.org/abs/2210.02186", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000072", "title": "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting",
        "authors": ["Shiyu Wang", "Haixu Wu", "Xiaoming Shi", "Tengge Hu", "Huakun Luo", "Lintao Ma", "James Y. Zhang", "Jun Zhou"], "year": 2024,
        "source_type": "paper", "url": "https://arxiv.org/abs/2405.14616", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000073", "title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
        "authors": ["Haoyi Zhou", "Shanghang Zhang", "Jieqi Peng", "Shuai Zhang", "Jianxin Li", "Hui Xiong", "Wancai Zhang"], "year": 2020,
        "source_type": "paper", "url": "https://arxiv.org/abs/2012.07436", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000074", "title": "TSMixer: An All-MLP Architecture for Time Series Forecasting",
        "authors": ["Si-An Chen", "Chun-Liang Li", "Nate Yoder", "Sercan O. Arik", "Tomas Pfister"], "year": 2023,
        "source_type": "paper", "url": "https://arxiv.org/abs/2303.06053", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000075", "title": "Long-term Forecasting with TiDE: Time-series Dense Encoder",
        "authors": ["Abhimanyu Das", "Weihao Kong", "Andrew Leach", "Shaan Mathur", "Rajat Sen", "Rose Yu"], "year": 2023,
        "source_type": "paper", "url": "https://arxiv.org/abs/2304.08424", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000076", "title": "SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction",
        "authors": ["Minhao Liu", "Ailing Zeng", "Muxi Chen", "Zhijian Xu", "Qiuxia Lai", "Lingna Ma", "Qiang Xu"], "year": 2021,
        "source_type": "paper", "url": "https://arxiv.org/abs/2106.09305", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000077", "title": "SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention",
        "authors": ["Romain Ilbert", "Ambroise Odonnat", "Vasilii Feofanov", "Aladin Virmaux", "Giuseppe Paolo", "Themis Palpanas", "Ievgen Redko"], "year": 2024,
        "source_type": "paper", "url": "https://arxiv.org/abs/2402.10198", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000078", "title": "Timer-S1: A Billion-Scale Time Series Foundation Model with Serial Scaling",
        "authors": ["Yong Liu", "Xingjian Su", "Shiyu Wang", "Haoran Zhang", "Haixuan Liu", "Yuxuan Wang", "Zhou Ye", "Yang Xiang", "Jianmin Wang", "Mingsheng Long"], "year": 2026,
        "source_type": "paper", "url": "https://arxiv.org/abs/2603.04791", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000079", "title": "Regression Quantiles",
        "authors": ["Roger Koenker", "Gilbert Bassett Jr."], "year": 1978,
        "source_type": "paper", "url": "https://doi.org/10.2307/1913643", "doi": "10.2307/1913643", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000080", "title": "Chronos-2: From Univariate to Universal Forecasting",
        "authors": ["Abdul Fatir Ansari", "Oleksandr Shchur", "Jaris Kuken", "Andreas Auer", "Boran Han", "Pedro Mercado", "Syama Sundar Rangapuram", "Huibin Shen", "Lorenzo Stella", "Xiyuan Zhang", "Mononito Goswami", "Shubham Kapoor", "Danielle C. Maddix", "Pablo Guerron", "Tony Hu", "Junming Yin", "Nick Erickson", "Prateek Mutalik Desai", "Hao Wang", "Huzefa Rangwala", "George Karypis", "Yuyang Wang", "Michael Bohlke-Schneider"],
        "year": 2025, "source_type": "paper", "url": "https://arxiv.org/abs/2510.15821", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
    {
        "source_id": "source_000081", "title": "Moirai 2.0: When Less Is More for Time Series Forecasting",
        "authors": ["Chenghao Liu", "Taha Aksu", "Juncheng Liu", "Xu Liu", "Hanshu Yan", "Quang Pham", "Doyen Sahoo", "Caiming Xiong", "Silvio Savarese", "Junnan Li"],
        "year": 2025, "source_type": "paper", "url": "https://arxiv.org/abs/2511.11698", "doi": "", "isbn": "",
        "retrieved_at": "2026-08-18", "primary": True, "review_status": "verified",
    },
)


EXCLUDED_LEGACY_IDS = {
    "fft_dominant_frequency_extrapolation",
    "wavelet_trend_detail_forecast",
    "empirical_quantile_persistence",
}

SPECIFIC_DEFINITION_SOURCES: Mapping[str, tuple[str, ...]] = {
    "theta_classic": ("source_000004",),
    "theta_optimized": ("source_000002", "source_000004"),
    "croston": ("source_000005",),
    "croston_sba": ("source_000006",),
    "tsb": ("source_000007",),
    "adida": ("source_000008",),
    "imapa": ("source_000009",),
    "local_level_kalman": ("source_000003",),
    "local_linear_trend_kalman": ("source_000003",),
    "structural_time_series_bsm": ("source_000003",),
    "robust_loess_trend": ("source_000010",),
    "piecewise_linear_trend": ("source_000012",),
    "statistical_ensemble_mean": ("source_000011",),
}

EXTRA_STATISTICAL_METHOD_SPECS: tuple[Mapping[str, object], ...] = (
    {
        "name": "nearest_neighbor_lag_analogue",
        "category": "analogue",
        "description": "Match the latest lag vector to historical lag vectors and average their observed continuations.",
        "assumption": "Past states close in lag space have similar future continuations.",
        "failure": "Sparse history or a regime shift leaves no genuinely comparable analogue.",
        "sources": ["source_000013"],
        "hyperparameters": ["lag_length", "neighbor_count", "distance_metric"],
    },
    {
        "name": "dtw_analogue_forecast",
        "category": "analogue",
        "description": "Find historical subsequences with similar shape under dynamic time warping and aggregate their following paths.",
        "assumption": "Shape-similar episodes remain predictive despite local timing distortions.",
        "failure": "A permissive warping window can match unrelated episodes and erase timing information.",
        "sources": ["source_000013", "source_000016"],
        "hyperparameters": ["window_length", "warping_window", "neighbor_count"],
    },
    {
        "name": "split_conformal_residual_intervals",
        "category": "calibration",
        "description": "Calibrate forecast intervals with held-out absolute residual quantiles from a fixed point forecaster.",
        "assumption": "Calibration and forecast residuals are sufficiently exchangeable over the evaluation period.",
        "failure": "Abrupt distribution shifts invalidate residual coverage learned from the calibration window.",
        "sources": ["source_000014"],
        "hyperparameters": ["miscoverage_level", "calibration_window"],
        "probabilistic": True,
    },
    {
        "name": "adaptive_conformal_time_series",
        "category": "calibration",
        "description": "Update conformal interval widths online so recent coverage errors change the next interval.",
        "assumption": "Coverage drift is gradual enough for online score updates to track it.",
        "failure": "Fast alternating regimes cause the adaptation rule to lag or oscillate.",
        "sources": ["source_000014"],
        "hyperparameters": ["miscoverage_level", "adaptation_rate"],
        "probabilistic": True,
    },
    {
        "name": "poisson_dynamic_regression",
        "category": "count_forecasting",
        "description": "Model a count target with a log-linked Poisson mean driven by time and optional predictors.",
        "assumption": "Conditional count variance is close to the conditional mean after included predictors.",
        "failure": "Strong overdispersion, zero inflation, or dependence left in residuals biases uncertainty.",
        "sources": ["source_000001"],
        "hyperparameters": ["lag_features", "regularization"],
        "covariates": True,
        "probabilistic": True,
    },
    {
        "name": "negative_binomial_dynamic_regression",
        "category": "count_forecasting",
        "description": "Forecast overdispersed counts with a dynamic negative-binomial regression mean.",
        "assumption": "A negative-binomial conditional law captures extra-Poisson dispersion.",
        "failure": "Structural zeros or changing dispersion violate the fitted conditional distribution.",
        "sources": ["source_000001"],
        "hyperparameters": ["dispersion", "lag_features"],
        "covariates": True,
        "probabilistic": True,
    },
    {
        "name": "integer_autoregressive_inar",
        "category": "count_forecasting",
        "description": "Use binomial thinning of past integer counts plus an integer-valued innovation process.",
        "assumption": "Count dependence can be represented by stable thinning probabilities and innovations.",
        "failure": "Nonlinear regime changes or strong covariate effects are not represented by a fixed INAR law.",
        "sources": ["source_000001"],
        "hyperparameters": ["order", "innovation_distribution"],
        "probabilistic": True,
    },
    {
        "name": "gaussian_process_autoregression",
        "category": "kernel",
        "description": "Place a Gaussian-process prior over the nonlinear mapping from lag vectors to future values.",
        "assumption": "The chosen kernel expresses the smoothness and similarity structure of the lag-response map.",
        "failure": "Poor kernels extrapolate badly and exact inference scales cubically with training cases.",
        "sources": ["source_000015", "source_000016"],
        "hyperparameters": ["kernel", "lag_length", "noise_level"],
        "probabilistic": True,
    },
    {
        "name": "support_vector_lag_regression",
        "category": "kernel",
        "description": "Fit epsilon-insensitive support-vector regression to lag and calendar features and roll it forward.",
        "assumption": "A fixed kernel can represent the stable nonlinear relationship between features and target.",
        "failure": "Recursive multi-step use compounds error and kernel tuning can be unstable on short series.",
        "sources": ["source_000016", "source_000029"],
        "hyperparameters": ["kernel", "C", "epsilon", "lag_features"],
        "covariates": True,
    },
    {
        "name": "kernel_ridge_lag_regression",
        "category": "kernel",
        "description": "Apply ridge-regularized kernel regression to lagged and optional exogenous features.",
        "assumption": "Future responses vary smoothly under the selected kernel geometry.",
        "failure": "Kernel matrices become costly on large samples and extrapolation beyond observed features is weak.",
        "sources": ["source_000015", "source_000016"],
        "hyperparameters": ["kernel", "ridge_penalty", "lag_features"],
        "covariates": True,
    },
    {
        "name": "neural_network_autoregression",
        "category": "neural",
        "description": "Feed lagged observations and seasonal lags to a shallow multilayer perceptron and recursively forecast.",
        "assumption": "A stable nonlinear lag relationship can be learned from enough repeated patterns.",
        "failure": "Small samples, regime shifts, and recursive error accumulation degrade long-horizon forecasts.",
        "sources": ["source_000001"],
        "hyperparameters": ["lag_order", "hidden_units", "weight_decay"],
    },
    {
        "name": "deepar",
        "category": "neural",
        "description": "Train one autoregressive recurrent network across related series to output a parametric predictive distribution.",
        "assumption": "Related training series share learnable dynamics and the chosen likelihood is suitable.",
        "failure": "Few unrelated series or a misspecified likelihood prevent useful global transfer.",
        "sources": ["source_000017"],
        "hyperparameters": ["context_length", "rnn_layers", "likelihood"],
        "covariates": True,
        "probabilistic": True,
    },
    {
        "name": "nbeats",
        "category": "neural",
        "description": "Stack backward and forward fully connected residual blocks with optional trend and seasonal bases.",
        "assumption": "A large collection of training windows supports learning reusable basis expansions.",
        "failure": "Very small samples or abrupt unseen regimes cause high-variance extrapolation.",
        "sources": ["source_000018"],
        "hyperparameters": ["stack_types", "blocks", "hidden_units"],
    },
    {
        "name": "nhits",
        "category": "neural",
        "description": "Use hierarchical interpolation and multi-rate input pooling to model different forecast frequencies.",
        "assumption": "Forecast structure can be decomposed across multiple temporal resolutions.",
        "failure": "Resolution choices can suppress sharp high-frequency events important to the target.",
        "sources": ["source_000019"],
        "hyperparameters": ["pooling_sizes", "frequency_downsample", "blocks"],
    },
    {
        "name": "temporal_fusion_transformer",
        "category": "neural",
        "description": "Combine recurrent local processing, variable selection, gating, and attention for multi-horizon forecasts with covariates.",
        "assumption": "Known-future and observed covariates contain stable cross-series predictive information.",
        "failure": "Sparse data or unreliable future covariates make the high-capacity architecture overfit.",
        "sources": ["source_000020"],
        "hyperparameters": ["hidden_size", "attention_heads", "dropout"],
        "covariates": True,
        "probabilistic": True,
    },
    {
        "name": "intervention_arima",
        "category": "outlier_handling",
        "description": "Jointly estimate ARIMA dynamics and explicit pulse, level-shift, or temporary-change intervention effects.",
        "assumption": "Outlier timing or intervention shape can be identified separately from recurring dynamics.",
        "failure": "Many overlapping anomalies are confounded with structural breaks and ARIMA parameters.",
        "sources": ["source_000021"],
        "hyperparameters": ["intervention_types", "arima_order"],
    },
    {
        "name": "outlier_adjusted_arima",
        "category": "outlier_handling",
        "description": "Detect additive and innovative outliers, estimate their effects, clean the series, and refit ARIMA.",
        "assumption": "A sparse set of abnormal observations contaminates an otherwise stable ARIMA process.",
        "failure": "Persistent regime changes are incorrectly removed as isolated contamination.",
        "sources": ["source_000021"],
        "hyperparameters": ["outlier_threshold", "outlier_types", "arima_order"],
    },
    {
        "name": "markov_switching_autoregression",
        "category": "regime_switching",
        "description": "Use a latent Markov state to switch autoregressive parameters, level, or variance across regimes.",
        "assumption": "A small set of persistent latent regimes generates the observed dynamics.",
        "failure": "Regimes with few visits are weakly identified and unprecedented states cannot be represented.",
        "sources": ["source_000022"],
        "hyperparameters": ["regime_count", "ar_order", "switching_variance"],
        "probabilistic": True,
    },
    {
        "name": "threshold_autoregression",
        "category": "regime_switching",
        "description": "Switch autoregressive equations when a threshold variable crosses estimated cut points.",
        "assumption": "Regime transitions are driven by observable threshold conditions.",
        "failure": "Smooth transitions or time-varying thresholds make hard partitions unstable.",
        "sources": ["source_000023"],
        "hyperparameters": ["threshold_variable", "thresholds", "ar_orders"],
    },
    {
        "name": "smooth_transition_autoregression",
        "category": "regime_switching",
        "description": "Blend autoregressive regimes continuously with a logistic or exponential transition function.",
        "assumption": "Nonlinear dynamics vary smoothly with an observable transition variable.",
        "failure": "Limited observations near the transition region make slope and threshold estimates unstable.",
        "sources": ["source_000024"],
        "hyperparameters": ["transition_function", "threshold", "smoothness", "ar_order"],
    },
    {
        "name": "bats",
        "category": "seasonal",
        "description": "Combine Box-Cox transformation, ARMA errors, trend, and one seasonal component in state space.",
        "assumption": "Transformed seasonal dynamics and residual autocorrelation remain stable.",
        "failure": "Multiple or rapidly changing seasonalities are not captured by a single seasonal state.",
        "sources": ["source_000025"],
        "hyperparameters": ["boxcox", "arma_errors", "damped_trend", "season_length"],
        "probabilistic": True,
    },
    {
        "name": "tbats",
        "category": "seasonal",
        "description": "Extend BATS with trigonometric states so multiple and non-integer seasonal periods can be modeled.",
        "assumption": "Complex seasonalities can be represented by stable Fourier state components.",
        "failure": "Long histories and many harmonics make fitting expensive and can overfit weak cycles.",
        "sources": ["source_000025"],
        "hyperparameters": ["seasonal_periods", "harmonics", "boxcox", "arma_errors"],
        "probabilistic": True,
    },
    {
        "name": "dynamic_harmonic_regression_arima",
        "category": "seasonal",
        "description": "Regress on Fourier terms for one or more seasonal periods while modeling remaining errors with ARIMA.",
        "assumption": "Seasonal shapes are smooth and repeat while residual autocorrelation follows ARIMA dynamics.",
        "failure": "Changing seasonal timing or event-driven pulses cannot be represented by fixed harmonics.",
        "sources": ["source_000001"],
        "hyperparameters": ["fourier_orders", "seasonal_periods", "arima_order"],
    },
    {
        "name": "mstl_ets",
        "category": "seasonal",
        "description": "Decompose multiple seasonal periods with repeated STL, forecast the adjusted series with ETS, and restore each seasonal component.",
        "assumption": "Several stable seasonal periods coexist and can be separated additively.",
        "failure": "Short histories or interacting seasonal amplitudes make the sequential decomposition unreliable.",
        "sources": ["source_000002", "source_000010"],
        "hyperparameters": ["seasonal_periods", "stl_windows", "ets_model"],
        "probabilistic": True,
    },
    {
        "name": "random_forest_lag_forecast",
        "category": "tree",
        "description": "Train an ensemble of randomized regression trees on lag, rolling, calendar, and optional exogenous features.",
        "assumption": "Historical feature-target relationships recur within the range covered by training data.",
        "failure": "Trees do not extrapolate trends beyond observed target regions and recursive forecasts compound errors.",
        "sources": ["source_000016", "source_000026"],
        "hyperparameters": ["lag_features", "tree_count", "max_depth"],
        "covariates": True,
    },
    {
        "name": "xgboost_lag_forecast",
        "category": "tree",
        "description": "Boost regularized regression trees over lag, rolling, calendar, and exogenous features.",
        "assumption": "Engineered features expose repeatable nonlinear splits predictive of future targets.",
        "failure": "Extrapolation and unseen regimes remain weak even when in-range fit is strong.",
        "sources": ["source_000016", "source_000027"],
        "hyperparameters": ["lag_features", "depth", "learning_rate", "estimators"],
        "covariates": True,
    },
    {
        "name": "lightgbm_lag_forecast",
        "category": "tree",
        "description": "Fit histogram-based leaf-wise gradient-boosted trees to time-series lag and covariate features.",
        "assumption": "Large enough supervised windows support stable nonlinear partitions of engineered features.",
        "failure": "Leaf-wise growth overfits small series and cannot reliably extrapolate outside training targets.",
        "sources": ["source_000016", "source_000028"],
        "hyperparameters": ["lag_features", "num_leaves", "learning_rate", "estimators"],
        "covariates": True,
    },
)


DEPTH_EXPANSION_STATISTICAL_METHOD_SPECS: tuple[Mapping[str, object], ...] = (
    {
        "name": "bayesian_online_changepoint_forecast", "category": "change_point",
        "description": "Maintain a posterior over the current run length and forecast from the regime implied by that posterior.",
        "assumption": "Abrupt changes reset locally stable predictive parameters with a meaningful hazard prior.",
        "failure": "Gradual drift or a misspecified hazard spreads posterior mass across misleading run lengths.",
        "sources": ["source_000061"], "hyperparameters": ["hazard_rate", "predictive_model"], "probabilistic": True,
    },
    {
        "name": "pelt_segment_then_forecast", "category": "change_point",
        "description": "Use PELT to locate an optimal penalized segmentation and fit the forecasting model only to the latest regime.",
        "assumption": "The most recent detected segment is long enough and more relevant than older regimes.",
        "failure": "Penalty errors create a tiny final segment or retain obsolete observations after a missed break.",
        "sources": ["source_000062"], "hyperparameters": ["cost_function", "penalty", "minimum_segment_length"],
    },
    {
        "name": "robust_stl_ets", "category": "robust",
        "description": "Apply robust STL weights, forecast the seasonally adjusted component with ETS, and restore seasonality.",
        "assumption": "A small fraction of observations are contaminated while trend and seasonality remain locally smooth.",
        "failure": "Persistent level shifts are downweighted as outliers instead of becoming the new forecast regime.",
        "sources": ["source_000001", "source_000010"], "hyperparameters": ["stl_window", "robust_iterations", "ets_model"],
    },
    {
        "name": "median_seasonal_profile_forecast", "category": "robust",
        "description": "Estimate each seasonal position by a median across cycles and extrapolate the robust profile around a recent median level.",
        "assumption": "Seasonal phase is stable and fewer than half of comparable cycles are contaminated.",
        "failure": "Changing seasonal amplitude or phase makes the historical median profile obsolete.",
        "sources": ["source_000001"], "hyperparameters": ["season_length", "cycles", "level_window"],
    },
    {
        "name": "state_space_trigonometric_harmonics", "category": "spectral",
        "description": "Represent periodic behavior with stochastic trigonometric state pairs whose amplitudes evolve through a state-space model.",
        "assumption": "A small set of harmonic frequencies explains recurring behavior while amplitudes change gradually.",
        "failure": "Irregular event timing or rapidly changing frequencies cannot be tracked by fixed harmonic states.",
        "sources": ["source_000003", "source_000025"], "hyperparameters": ["periods", "harmonics", "state_variance"], "probabilistic": True,
    },
    {
        "name": "quantile_regression_forecast", "category": "probabilistic",
        "description": "Fit separate conditional quantile regressions over lagged, calendar, and optional exogenous predictors.",
        "assumption": "Conditional quantiles are stable functions of the supplied predictors.",
        "failure": "Sparse tail observations and independently fitted levels can produce unstable or crossing quantiles.",
        "sources": ["source_000079"], "hyperparameters": ["quantile_levels", "lag_features", "regularization"], "covariates": True, "probabilistic": True,
    },
    {
        "name": "forecast_residual_bootstrap", "category": "probabilistic",
        "description": "Simulate future paths by adding resampled historical forecast residuals to recursive point forecasts.",
        "assumption": "Residual blocks approximate the dependence and scale of future forecast errors.",
        "failure": "Heteroscedasticity or regime change makes historical residual draws miscalibrated.",
        "sources": ["source_000001"], "hyperparameters": ["bootstrap_paths", "block_length", "residual_window"], "probabilistic": True,
    },
    {
        "name": "bottom_up_reconciliation", "category": "reconciliation",
        "description": "Forecast the bottom-level series and aggregate them through the hierarchy summing matrix.",
        "assumption": "Bottom-level forecasts are reliable enough to determine every aggregate.",
        "failure": "Noisy disaggregated series propagate high variance into all upper levels.",
        "sources": ["source_000063"], "hyperparameters": ["bottom_level_models"],
    },
    {
        "name": "top_down_historical_proportions", "category": "reconciliation",
        "description": "Forecast the total and disaggregate it using proportions estimated from historical observations.",
        "assumption": "Historical component shares persist through the forecast horizon.",
        "failure": "Structural share shifts allocate an accurate total to the wrong bottom series.",
        "sources": ["source_000063"], "hyperparameters": ["proportion_window", "averaging_rule"],
    },
    {
        "name": "top_down_forecast_proportions", "category": "reconciliation",
        "description": "Disaggregate an aggregate forecast using proportions derived from independently generated base forecasts.",
        "assumption": "Relative base forecasts contain useful allocation information even when they are incoherent.",
        "failure": "Biased or near-zero component base forecasts create unstable proportions.",
        "sources": ["source_000063"], "hyperparameters": ["proportion_rule", "zero_guard"],
    },
    {
        "name": "middle_out_reconciliation", "category": "reconciliation",
        "description": "Forecast a selected middle level, aggregate upward, and disaggregate downward with estimated proportions.",
        "assumption": "The chosen middle level balances signal strength and allocation stability.",
        "failure": "A poor middle-level choice inherits both noisy forecasts and unstable disaggregation.",
        "sources": ["source_000063"], "hyperparameters": ["middle_level", "proportion_rule"],
    },
    {
        "name": "ols_projection_reconciliation", "category": "reconciliation",
        "description": "Project incoherent base forecasts onto the coherent subspace using ordinary least squares.",
        "assumption": "Base forecast errors have comparable, uncorrelated variance after scaling.",
        "failure": "Strongly unequal or correlated errors make the unweighted projection inefficient.",
        "sources": ["source_000063"], "hyperparameters": ["scaling"],
    },
    {
        "name": "wls_structural_reconciliation", "category": "reconciliation",
        "description": "Reconcile base forecasts with diagonal weights derived from hierarchy structure or forecast-error variances.",
        "assumption": "Diagonal weights capture the dominant reliability differences across nodes.",
        "failure": "Ignoring cross-series error covariance loses important offsetting information.",
        "sources": ["source_000063"], "hyperparameters": ["weight_estimator"],
    },
    {
        "name": "mint_shrinkage_reconciliation", "category": "reconciliation",
        "description": "Use a shrinkage estimate of the full base-error covariance in minimum-trace forecast reconciliation.",
        "assumption": "Historical base forecast errors estimate future cross-series dependence after regularization.",
        "failure": "Few origins or changing dependence make even the shrunk covariance misleading.",
        "sources": ["source_000065"], "hyperparameters": ["covariance_shrinkage", "error_origins"],
    },
    {
        "name": "temporal_hierarchy_reconciliation", "category": "reconciliation",
        "description": "Forecast multiple non-overlapping temporal aggregation levels and reconcile them to one coherent path.",
        "assumption": "Different temporal scales offer complementary information and share stable aggregation constraints.",
        "failure": "Misaligned calendars or scale-specific structural breaks invalidate the temporal hierarchy.",
        "sources": ["source_000064"], "hyperparameters": ["aggregation_levels", "reconciliation_matrix"],
    },
    {
        "name": "ltsf_dlinear", "category": "neural",
        "description": "Decompose the input into trend and remainder and map each component directly to the forecast with linear layers.",
        "assumption": "A fixed linear projection of decomposed history captures the target horizon.",
        "failure": "Nonlinear conditional dynamics or new regimes cannot be represented by the fixed projection.",
        "sources": ["source_000066"], "hyperparameters": ["lookback", "moving_average_window", "individual_channels"],
    },
    {
        "name": "informer", "category": "neural",
        "description": "Use ProbSparse attention, attention distillation, and a one-shot generative decoder for long-sequence forecasting.",
        "assumption": "A small set of dominant attention interactions captures long-range dependencies.",
        "failure": "Diffuse dependencies or limited training data make sparse attention discard useful signals.",
        "sources": ["source_000073"], "hyperparameters": ["lookback", "attention_factor", "encoder_layers"], "covariates": True,
    },
    {
        "name": "autoformer", "category": "neural",
        "description": "Embed progressive decomposition and sub-series auto-correlation blocks inside a Transformer architecture.",
        "assumption": "Periodic sub-series dependencies and decomposable trend-seasonal structure persist.",
        "failure": "Event-driven or aperiodic dynamics undermine auto-correlation aggregation.",
        "sources": ["source_000068"], "hyperparameters": ["lookback", "moving_average_window", "attention_factor"], "covariates": True,
    },
    {
        "name": "fedformer", "category": "neural",
        "description": "Combine seasonal-trend decomposition with sparse Fourier or wavelet-domain Transformer operations.",
        "assumption": "The target has a compact frequency-domain representation transferable across windows.",
        "failure": "Sharp transient events spread energy across frequencies and are poorly reconstructed from sparse modes.",
        "sources": ["source_000069"], "hyperparameters": ["lookback", "frequency_modes", "transform"], "covariates": True,
    },
    {
        "name": "patchtst", "category": "neural",
        "description": "Tokenize each variable into temporal patches and apply a channel-independent Transformer across patches.",
        "assumption": "Local patch structure transfers and channel independence does not discard essential interactions.",
        "failure": "Strong cross-variable causal structure or badly chosen patch sizes reduces accuracy.",
        "sources": ["source_000067"], "hyperparameters": ["lookback", "patch_length", "stride"],
    },
    {
        "name": "itransformer", "category": "neural",
        "description": "Invert standard tokenization so complete variable histories become tokens and attention models cross-variate relations.",
        "assumption": "Cross-variable relationships are stable and full-history variable embeddings preserve predictive temporal information.",
        "failure": "Very many variables or unstable cross-series relations make variate attention expensive or misleading.",
        "sources": ["source_000070"], "hyperparameters": ["lookback", "model_dimension", "attention_heads"], "covariates": True,
    },
    {
        "name": "timesnet", "category": "neural",
        "description": "Reshape one-dimensional histories into multiple period-aligned two-dimensional tensors and model them with inception blocks.",
        "assumption": "A small number of discoverable periods organizes intra-period and inter-period variation.",
        "failure": "Weak or drifting periodicity yields unstable two-dimensional reshaping.",
        "sources": ["source_000071"], "hyperparameters": ["top_periods", "inception_kernels", "blocks"],
    },
    {
        "name": "tsmixer", "category": "neural",
        "description": "Stack MLP blocks that alternately mix information across time and feature dimensions.",
        "assumption": "Time and feature mixing through dense projections captures the relevant multivariate dynamics.",
        "failure": "Long irregular dependencies or small training sets make dense mixers overfit.",
        "sources": ["source_000074"], "hyperparameters": ["lookback", "mixer_blocks", "hidden_dimension"], "covariates": True,
    },
    {
        "name": "tide", "category": "neural",
        "description": "Encode history and covariates with residual MLP blocks and decode the full long-horizon forecast directly.",
        "assumption": "Dense nonlinear feature maps and known covariates are sufficient without explicit attention.",
        "failure": "Unobserved long-range dependencies or unreliable future covariates produce systematic errors.",
        "sources": ["source_000075"], "hyperparameters": ["lookback", "hidden_dimension", "residual_blocks"], "covariates": True,
    },
    {
        "name": "scinet", "category": "neural",
        "description": "Recursively downsample the sequence, convolve each subsequence, and exchange information across resolutions.",
        "assumption": "Downsampled subsequences preserve complementary temporal relations at multiple resolutions.",
        "failure": "Aliasing or irregular sampling breaks the interleaved multi-resolution representation.",
        "sources": ["source_000076"], "hyperparameters": ["levels", "kernel_size", "stacks"],
    },
    {
        "name": "timemixer", "category": "neural",
        "description": "Decompose multi-scale histories, mix seasonal and trend information across scales, and combine scale-specific predictors.",
        "assumption": "Fine and coarse sampling scales contain complementary stable patterns.",
        "failure": "Scale construction smears sharp events or aliases irregular cycles.",
        "sources": ["source_000072"], "hyperparameters": ["downsampling_layers", "decomposition", "predictor_count"],
    },
    {
        "name": "samformer", "category": "neural",
        "description": "Train a shallow channel-attention Transformer with sharpness-aware minimization for multivariate forecasting.",
        "assumption": "Flat-minimum optimization improves generalization of a compact cross-channel attention model.",
        "failure": "Optimization cost or unstable channel relations remove the expected generalization benefit.",
        "sources": ["source_000077"], "hyperparameters": ["lookback", "attention_dimension", "sam_radius"], "covariates": True,
    },
)

FOUNDATION_METHOD_SPECS: tuple[Mapping[str, object], ...] = (
    {
        "name": "TimesFM 1.0",
        "category": "zero_shot",
        "description": "A patched decoder-only transformer pretrained across heterogeneous series for direct point forecasting.",
        "assumption": "Pretraining patterns transfer to the target frequency and forecast horizon.",
        "failure": "Large domain shifts or required context beyond the checkpoint limit reduce zero-shot reliability.",
        "definition_sources": ["source_000030"], "implementation_sources": ["source_000031"],
        "metadata": ["google/timesfm-1.0-200m", "1.0", 512, 128, "zero_shot", False, False, "CPU or accelerator", "Apache-2.0", True, True],
    },
    {
        "name": "Chronos T5",
        "category": "probabilistic_tsfm",
        "description": "Scale and quantize values into tokens, then sample continuations from a pretrained T5 language-model family.",
        "assumption": "Quantized sequence patterns learned from public and synthetic corpora transfer to the target.",
        "failure": "Quantization, unmodeled covariates, and autoregressive sampling can miss sharp conditional events.",
        "definition_sources": ["source_000032"], "implementation_sources": ["source_000033"],
        "metadata": ["amazon/chronos-t5-base", "original", 512, "autoregressive", "zero_shot_sampling", True, False, "CPU or accelerator", "Apache-2.0", True, True],
    },
    {
        "name": "Moirai 1.x",
        "category": "covariate_tsfm",
        "description": "A masked encoder universal forecaster using multi-patch-size projections and an any-variate attention design.",
        "assumption": "Cross-frequency and cross-variate patterns in LOTSA transfer to the target variables.",
        "failure": "Unseen covariate semantics or data distributions can defeat the universal patch representation.",
        "definition_sources": ["source_000034"], "implementation_sources": ["source_000035"],
        "metadata": ["Salesforce/moirai-1.1-R-base", "1.1", "configurable", "configurable", "zero_shot_distribution", True, True, "CPU or accelerator", "Apache-2.0", True, True],
    },
    {
        "name": "Lag-Llama",
        "category": "probabilistic_tsfm",
        "description": "A decoder-only probabilistic forecaster that conditions on lagged values and time features across frequencies.",
        "assumption": "Lag covariates provide a frequency-robust representation transferable across domains.",
        "failure": "Long nonseasonal dependencies and unseen exogenous shocks are poorly represented by fixed lag features.",
        "definition_sources": ["source_000036", "source_000037"], "implementation_sources": ["source_000037"],
        "metadata": ["time-series-foundation-models/Lag-Llama", "2024 release", "checkpoint_config", "autoregressive", "zero_shot_or_finetuned_sampling", True, False, "CPU or accelerator", "Apache-2.0", True, True],
    },
    {
        "name": "MOMENT-1",
        "category": "fine_tuned",
        "description": "A masked-patch encoder pretrained on the Time-series Pile and adapted with task-specific forecasting heads.",
        "assumption": "Pretrained representations can be adapted with a small amount of task supervision.",
        "failure": "A fixed patch context and weak target-specific adaptation can miss long-horizon dynamics.",
        "definition_sources": ["source_000038"], "implementation_sources": ["source_000039"],
        "metadata": ["AutonLab/MOMENT-1-large", "1", 512, "task_configured", "linear_probe_or_finetune", False, False, "GPU recommended", "MIT", True, True],
    },
    {
        "name": "Granite Tiny Time Mixer R2",
        "category": "fine_tuned",
        "description": "A compact adaptive-patching mixer pretrained for zero-shot use and lightweight target or covariate adaptation.",
        "assumption": "Resolution-diverse pretraining transfers and a small adaptation head captures target specifics.",
        "failure": "Checkpoint-specific context and horizon pairs constrain direct use on mismatched tasks.",
        "definition_sources": ["source_000040", "source_000041"], "implementation_sources": ["source_000041"],
        "metadata": ["ibm-granite/granite-timeseries-ttm-r2", "r2", "checkpoint_dependent", "checkpoint_dependent", "zero_shot_or_finetuned", False, "fine_tuning", "CPU or accelerator", "Apache-2.0", True, True],
    },
    {
        "name": "Timer",
        "category": "fine_tuned",
        "description": "A GPT-style model pretrained by next-patch prediction and adapted to forecasting and other generative time-series tasks.",
        "assumption": "Next-patch generative pretraining learns patterns reusable under the target sampling process.",
        "failure": "Channel-independent generation and finite context can miss new cross-variate causal structure.",
        "definition_sources": ["source_000042"], "implementation_sources": ["source_000043"],
        "metadata": ["thuml/timer-base", "original", "checkpoint_dependent", "autoregressive", "zero_shot_or_finetuned", False, False, "GPU recommended", "research release", True, True],
    },
    {
        "name": "Time-MoE",
        "category": "zero_shot",
        "description": "A sparse mixture-of-experts decoder pretrained autoregressively on the Time-300B corpus.",
        "assumption": "Sparse experts specialize in reusable temporal regimes across large heterogeneous pretraining data.",
        "failure": "Unsupported dynamic covariates and context-plus-horizon beyond the trained window limit applicability.",
        "definition_sources": ["source_000044"], "implementation_sources": ["source_000045"],
        "metadata": ["Maple728/TimeMoE-200M", "2024", 4096, "context_plus_horizon_up_to_4096", "zero_shot_autoregressive", False, False, "CPU or accelerator; GPU recommended", "Apache-2.0", True, True],
    },
    {
        "name": "ForecastPFN",
        "category": "zero_shot",
        "description": "A prior-data-fitted network trained on synthetic generators to approximate Bayesian zero-shot forecasting.",
        "assumption": "The synthetic prior assigns meaningful probability to the target data-generating behavior.",
        "failure": "Real dynamics outside the synthetic training prior lead to systematic zero-shot errors.",
        "definition_sources": ["source_000046"], "implementation_sources": ["source_000047"],
        "metadata": ["abacusai/ForecastPFN", "NeurIPS-2023", "model_config", "model_config", "zero_shot", True, False, "CPU or accelerator", "repository license", True, True],
    },
    {
        "name": "TimeGPT-1",
        "category": "covariate_tsfm",
        "description": "A hosted pretrained forecasting service supporting zero-shot forecasts, intervals, and exogenous inputs.",
        "assumption": "The proprietary pretraining corpus and service adaptation transfer to the submitted target.",
        "failure": "Closed weights and training data limit diagnosis under domain shift or service changes.",
        "definition_sources": ["source_000048", "source_000049"], "implementation_sources": ["source_000049"],
        "metadata": ["Nixtla TimeGPT API", "service-current-2026-08-17", "service_managed", "service_managed", "zero_shot_api", True, True, "hosted API", "proprietary service", False, False],
    },
    {
        "name": "TEMPO",
        "category": "zero_shot",
        "description": "A prompt-conditioned pretrained transformer that decomposes trend, seasonality, and residual structure for transfer.",
        "assumption": "Prompt and decomposition components align the target distribution with pretrained regimes.",
        "failure": "Incorrect decomposition or a target outside prompt coverage produces negative transfer.",
        "definition_sources": ["source_000050"], "implementation_sources": [],
        "metadata": ["TEMPO research checkpoint", "2023", "checkpoint_config", "checkpoint_config", "zero_shot_prompting", False, "multimodal_context", "GPU recommended", "research release", True, True],
    },
    {
        "name": "UniTS",
        "category": "fine_tuned",
        "description": "A unified task-tokenized time-series model that supports forecasting, imputation, classification, and anomaly detection.",
        "assumption": "A shared sequence-variable representation transfers across tasks and domains with prompting or adaptation.",
        "failure": "Task and domain shifts not represented during pretraining require substantial target-specific tuning.",
        "definition_sources": ["source_000051"], "implementation_sources": ["source_000052"],
        "metadata": ["mims-harvard/UniTS", "NeurIPS-2024", "task_configured", "task_configured", "zero_few_shot_or_finetuned", False, True, "GPU recommended", "repository license", True, True],
    },
    {
        "name": "Sundial",
        "category": "probabilistic_tsfm",
        "description": "A native continuous-value transformer trained with flow matching to generate multiple future paths.",
        "assumption": "Flow-matching pretraining on TimeBench captures a transferable conditional future distribution.",
        "failure": "Unseen domains or event-conditioned futures absent from numeric context remain underrepresented.",
        "definition_sources": ["source_000053"], "implementation_sources": ["source_000054"],
        "metadata": ["thuml/sundial-base-128m", "2025", "arbitrary_with_model_limit", "generative", "zero_shot_sampling", True, False, "GPU recommended", "research release", True, True],
    },
    {
        "name": "Toto 2.0",
        "category": "covariate_tsfm",
        "description": "A multivariate decoder-only probabilistic forecaster scaled for observability and general forecasting data.",
        "assumption": "Large observability and synthetic pretraining transfers to the target variables and covariates.",
        "failure": "Targets far from observability dynamics or hardware limits for larger checkpoints reduce usefulness.",
        "definition_sources": ["source_000055"], "implementation_sources": ["source_000056"],
        "metadata": ["Datadog/Toto-2.0-313m", "2.0", "variable", "variable", "zero_shot_probabilistic", True, True, "GPU recommended", "Apache-2.0", True, True],
    },
)


DEPTH_EXPANSION_FOUNDATION_METHOD_SPECS: tuple[Mapping[str, object], ...] = (
    {
        "name": "Timer-S1",
        "category": "zero_shot",
        "description": "An 8.3B-parameter sparse mixture-of-experts foundation model trained for serial-token time-series prediction with an 11.5K context.",
        "assumption": "Serial scaling over architecture, data, and training transfers to the target series and horizon.",
        "failure": "Numeric history alone cannot resolve unseen event-driven futures or unsupported target semantics.",
        "definition_sources": ["source_000078"],
        "implementation_sources": ["source_000043"],
        "metadata": ["thuml/Timer-S1", "2026-v3", 11520, "serial_generation", "zero_shot", True, False, "GPU recommended", "research release", True, True],
    },
    {
        "name": "Chronos-2",
        "category": "covariate_tsfm",
        "description": "A zero-shot universal forecaster using group attention for in-context information sharing among related series, variates, targets, and covariates.",
        "assumption": "Synthetic multivariate structures teach group attention relationships that transfer to target and covariate groups.",
        "failure": "Unseen group semantics or future events absent from the provided numeric covariates remain unresolved.",
        "definition_sources": ["source_000080"],
        "implementation_sources": ["source_000033"],
        "metadata": ["amazon/chronos-2", "2025", "model_config", "model_config", "zero_shot_in_context", True, True, "CPU or accelerator; GPU recommended", "Apache-2.0", True, True],
    },
    {
        "name": "Moirai 2.0",
        "category": "probabilistic_tsfm",
        "description": "A decoder-only single-patch foundation model trained with quantile loss and recursive multi-token prediction.",
        "assumption": "The 36-million-series corpus and direct quantile objective transfer to the target distribution.",
        "failure": "Performance declines at long horizons or under domains outside the pretraining corpus.",
        "definition_sources": ["source_000081"],
        "implementation_sources": ["source_000035"],
        "metadata": ["Salesforce/moirai-2.0", "2.0", "model_config", "recursive_multi_token", "zero_shot_quantiles", True, False, "CPU or accelerator", "Apache-2.0", True, True],
    },
)

COMBINED_METHOD_SPECS: tuple[Mapping[str, object], ...] = (
    {
        "name": "median_forecast_combination", "category": "ensemble",
        "description": "Take the pointwise median of forecasts from heterogeneous component methods.",
        "assumption": "At least half of the component forecasts remain reasonably centered.",
        "failure": "A majority of similarly biased components moves the median in the wrong direction.",
        "sources": ["source_000011"],
        "parents": ["method_seed_0012", "method_seed_0018", "method_seed_0014"],
        "hyperparameters": [],
    },
    {
        "name": "inverse_error_weighted_combination", "category": "ensemble",
        "description": "Weight component forecasts inversely to their rolling-origin validation errors.",
        "assumption": "Recent relative validation performance persists into the forecast window.",
        "failure": "Noisy short validation windows assign extreme weight to accidental winners.",
        "sources": ["source_000057", "source_000011"],
        "parents": ["method_seed_0012", "method_seed_0018", "method_seed_0014"],
        "hyperparameters": ["validation_window", "weight_power"],
    },
    {
        "name": "covariance_optimal_forecast_combination", "category": "ensemble",
        "description": "Estimate component error covariance and choose minimum-variance linear combination weights.",
        "assumption": "The estimated error covariance is stable and component forecasts are approximately unbiased.",
        "failure": "Ill-conditioned covariance estimates produce unstable or extreme weights.",
        "sources": ["source_000057"],
        "parents": ["method_seed_0012", "method_seed_0018"],
        "hyperparameters": ["covariance_regularization"],
    },
    {
        "name": "cross_validated_stacked_forecast", "category": "ensemble",
        "description": "Train a second-level regressor on out-of-fold component forecasts to produce the final forecast.",
        "assumption": "Out-of-fold component error relationships transfer to the future window.",
        "failure": "Leakage or too few folds causes the meta-regressor to overfit component noise.",
        "sources": ["source_000058", "source_000011"],
        "parents": ["method_seed_0012", "method_seed_0018", "method_tsfm_0001"],
        "hyperparameters": ["folds", "meta_regressor", "weight_constraints"],
    },
    {
        "name": "quantile_forecast_averaging", "category": "ensemble",
        "description": "Average corresponding predictive quantiles from multiple probabilistic forecasters.",
        "assumption": "Component quantiles share the same probability levels and are individually ordered.",
        "failure": "Miscalibrated or dependent components can preserve bias and yield incoherent quantile curves.",
        "sources": ["source_000011"],
        "parents": ["method_tsfm_0002", "method_tsfm_0003", "method_tsfm_0013"],
        "hyperparameters": ["quantile_levels", "component_weights"],
        "probabilistic": True,
    },
    {
        "name": "rolling_origin_champion_selector", "category": "selector",
        "description": "Select the single component with the lowest rolling-origin validation loss for the target horizon.",
        "assumption": "Historical forecast-origin errors rank candidate methods similarly to the next origin.",
        "failure": "A regime change after the last validation origin makes the historical champion obsolete.",
        "sources": ["source_000001", "source_000011"],
        "parents": ["method_seed_0004", "method_seed_0012", "method_tsfm_0001"],
        "hyperparameters": ["folds", "loss", "tie_break"],
    },
    {
        "name": "fforma_feature_weighted_selector", "category": "selector",
        "description": "Map time-series features to learned nonnegative weights over a pool of forecasting methods.",
        "assumption": "A representative reference collection links series features to relative model performance.",
        "failure": "Targets outside the reference feature distribution receive unreliable weights.",
        "sources": ["source_000059"],
        "parents": ["method_seed_0012", "method_seed_0018", "method_seed_0014"],
        "hyperparameters": ["feature_set", "meta_model", "loss"],
    },
    {
        "name": "per_horizon_model_selector", "category": "selector",
        "description": "Choose a potentially different validated component for each forecast horizon step.",
        "assumption": "Enough historical origins exist to estimate horizon-specific error rankings.",
        "failure": "Sparse long-horizon validation creates a jagged and unstable sequence of selections.",
        "sources": ["source_000001", "source_000011"],
        "parents": ["method_seed_0004", "method_seed_0018", "method_tsfm_0002"],
        "hyperparameters": ["folds", "horizon_loss", "switch_penalty"],
    },
    {
        "name": "feature_gated_mixture_of_forecasters", "category": "selector",
        "description": "Use a learned gating model to assign target-dependent weights to statistical and foundation forecasters.",
        "assumption": "Observable history features identify which expert is reliable for each target.",
        "failure": "The gate overfits a small task collection or ignores a new failure mode shared by all experts.",
        "sources": ["source_000059", "source_000011"],
        "parents": ["method_seed_0018", "method_tsfm_0001", "method_tsfm_0002"],
        "hyperparameters": ["gate_model", "feature_set", "temperature"],
    },
    {
        "name": "arima_neural_residual_hybrid", "category": "residual_correction",
        "description": "Forecast a linear ARIMA component and add a neural forecast fitted to its residual structure.",
        "assumption": "The series contains separable linear and nonlinear components.",
        "failure": "Residual dependence changes after fitting or both components model the same signal twice.",
        "sources": ["source_000060"],
        "parents": ["method_seed_0018", "method_stat_0011"],
        "hyperparameters": ["arima_order", "neural_lags", "combination_scale"],
    },
    {
        "name": "timesfm_arima_residual_correction", "category": "residual_correction",
        "description": "Use TimesFM as the primary path and add an ARIMA forecast of historical primary-model residuals.",
        "assumption": "Systematic residual autocorrelation remains after the foundation forecast.",
        "failure": "Residuals estimated from too few pseudo-origins are noisy or change regime at deployment.",
        "sources": ["source_000060", "source_000030"],
        "parents": ["method_tsfm_0001", "method_seed_0018"],
        "hyperparameters": ["residual_origins", "residual_arima_order"],
    },
    {
        "name": "chronos_gradient_boost_residual_correction", "category": "residual_correction",
        "description": "Add a lag-feature gradient-boosting correction trained on pseudo-out-of-sample Chronos residuals.",
        "assumption": "Residual bias is predictable from historical, calendar, or known-future features.",
        "failure": "In-sample residual leakage or unseen covariate states causes an oversized correction.",
        "sources": ["source_000058", "source_000032"],
        "parents": ["method_tsfm_0002", "method_stat_0027"],
        "hyperparameters": ["residual_origins", "lag_features", "shrinkage"],
    },
    {
        "name": "ets_autoregressive_residual_correction", "category": "residual_correction",
        "description": "Forecast level, trend, and seasonality with ETS, then add an autoregressive residual forecast.",
        "assumption": "ETS captures structural components while remaining residual dependence is linear and stable.",
        "failure": "Residual correction double-counts seasonal dynamics or amplifies a structural break.",
        "sources": ["source_000001", "source_000060"],
        "parents": ["method_seed_0012", "method_seed_0015"],
        "hyperparameters": ["ets_form", "residual_ar_order"],
    },
    {
        "name": "validated_foundation_statistical_fallback", "category": "fallback",
        "description": "Use a foundation forecast only when it clears a margin over a statistical challenger in rolling validation.",
        "assumption": "History-only validation can detect harmful foundation-model mismatch before deployment.",
        "failure": "A future event changes the regime in a way no historical fold can reveal.",
        "sources": ["source_000001", "source_000011"],
        "parents": ["method_tsfm_0001", "method_seed_0012"],
        "hyperparameters": ["validation_margin", "folds", "default_parent"],
    },
    {
        "name": "short_history_foundation_fallback", "category": "fallback",
        "description": "Use a zero-shot foundation model below a data threshold and switch to a fitted statistical model once history is sufficient.",
        "assumption": "Pretraining is more reliable than local estimation only in the short-history regime.",
        "failure": "The threshold is poorly calibrated or the foundation model is mismatched to the target domain.",
        "sources": ["source_000011", "source_000046"],
        "parents": ["method_tsfm_0009", "method_seed_0012"],
        "hyperparameters": ["history_threshold", "statistical_parent"],
    },
    {
        "name": "outlier_guarded_model_fallback", "category": "fallback",
        "description": "Choose a robust or intervention-adjusted forecaster when anomaly diagnostics exceed a threshold, otherwise use the primary model.",
        "assumption": "The anomaly diagnostic separates contamination from genuine persistent changes.",
        "failure": "A true regime shift is labeled as an outlier and routed to an inappropriate robust fallback.",
        "sources": ["source_000021", "source_000011"],
        "parents": ["method_stat_0017", "method_tsfm_0002"],
        "hyperparameters": ["anomaly_threshold", "guard_window"],
    },
    {
        "name": "availability_aware_forecast_fallback", "category": "fallback",
        "description": "Execute an ordered list of forecasters and fall back when a model cannot satisfy context, covariate, device, or runtime constraints.",
        "assumption": "Fallback parents are valid substitutes and operational failures are detected before emitting a forecast.",
        "failure": "Silent model degradation passes the availability checks and prevents fallback activation.",
        "sources": ["source_000011"],
        "parents": ["method_tsfm_0014", "method_tsfm_0001", "method_seed_0004"],
        "hyperparameters": ["ordered_parents", "runtime_budget", "failure_checks"],
    },
)


def _jsonl(records: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def _legacy_card(method: Mapping[str, object], index: int) -> MethodCard:
    legacy_id = str(method["method_id"])
    source_ids = SPECIFIC_DEFINITION_SOURCES.get(legacy_id, ("source_000001",))
    return MethodCard.from_payload(
        {
            "method_uid": f"method_seed_{index:04d}",
            "definition_version": 1,
            "canonical_name": legacy_id,
            "aliases": [legacy_id],
            "family": "statistical",
            "category": CATEGORY_BY_LEGACY_ID[legacy_id],
            "description": method["description"],
            "assumptions": method["assumptions"],
            "failure_conditions": method["failure_conditions"],
            "applicability": {
                "minimum_history": 1,
                "frequencies": ["any"],
                "supports_univariate": True,
                "supports_covariates": False,
                "supports_probabilistic_output": (
                    CATEGORY_BY_LEGACY_ID[legacy_id] == "probabilistic"
                ),
            },
            "hyperparameters": [],
            "definition_source_ids": list(source_ids),
            "implementation_source_ids": [],
            "implementation_availability": "unknown",
            "verification_status": "verified",
            "lineage": {
                "operation": "verified_migrated_seed",
                "parent_method_uids": [],
                "legacy_dictionary_id": "statistical_base_methods_v000",
                "legacy_method_id": legacy_id,
            },
            "foundation_metadata": {},
        }
    )


def _extra_statistical_card(spec: Mapping[str, object], index: int) -> MethodCard:
    source_ids = tuple(str(item) for item in spec["sources"])
    implementation_sources = tuple(
        source_id for source_id in source_ids if source_id in {"source_000002", "source_000016"}
    )
    return MethodCard.from_payload(
        {
            "method_uid": f"method_stat_{index:04d}",
            "definition_version": 1,
            "canonical_name": spec["name"],
            "aliases": [],
            "family": "statistical",
            "category": spec["category"],
            "description": spec["description"],
            "assumptions": [spec["assumption"]],
            "failure_conditions": [spec["failure"]],
            "applicability": {
                "minimum_history": 10,
                "frequencies": ["any"],
                "supports_univariate": True,
                "supports_covariates": bool(spec.get("covariates", False)),
                "supports_probabilistic_output": bool(
                    spec.get("probabilistic", False)
                ),
            },
            "hyperparameters": list(spec.get("hyperparameters", [])),
            "definition_source_ids": list(source_ids),
            "implementation_source_ids": list(implementation_sources),
            "implementation_availability": (
                "available" if implementation_sources else "unknown"
            ),
            "verification_status": "verified",
            "lineage": {"operation": "collected", "parent_method_uids": []},
            "foundation_metadata": {},
        }
    )


def _foundation_card(spec: Mapping[str, object], index: int) -> MethodCard:
    metadata_values = tuple(spec["metadata"])
    metadata_keys = (
        "checkpoint_or_api",
        "release_version",
        "context_length",
        "prediction_length",
        "inference_mode",
        "probabilistic_output",
        "covariate_support",
        "device_requirements",
        "license",
        "weights_available",
        "code_available",
    )
    metadata = dict(zip(metadata_keys, metadata_values, strict=True))
    return MethodCard.from_payload(
        {
            "method_uid": f"method_tsfm_{index:04d}",
            "definition_version": 1,
            "canonical_name": spec["name"],
            "aliases": [],
            "family": "foundation",
            "category": spec["category"],
            "description": spec["description"],
            "assumptions": [spec["assumption"]],
            "failure_conditions": [spec["failure"]],
            "applicability": {
                "minimum_history": 1,
                "frequencies": ["cross_frequency"],
                "supports_univariate": True,
                "supports_covariates": bool(metadata["covariate_support"]),
                "supports_probabilistic_output": bool(
                    metadata["probabilistic_output"]
                ),
            },
            "hyperparameters": ["context_length", "prediction_length"],
            "definition_source_ids": list(spec["definition_sources"]),
            "implementation_source_ids": list(spec["implementation_sources"]),
            "implementation_availability": (
                "available" if metadata["code_available"] else "partial"
            ),
            "verification_status": "verified",
            "lineage": {"operation": "collected", "parent_method_uids": []},
            "foundation_metadata": metadata,
        }
    )


def _combined_card(spec: Mapping[str, object], index: int) -> MethodCard:
    return MethodCard.from_payload(
        {
            "method_uid": f"method_combined_{index:04d}",
            "definition_version": 1,
            "canonical_name": spec["name"],
            "aliases": [],
            "family": "combined",
            "category": spec["category"],
            "description": spec["description"],
            "assumptions": [spec["assumption"]],
            "failure_conditions": [spec["failure"]],
            "applicability": {
                "minimum_history": 1,
                "frequencies": ["any"],
                "supports_univariate": True,
                "supports_covariates": True,
                "supports_probabilistic_output": bool(
                    spec.get("probabilistic", False)
                ),
            },
            "hyperparameters": list(spec.get("hyperparameters", [])),
            "definition_source_ids": list(spec["sources"]),
            "implementation_source_ids": [],
            "implementation_availability": "unknown",
            "verification_status": "verified",
            "lineage": {
                "operation": "composed",
                "parent_method_uids": list(spec["parents"]),
            },
            "foundation_metadata": {},
        }
    )


def write_catalog_manifests(
    legacy_source: str | Path,
    source_destination: str | Path,
    method_destination: str | Path,
) -> tuple[tuple[SourceRecord, ...], tuple[MethodCard, ...]]:
    """Write the reviewed classical batch as deterministic JSONL manifests."""

    legacy_payload = json.loads(Path(legacy_source).read_text(encoding="utf-8"))
    legacy_methods = legacy_payload.get("methods")
    if not isinstance(legacy_methods, list):
        raise ValueError("legacy dictionary must contain a methods list")
    sources = tuple(
        SourceRecord.from_payload(payload)
        for payload in (
            SOURCE_PAYLOADS
            + ADDITIONAL_STATISTICAL_SOURCE_PAYLOADS
            + FOUNDATION_SOURCE_PAYLOADS
            + COMBINED_SOURCE_PAYLOADS
            + DEPTH_EXPANSION_SOURCE_PAYLOADS
        )
    )
    legacy_cards = tuple(
        _legacy_card(method, index)
        for index, method in enumerate(legacy_methods, start=1)
        if str(method["method_id"]) not in EXCLUDED_LEGACY_IDS
    )
    extra_cards = tuple(
        _extra_statistical_card(spec, index)
        for index, spec in enumerate(EXTRA_STATISTICAL_METHOD_SPECS, start=1)
    )
    expanded_statistical_cards = tuple(
        _extra_statistical_card(spec, index)
        for index, spec in enumerate(
            DEPTH_EXPANSION_STATISTICAL_METHOD_SPECS,
            start=len(EXTRA_STATISTICAL_METHOD_SPECS) + 1,
        )
    )
    foundation_cards = tuple(
        _foundation_card(spec, index)
        for index, spec in enumerate(FOUNDATION_METHOD_SPECS, start=1)
    )
    expanded_foundation_cards = tuple(
        _foundation_card(spec, index)
        for index, spec in enumerate(
            DEPTH_EXPANSION_FOUNDATION_METHOD_SPECS,
            start=len(FOUNDATION_METHOD_SPECS) + 1,
        )
    )
    combined_cards = tuple(
        _combined_card(spec, index)
        for index, spec in enumerate(COMBINED_METHOD_SPECS, start=1)
    )
    methods = (
        legacy_cards
        + extra_cards
        + expanded_statistical_cards
        + foundation_cards
        + expanded_foundation_cards
        + combined_cards
    )
    source_path = Path(source_destination)
    method_path = Path(method_destination)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    method_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        _jsonl([source.to_payload() for source in sources]), encoding="utf-8"
    )
    method_path.write_text(
        _jsonl([method.to_payload() for method in methods]), encoding="utf-8"
    )
    return sources, methods
