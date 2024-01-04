# AAAI ICWSM 2022 - The Web We Weave: Untangling the Social Graph of the IETF
## Instructions
Authors - Prashant Khare, Mladen Karan, Stephen McQuistin, Colin Perkins, Gareth Tyson, Matthew Purver, Patrick Healey, and Ignacio Castro

URL: https://ojs.aaai.org/index.php/ICWSM/article/view/19310

1.	Install [version 0.4 of the `ietfdata` library](https://github.com/glasgow-ipl/ietfdata/releases/tag/v0.4.0).
2.	Run `email-messages-year.py` to generate the data structures that are used in the subsequent analysis (for convenience, pre-generated versions are provided).
3.	Run `V2-generate_personID_personID_mappings.py` to generate interaction mappings of all the participants in the archive (for convenience, pre-generated versions are provided).
4.	Run the `notebook/ICWSM_analysis.ipynb` Jupyter notebook sequentially, to replicate the analysis and plot the figures. This will also generate `drafts_authors_SNA_features.csv` (for convenience, a pre-generated version is provided).
