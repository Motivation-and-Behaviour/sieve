# _______RAYYAN_________
RAYYAN_REVIEW_ID = 1869703
RAYYAN_LABELS = {
    "included": "Sieve Included",
    "excluded": "Sieve Excluded",
    "abstract_included": "Sieve Abstract Included",
    "abstract_excluded": "Sieve Abstract Excluded",
    "abstract_missing": "Missing Abstract",
    "batch_pending": "Sieve Batch Pending",
}
RAYYAN_EXCLUSION_LABELS = [
    "__EXR__wrong age",
    "__EXR__wrong study design",
    "__EXR__wrong intervention",
    "__EXR__wrong outcome",
    "__EXR__animal or in vitro",
    "__EXR__foreign language",
    "__EXR__grey literature",
]

# _______OPENAI_________
STUDY_OBJECTIVES = """
The objective of this review is to synthesise the available evidence on the effects of creatine supplementation on cognitive function in healthy and clinical adult populations.
"""  # noqa: E501

INCLUSION_HEADER = """
The following is an excerpt of 2 sets of criteria. A study is considered included if it meets all the inclusion criteria. If a study meets any of the exclusion criteria, it should be excluded. Here are the 2 sets of criteria:
"""  # noqa: E501

INCLUSION_CRITERIA = [
    "Healthy or clinical adult populations aged ≥18 years. Include studies with a mean sample age of 18 years or older. If a mean age is not available, use the midpoint of the reported age range.",  # noqa: E501
    "Randomised controlled trials (RCTs) or quasi-experimental studies with a control or placebo group.",  # noqa: E501
    "Intervention involves oral creatine supplementation (e.g., creatine monohydrate) administered alone or as the primary active ingredient. Studies using creatine as part of a multi-ingredient supplement are included only if creatine is the sole difference between intervention and control conditions.",  # noqa: E501
    "Quantitative measurement of at least one cognitive outcome. Examples include: (i) Memory: short-term memory, working memory, long-term recall; (ii) Executive function: attention, inhibition, cognitive flexibility, task switching; (iii) Processing speed: reaction time, rapid information processing; (iv) Higher-order cognition: reasoning, problem-solving, fluid intelligence; (v) Mood and mental fatigue: subjective cognitive fatigue, mental effort, mood state assessed in conjunction with cognitive tasks.",  # noqa: E501
]

EXCLUSION_CRITERIA = [
    "Children and adolescents aged <18 years. Exclude studies with a mean sample age younger than 18 years. If the mean age is unclear, exclude if the midpoint of the age range is less than 18 years.",  # noqa: E501
    "Studies without an experimental or quasi-experimental design, including: narrative or systematic reviews, meta-analyses, editorials, commentaries, case reports or case series, observational studies (cross-sectional, cohort, or case-control) with no intervention, and qualitative-only designs.",  # noqa: E501
    "Intervention does not include creatine supplementation, or creatine is combined with other active compounds (e.g., caffeine, beta-alanine) without an appropriate control condition isolating the effect of creatine.",  # noqa: E501
    "Does not include at least one quantitatively measured cognitive outcome. Studies reporting only physical performance outcomes (e.g., strength, power, body composition) without any cognitive measure are excluded.",  # noqa: E501
    "Animal studies or in vitro research.",  # noqa: E501
    "Full-text not available in English.",  # noqa: E501
    "Conference abstracts, dissertations, or grey literature where insufficient methodological detail is available to assess eligibility and risk of bias.",  # noqa: E501
]

FULLTEXT_SCREENING_INSTRUCTIONS = """
# Instructions
We now assess whether the paper should be included from the systematic review by evaluating it against each and every predefined inclusion and exclusion criterion. First, we will reflect on how we will decide whether a paper should be included or excluded. Then, we will think step by step for each criterion, giving reasons for why they are met or not met.
Follow the schema exactly:
- Use 1-based indices for criteria lists.
- If none apply, return an empty list [] for any list field.
- 'triggered_exclusion' and 'exclusion_reasons' must align in order and length.
- 'rationale' should be a short paragraph (3-6 sentences) and must not exceed 1000 characters."""  # noqa: E501

ABSTRACT_SCREENING_INSTRUCTIONS = """
# Instructions
We now assess whether the paper should be included in the systematic review by evaluating it against each and every predefined inclusion and exclusion criterion. First, we will reflect on how we will decide whether a paper should be included or excluded. Then, we will think step by step for each criterion, giving reasons for why they are met or not met.  Studies that may not fully align with the primary focus of our inclusion criteria but provide data or insights potentially relevant to our review deserve thoughtful consideration. Given the nature of abstracts as concise summaries of comprehensive research, some degree of interpretation is necessary.
Our aim should be to inclusively screen abstracts, ensuring broad coverage of pertinent studies while filtering out those that are clearly irrelevant. Only vote "exclude" if the paper warrants exclusion, and vote "include" if inclusion is advised. If there is any uncertainty, vote "include" so that the uncertainty can be addressed at the fulltext stage.
"""  # noqa: E501
