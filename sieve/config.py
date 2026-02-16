# _______RAYYAN_________
RAYYAN_REVIEW_ID = 1723555
RAYYAN_LABELS = {
    "included": "Sieve Included",
    "excluded": "Sieve Excluded",
    "abstract_included": "Sieve Abstract Included",
    "abstract_excluded": "Sieve Abstract Excluded",
    "abstract_missing": "Missing Abstract",
    "batch_pending": "Sieve Batch Pending",
}
RAYYAN_EXCLUSION_LABELS = [
    "__EXR__not scholarly or not empirical",
    "__EXR__conference abstract",
    "__EXR__wrong population",
    "__EXR__not school-based placement context",
    "__EXR__motivation not placement-linked",
    "__EXR__not initial teacher education context",
]

# _______OPENAI_________
STUDY_OBJECTIVES = """
The objective of this review is to systematically identify, appraise, and synthesise global empirical research examining how school-based professional experience placements (e.g., practicum, internship, student teaching) influence pre-service teachers’ motivation within Initial Teacher Education programs, and to determine which placement features most strongly foster or hinder motivation.
"""  # noqa: E501

INCLUSION_HEADER = """
The following is an excerpt of 2 sets of criteria. A study is considered included if it meets all the inclusion criteria. If a study meets any of the exclusion criteria, it should be excluded. Here are the 2 sets of criteria:
"""  # noqa: E501

INCLUSION_CRITERIA = [
    "Population are pre-service teachers (teacher candidates/student teachers) enrolled in a university education degree (undergraduate or postgraduate).",  # noqa: E501
    "Must be situated in a school-based professional experience placement that forms part of Initial Teacher Education (e.g., practicum, teaching practice, internship, student teaching, field experience) with authentic school/classroom engagement under supervision. Early childhood, primary (elementary), and secondary (high school) settings are eligible where the placement is school based.",  # noqa: E501
    "Must examine pre-service teacher motivation in the context of professional experience placement (during placement, immediately before/after, or explicitly linked to placement experiences) and treat motivation as a central construct, evidenced by either (a) a stated motivation measure (e.g., self-efficacy, SDT/need satisfaction, expectancy-value, achievement goals, interest/intrinsic motivation, engagement/commitment, controlled/external regulation, intention to stay/retention, persistence/attrition) or (b) motivation as a primary qualitative/analytical focus."  # noqa: E501
    "Must be empirical research with original data collection and/or analysis, using qualitative, quantitative, or mixed-methods designs.",  # noqa: E501
]

EXCLUSION_CRITERIA = [
    "Studies that are non-empirical or not a research study, such as an editorial, opinion/commentary piece, conceptual/theoretical paper without original data, protocol, book review, or other item lacking original analysis.",  # noqa: E501
    "Exclude conference abstracts or conference proceedings.",  # noqa: E501
    "Participants are not pre-service teachers in Initial Teacher Education (e.g., in-service teachers only, teacher educators only without a pre-service focus, school students, or general university students not in teacher education).",  # noqa: E501
    "The experience is not a school-based professional placement (e.g., campus-based simulation, coursework-only experiences, micro-teaching, peer teaching, or activities without authentic school/classroom placement engagement).",  # noqa: E501
    "Studies that discuss motivation in a way that is not connected to professional experience placement (e.g., motivation for coursework, general academic motivation, or motivation unrelated to practicum/internship/student teaching).",  # noqa: E501
    "Studies that are not situated within Initial Teacher Education professional experience (e.g., induction/early career teacher programs, in-service professional learning placements, or other contexts not part of formal ITE practicum/internship requirements).",  # noqa: E501
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
