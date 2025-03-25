
import pandas as pd

from keywords import JDKeywordMatrix
from typing import Any, Dict, List, Tuple


class JDSkills:
    def __init__(self, skills_df: pd.DataFrame, job_description: str):
        self.skills_df = skills_df
        self.job_description = job_description
        self.kw_matrix = JDKeywordMatrix()

    def get_skills_maps(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        job_description = self.job_description
        skills_df = self.skills_df

        universal_skills = []
        preferred_skills = []

        for index, row in skills_df.iterrows():
            tf_idf_keywords = row['keywords']['tfidf_keywords']
            universal_skills += tf_idf_keywords
            ner_edu_keywords = row['keywords']['ner_keywords']['education']
            universal_skills += ner_edu_keywords
            ner_skills = row['keywords']['ner_keywords']['skills']
            universal_skills += ner_skills
            ner_exp_keywords = row['keywords']['ner_keywords']['experience']
            universal_skills += ner_exp_keywords
            preferred_skills_keywords = row['keywords']['ner_keywords']['preferred_skills']
            preferred_skills += preferred_skills_keywords

        universal_skills = list(set(universal_skills))
        preferred_skills = list(set(preferred_skills))

        universal_skills_map = {}
        preferred_skills_map = {}

        jd = job_description.lower()

        for universal_skill in universal_skills:
            u_skill = universal_skill.lower()
            u_skill_count = jd.count(u_skill)
            universal_skills_map[u_skill] = u_skill_count

        for preferred_skill in preferred_skills:
            p_skill = preferred_skill.lower()
            p_skill_count = jd.count(p_skill)
            preferred_skills_map[p_skill] = p_skill_count

        return universal_skills_map, preferred_skills_map

    def get_skills_weights(self) -> Tuple[Dict[str, float], Dict[str, float]]:

        universal_skills_map, preferred_skills_map = self.get_skills_maps()

        u_skills_list = universal_skills_map.keys()
        p_skills_list = preferred_skills_map.keys()

        u_keyword_scores = self.kw_matrix.get_keyword_scores_for_skills(u_skills_list)
        p_keyword_scores = self.kw_matrix.get_keyword_scores_for_skills(p_skills_list)

        universal_skills_weights = {}
        preferred_skills_weights = {}

        for u_keyword, u_score in u_keyword_scores.items():
            if u_keyword in u_skills_list:
                universal_skills_weights[u_keyword] = universal_skills_map[u_keyword] * u_score
            else:
                universal_skills_weights[u_keyword] = u_score

        for p_keyword, p_score in p_keyword_scores.items():
            if p_keyword in p_skills_list:
                preferred_skills_weights[p_keyword] = preferred_skills_map[p_keyword] * p_score
            else:
                preferred_skills_weights[p_keyword] = p_score

        return universal_skills_weights, preferred_skills_weights
