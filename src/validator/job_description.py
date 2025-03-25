import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.matcher import Matcher
from typing import Dict, Any, List
from hugging_data import get_degree_type_mappings
from normalize import DataNormalize


class JobDescriptionParser:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_trf')
        self.vectorizer = None
        self.load_vectorizer()
        self.matcher = Matcher(self.nlp.vocab)
        self.dt_mappings = get_degree_type_mappings()
        self._add_patterns()
        self.data_normalize = DataNormalize()

    def _add_patterns(self):
        degree_types = [{"LOWER": dt.lower()} for dt in self.dt_mappings.keys()]
        education_patterns = [
            {"LOWER": "bachelor's"}, {"LOWER": "bachelor"},
            {"LOWER": "master's"}, {"LOWER": "master"},
            {"LOWER": "phd"}, {"LOWER": "doctorate"},
            {"LOWER": "mba"}, {"LOWER": "degree"}
        ]
        education_patterns += degree_types
        self.matcher.add("EDUCATION", [education_patterns])

    def load_vectorizer(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    def extract_keywords_advanced(
        self,
        job_description: Dict[str, Any],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        # TF-IDF
        tfidf_matrix = self.vectorizer.transform([job_description])
        tfidf_scores = tfidf_matrix.toarray()[0]
        tfidf_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-10:][::-1]]

        # NER
        doc = self.nlp(job_description)
        print("Getting NER Keywords...")
        ner_keywords = {
            'skills': [ent.text for ent in doc.ents if ent.label_ in ['SKILL', 'LANGUAGE', 'ORG']],
            'experience': [ent.text for ent in doc.ents if ent.label_ in ['DATE', 'TIME']],
            'education': [self.data_normalize.get_normalized_degree_type(ent.text)
                          for ent in doc.ents if ent.label_ in ['WORK_OF_ART', 'TASK']],
            'preferred_skills': [ent.text for ent in doc.ents if ent.label_ in ['FAC']]
        }

        # matches = self.matcher(doc)
        # education_levels = []
        # for match_id, start, end in matches:
        #     span = doc[start:end]
        #     education_levels.append(span.text)
        #
        # ner_keywords['education'] = [self.data_normalize.get_normalized_degree_type(el) for el in education_levels]

        return {
            'tfidf_keywords': tfidf_keywords,
            'ner_keywords': ner_keywords
        }

    def get_skills_dataframe(self, job_description: Dict[str, Any]) -> pd.DataFrame:
        train_data = [{'description': job_description}]
        df = pd.DataFrame(train_data)

        self.vectorizer.fit(df['description'])

        feature_names = self.vectorizer.get_feature_names_out()

        df['keywords'] = df['description'].apply(
            lambda x: self.extract_keywords_advanced(x, feature_names))

        return df

    def get_formatted_jd(self, df: pd.DataFrame) -> Dict[str, Any]:
        skills_df = df
        formatted_jd = skills_df['keywords'].iloc[0]
        return formatted_jd

    def __str__(self) -> str:
        skills_dict = self.get_skills_dataframe().to_dict(orient='records')
        return str(skills_dict)


if __name__ == '__main__':
    jd = JobDescriptionParser()
    print(f"Skills: {jd.get_formatted_jd()}")
