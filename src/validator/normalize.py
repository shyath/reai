
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List
from hugging_data import get_degree_majors, get_degree_type_mappings, get_job_title_mappings


class DataNormalize:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def find_best_match(self, title: str, mappings: Dict[str, str]) -> str:
        titles = list([k for k in mappings.keys()])
        title_vec = self.vectorizer.fit_transform([title] + titles)
        cosine_similarities = cosine_similarity(title_vec[0:1], title_vec[1:]).flatten()
        best_match_index = np.argmax(cosine_similarities)
        best_match = titles[best_match_index]
        return best_match

    def find_closest_major(self, major: str, degree_majors: List[str]) -> str:
        title_vec = self.vectorizer.fit_transform([major] + degree_majors)
        cosine_similarities = cosine_similarity(title_vec[0:1], title_vec[1:]).flatten()
        best_match_index = np.argmax(cosine_similarities)
        closest_major = degree_majors[best_match_index]
        return closest_major

    def get_normalized_job_title(self, title: str) -> str:
        job_mappings = get_job_title_mappings()
        best_match = self.find_best_match(title, job_mappings)
        normalized_job_title = job_mappings[best_match]
        return normalized_job_title

    def get_normalized_degree_type(self, degree_type: str) -> str:
        degree_mappings = get_degree_type_mappings()
        best_match = self.find_best_match(degree_type, degree_mappings)
        normalized_degree_type = degree_mappings[best_match]
        return normalized_degree_type

    def get_normalized_degree_major(self, degree_major: str) -> str:
        degree_majors = get_degree_majors()
        normalized_degree_major = self.find_closest_major(degree_major, degree_majors)
        return normalized_degree_major

    def normalize_job_titles(self, job_titles: List[str]) -> List[str]:
        normalized_job_titles = []
        for job_title in job_titles:
            normalized_job_title = self.get_normalized_job_title(job_title)
            normalized_job_titles.append(normalized_job_title)
        return normalized_job_titles

    def normalize_degree_titles(self, degrees: List[Dict[str, str]]) -> List[Dict[str, str]]:
        normalized_degrees = []
        for degree in degrees:
            degree_type = degree['type']
            degree_major = degree['major']
            normalized_degree_type = self.get_normalized_degree_type(degree_type)
            normalized_degree_major = self.get_normalized_degree_major(degree_major)
            normalized_degree = {
                'type': normalized_degree_type,
                'major': normalized_degree_major
            }
            normalized_degrees.append(normalized_degree)
        return normalized_degrees


if __name__ == '__main__':
    example_job_titles = ['Junior Web Developer', 'Backend Developer II', 'Sr. Software Engineer']
    example_degrees = [
        {
            'type': 'MS',
            'major': 'Information Systems'
        },
        {
            'type': 'Bachelor of Science',
            'major': 'Computer Science'
        }
    ]
    data_normalize = DataNormalize()
    normal_jobs, normal_degrees = data_normalize.normalize_titles(example_job_titles, example_degrees)
    print(f"Normalized Job Titles: {normal_jobs}")
    for normal_degree in normal_degrees:
        print(f"Normalized Degrees: {normal_degree['type']}, {normal_degree['major']}")
