import json
from datetime import datetime
from collections import defaultdict
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from hugging_data import get_certifications_dataset, get_colleges
from normalize import DataNormalize
from typing import Dict, List, Any, Tuple

sample_resume_data = {
    "skills": [
        "Data Analysis",
        "Data Visualization",
        "Java"
    ],
    "certifications": [
        "Programming Fundamentals | Coursera",
        "Microsoft Certified: Azure Data Scientist Associate",
        "Certified ScrumMaster (CSM)"
    ],
    "work_experience": [
        {
            "job_title": "Theme park manager",
            "company": "Future Solutions",
            "start_date": "2021-02-28",
            "end_date": "2021-07-30",
            "roles": "Managed cloud infrastructure and optimized costs using AWS services."
        }
    ],
    "education": [
        {
            "school": "Harvard University",
            "major": "Computer Science",
            "degree": "Doctorate in Computer Science",
            "degree_type": "Doctorate",
            "start_date": "2010-11-30",
            "graduation_date": "2014-11-29"
        },
        {
            "school": "Nonexistent University",
            "major": "Physics",
            "degree": "Bachelor of Science",
            "degree_type": "Bachelor",
            "start_date": "2005-09-01",
            "graduation_date": "2009-06-30"
        }
    ],
    "projects": [
        {
            "name": "Face-to-face methodical archive",
            "roles": "Developed a machine learning model to predict customer churn, resulting in a 10% increase in retention.",
            "start_date": "2021-07-03",
            "end_date": "2023-11-13"
        },
        {
            "name": "Down-sized composite installation",
            "roles": "Created a real-time data visualization dashboard for tracking key business metrics.",
            "start_date": "2021-12-01",
            "end_date": "2023-02-05"
        }
    ]
}


class ResumeExtractor:
    def __init__(self, resume_data: Dict[str, Any] | None = None):
        if resume_data is None:
            self.resume_data = sample_resume_data
        else:
            self.resume_data = None
            self.add_resume_data(resume_data)
        self.vectorizer = TfidfVectorizer()
        self.dataset = get_certifications_dataset()
        self.certifications = [cert["Class"] for cert in self.dataset]
        self.universities = get_colleges()
        self.knn = self.load_knn_model()
        self.universal_skills = defaultdict(int)
        self.education_dict = defaultdict(lambda: {"exists": 0, "major": ""})
        self.work_experience_dict = defaultdict(float)
        self.workskills_dict = defaultdict(float)
        self.certification_dict = {}
        self.project_timelines = []
        self.job_titles = []
        self.data_normalizer = DataNormalize()

    def check_university_exists(self, university_name: str, universities: List[str]) -> 0 | 1:
        for uni in universities:
            if uni['name'].lower() == university_name.lower():
                return 1
        return 0

    def process_skills(self, skills: List[str], universal_skills: Dict[str, Any]) -> Dict[str, int]:
        for skill in skills:
            self.universal_skills[skill] += 1
            if skill in universal_skills:
                universal_skills[skill] += 1
            else:
                universal_skills[skill] = 1
        return universal_skills

    def find_nearest_certifications(
        self,
        query: str,
        knn_model: NearestNeighbors,
        vectorizer: TfidfVectorizer,
        certifications: List[str],
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        query_vec = vectorizer.transform([query])
        distances, indices = knn_model.kneighbors(query_vec, n_neighbors=len(certifications))
        nearest_certs = [(certifications[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        nearest_certs = [(cert, dist) for cert, dist in nearest_certs if dist < threshold]

        # If no nearest certs within threshold, try fuzzy matching
        if not nearest_certs:
            best_cert = max(certifications, key=lambda cert: fuzz.ratio(query, cert))
            return [(best_cert, 0.0)]  # Assuming perfect match with ratio 100

        return nearest_certs

    def calculate_years(self, start_date: str, end_date: str) -> float:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return (end - start).days / 365.25

    def skills_from_certs(self) -> Dict[str, Any]:
        certification_dict = self.certification_dict
        for cert in self.resume_data["certifications"]:
            nearest_certs = self.find_nearest_certifications(cert, self.knn, self.vectorizer, self.certifications, threshold=0.3)
            if nearest_certs:
                best_cert, _ = nearest_certs[0]
                matching_cert = self.dataset.filter(lambda x: x['Class'] == best_cert)
                if len(matching_cert) > 0:
                    cert_info = matching_cert[0]
                    skills_gained = cert_info['Skills Gained'].split(", ")
                    certification_dict[best_cert] = skills_gained
                    for skill in skills_gained:
                        if skill:
                            self.universal_skills[skill] += 1
                else:
                    certification_dict[best_cert] = []
                    for skill in best_cert.split():
                        self.universal_skills[skill] += 1
            else:
                certification_dict[cert] = []
                for skill in cert.split():
                    self.universal_skills[skill] += 1
        return certification_dict

    def verify_education(self) -> Dict[str, Any]:
        education_dict = self.education_dict
        for edu in self.resume_data["education"]:
            university_name = edu["school"]  # Fetch university name directly from the dict
            major = self.data_normalizer.get_normalized_degree_major(edu["major"])
            if self.check_university_exists(university_name, self.universities):
                education_dict[major]["exists"] = 1  # University exists
                education_dict[major]["major"] = major
            else:
                education_dict[major]["exists"] = 0  # University doesn't exist
                education_dict[major]["major"] = major
        return education_dict

    def process_work_experience(self):
        work_experience_dict = self.work_experience_dict
        workskills_dict = self.workskills_dict
        for work in self.resume_data["work_experience"]:
            job_title = work["job_title"]
            years_of_experience = self.calculate_years(work["start_date"], work["end_date"])
            work_experience_dict[job_title] += years_of_experience
            self.job_titles.append(job_title)
            # Extract skills from roles
            roles_skills = work["roles"].split()
            for skill in roles_skills:
                self.universal_skills[skill] += 1
                workskills_dict[skill] += years_of_experience

    def calculate_timelines(self):
        project_timelines = self.project_timelines
        for project in self.resume_data["projects"]:
            project_duration = self.calculate_years(project["start_date"], project["end_date"])
            project_timelines.append(project_duration)

    def get_segments(self) -> Dict[str, Any]:
        self.universal_skills = self.process_skills(self.resume_data["skills"])
        self.verify_education()
        self.process_work_experience()
        self.calculate_timelines()
        self.skills_from_certs()

        result = {
            "universal_skills": dict(self.universal_skills),
            "education_dict": dict(self.education_dict),
            "work_experience_dict": dict(self.work_experience_dict),
            "workskills_dict": dict(self.workskills_dict),
            "project_timelines": self.project_timelines,
            "certification_dictionary": self.certification_dict,
            "job_titles": self.data_normalizer.normalize_job_titles(self.job_titles)
        }
        return result

    def load_knn_model(self) -> NearestNeighbors:
        X = self.vectorizer.fit_transform(self.certifications)
        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(X)
        return knn

    def add_resume_data(self, resume_data: Dict[str, Any]):
        self.resume_data = [v for v in resume_data.values()][0]

    def reset(self):
        self.universal_skills = defaultdict(int)
        self.education_dict = defaultdict(lambda: {"exists": 0, "major": ""})
        self.work_experience_dict = defaultdict(float)
        self.workskills_dict = defaultdict(float)
        self.certification_dict = {}
        self.project_timelines = []
        self.job_titles = []


if __name__ == '__main__':
    resume_extractor = ResumeExtractor(resume_data=sample_resume_data)
    result = resume_extractor.get_segments()
    print("\nResult:")
    print(json.dumps(result, indent=4))
    resume_extractor.reset()
