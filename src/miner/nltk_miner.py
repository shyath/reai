
import random
import json
import re
from datetime import datetime, timedelta
from collections import Counter
from math import log
from typing import Any, Dict, List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datasets import load_dataset
from miner.base_miner import BaseMiner
from miner.resume_dataclasses import Resume, JobExperience, Education

class DataLoader:
    """
    A class for loading and managing various datasets related to schools, majors,
    skills, and job titles.

    This class provides methods to load data from JSON files and external datasets,
    and stores the loaded data in a dictionary for use in Resume().

    Attributes:
        data (dict): A dictionary containing the loaded datasets:
            - 'schools': List of school names
            - 'majors': List of major names
            - 'skills': List of skills
            - 'job_title_data': Dataset containing job title information

    Methods:
        load_school_names(file_path): Load school names from a JSON file
        load_majors(file_path): Load major names from a JSON file
        load_skills(): Load skills from an external dataset
        load_job_title_data(): Load job title data from an external dataset
    """

    def __init__(self):
        self.data = {
            'schools': self.load_school_names(),
            'majors': self.load_majors(),
            'skills': self.load_skills(),
            'job_title_data': self.load_job_title_data()
        }

    def load_school_names(self) -> List[str]:
        data = load_dataset("mw4/schools")["train"]
        return [school["name"] for school in data]

    def load_majors(self) -> List[str]:
        data = load_dataset("mw4/majors")["train"]
        return [major["name"] for major in data]

    def load_skills(self) -> List[str]:
        skills_data = load_dataset("DrDominikDellermann/SkillsDataset")["train"]["skills"]
        return [item['skill'] for sublist in skills_data for item in sublist]

    def load_job_title_data(self) -> List[str]:
        job_title_data = load_dataset("jacob-hugging-face/job-descriptions")
        return job_title_data


class RelevanceScorer:
    """
    A class to compute relevance scores for various text entries based on IDF 
    (Inverse Document Frequency).

    This class preprocesses text, calculates IDF values for words across a set 
    of documents, and uses these to determine relevance scores for jobs, skills, 
    and academic majors based on a given job description.

    Attributes:
        data (dict): A dictionary containing datasets used in relevance calculations. 
            Expected keys are 'job_title_data', 'skills', and 'majors'.
        stop_words (set): A set of stopwords for text preprocessing to ignore common 
            words that might skew relevance calculations.
        stemmer (PorterStemmer): An instance of PorterStemmer used to stem words 
            during text preprocessing.
        all_documents (list): A combined list of all documents from job titles, skills,
            and majors for IDF computation.
        idf (dict): A dictionary storing IDF scores for each word in `all_documents`.

    Methods:
        preprocess(text): Process the input text by lowering case, removing stopwords, 
            and stemming the remaining words.
        _calculate_idf(documents): Calculate the IDF for each unique word in the 
            provided documents list.
        calculate_relevance(text, documents): Calculate the relevance of each document 
            in `documents` with respect to the `text` based on the computed IDF scores.
        find_relevant_matches(job_description, num_jobs=3, num_skills=5, num_majors=1):
            Identify the top relevant job titles, skills, and majors for a given job 
            description. Returns a dictionary with keys 'job_titles', 'skills', and 
            'major'.
    """

    def __init__(self, data: Dict[str, Any]):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.data = data
        self.all_documents = (
            data['job_title_data']['train']['position_title'] +
            data['skills'] +
            data['majors']
        )
        self.idf = self._calculate_idf(self.all_documents)

    def preprocess(self, text: str) -> List[str]:
        words = re.findall(r'\w+', text.lower())
        return [self.stemmer.stem(word) for word in words if word not in self.stop_words]

    def _calculate_idf(self, documents: List[str]) -> Dict[str, float]:
        num_docs = len(documents)
        word_in_docs = Counter()
        for doc in documents:
            word_in_docs.update(set(self.preprocess(doc)))
        return {word: log(num_docs / (count + 1)) for word, count in word_in_docs.items()}

    def calculate_relevance(self, text: str, documents: List[str]) -> Dict[str, float]:
        text_words = self.preprocess(text)
        text_word_count = Counter(text_words)
        scores = {}
        for doc in documents:
            doc_words = self.preprocess(doc)
            score = sum(text_word_count[word] * self.idf.get(word, 0) for word in doc_words)
            scores[doc] = score
        return scores

    def find_relevant_matches(
            self,
            job_description: Dict[str, Any],
            num_jobs: int = 3,
            num_skills: int = 5,
            num_majors: int = 1
    ) -> Dict[str, List[Any]]:
        job_title_scores = self.calculate_relevance(
            job_description,
            self.data['job_title_data']['train']['position_title']
        )
        skill_scores = self.calculate_relevance(
            job_description,
            self.data['skills']
        )
        major_scores = self.calculate_relevance(
            job_description,
            self.data['majors']
        )

        def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
            max_score = max(scores.values()) if scores else 1
            if max_score == 0:
                return {k: 0 for k in scores}
            return {k: v / max_score for k, v in scores.items()}

        job_title_scores = normalize_scores(job_title_scores)
        skill_scores = normalize_scores(skill_scores)
        major_scores = normalize_scores(major_scores)

        top_job_titles = sorted(
            job_title_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_jobs]
        top_skills = sorted(
            skill_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_skills]
        top_major = sorted(major_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_majors]

        return {
            'job_titles': [(
                self.data['job_title_data']['train']['position_title'].index(title),
                title
            )
                for title, _ in top_job_titles
            ],
            'skills': [skill for skill, _ in top_skills],
            'major': [major for major, _ in top_major]
        }


class ResumeGenerator:
    """
    A class to generate resumes based on given data and job descriptions.

    This class uses a relevance scoring system to tailor resumes highlighting relevant 
    job titles, skills, and educational background tailored to specific job descriptions.

    Methods:
        __init__(self, data): Initializes the Resume class with provided data.
        get_scaled_periods(self, num_jobs, scale_factor): Calculate proportional work 
            periods for jobs.
        get_job_info(self, job_index, data): Fetch job information from the dataset.
        get_work_experience(self, relevant_job_titles, graduation_year): Compile work 
            experience section of the resume.
        get_education(self, major, graduation_year): Compile education section of the 
            resume.
        generate_resume(self, job_description): Generate a complete resume based on a job
            description.

    Attributes:
        scorer (RelevanceScorer): Utilizes RelevanceScorer to calculate relevance of data
            points.
        data (dict): A dictionary containing datasets used in relevance calculations. 
            Expected keys are 'job_title_data', 'skills', and 'majors'.
    """

    def __init__(self, data: Dict[str, Any]):
        self.scorer = RelevanceScorer(data)
        self.data = data

    def get_scaled_periods(self, num_jobs: int, scale_factor: float) -> List[float]:
        job_periods = [random.random() for _ in range(num_jobs)]
        sum_periods = sum(job_periods)
        return [x / sum_periods * scale_factor for x in job_periods]

    def get_job_info(self, job_index: int, data: Dict[str, Any]) -> Tuple[str, Any]:
        company_name = data['job_title_data']['train']['company_name'][job_index]
        model_response = json.loads(data['job_title_data']['train']['model_response'][job_index])
        core_responsibilities = model_response.get(
            "Core Responsibilities",
            "No core responsibilities found"
        )
        return company_name, core_responsibilities

    def get_work_experience(
            self, 
            relevant_job_titles: List[str], 
            graduation_year: int
        ) -> List[JobExperience]:
        
        work_experience = []
        total_days = 365 * random.randint(5, datetime.now().year - graduation_year + 1)

        normalized_periods = self.get_scaled_periods(
            len(relevant_job_titles),
            random.uniform(0.7, 1.0)
        )
        time_not_working = 1 - sum(normalized_periods)
        work_experience_coefficients = normalized_periods + [time_not_working]

        start_date = datetime.now() - timedelta(days=total_days)

        for index, (job_index, title) in enumerate(relevant_job_titles):
            company_name, description = self.get_job_info(job_index, self.data)
            job_duration_days = int(work_experience_coefficients[index] * total_days)
            end_date = start_date + timedelta(days=job_duration_days)

            job = JobExperience(
                title=title,
                company_name=company_name,
                description=description,
                start_date=start_date.strftime('%m-%Y'),
                end_date=end_date.strftime('%m-%Y')
            )

            work_experience.append(job)

            gap_days = random.randint(0, int(work_experience_coefficients[-1] * total_days / 3))
            start_date = end_date + timedelta(days=gap_days)

        return work_experience

    def get_education(self, major: str, graduation_year: int) -> List[Education]:
        degree_type = "Bachelor's"
        school_name = random.choice(self.data["schools"])
        graduation_month = random.choice(["05", "12"])
        degree = Education(
            school=school_name,
            major=major,
            degree=degree_type,
            start_date=f"0{random.randint(7, 9)}-{graduation_year - 4}",
            end_date=f"{graduation_month}-{graduation_year}"
        )
        return [degree]

    def generate_resume(self, job_description: Dict[str, Any]) -> Resume:
        results = self.scorer.find_relevant_matches(job_description)
        relevant_job_titles = results['job_titles']
        relevant_skills = results['skills']
        relevant_major = results['major'][0] if results['major'] else "No major found"
        graduation_year = random.randint(2008, 2018)

        resume = Resume(
            skills=relevant_skills,
            work_experience=self.get_work_experience(relevant_job_titles, graduation_year),
            education=self.get_education(relevant_major, graduation_year),
        )
        return resume


class NltkMiner(BaseMiner):
    """
    A specialized miner class that uses natural language processing (NLP) techniques to 
    generate resumes based on provided job descriptions. This class integrates NLTK-based
    processing to analyze and extract relevant data points for resume creation.

    Attributes:
        data_loader (DataLoader): An instance of DataLoader to fetch and prepare 
            necessary data.
        resume (Resume): An instance of Resume that uses loaded data to generate 
            tailored resumes.

    Methods:
        __init__(self): Initializes the NltkMiner with a DataLoader and a Resume 
            instance.
        generate_response(self, prompt: str): Generates a resume based on the provided 
            job description prompt. This method serves as the interface for inputting job 
            descriptions and receiving the generated resumes.
    """

    def __init__(self):
        super().__init__()
        self.data_loader = DataLoader()
        self.resume_generator = ResumeGenerator(self.data_loader.data)

    def generate_response(self, prompt: str) -> Resume:
        return self.resume_generator.generate_resume(prompt)
