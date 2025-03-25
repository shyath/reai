
import time
from abc import abstractmethod
from typing import Dict, Any
from loguru import logger
from communex.module import Module, endpoint
from miner.resume_dataclasses import Resume

class BaseMiner(Module):
    """
    A base class for implementing mining operations with the ability to process and 
    respond to prompts.
    
    This abstract class provides the structural foundation for generating responses 
    based on given prompts, specifically designed to be extended by classes that 
    implement specific types of data processing or response generation, such as resume 
    building from job descriptions.

    Attributes:
        Inherits attributes from the Module class.

    Methods:
        generate(self, prompt: str) -> Dict[str, Any]: A high-level method that handles
        the logging and timing of response generation. It calls the abstract method
        generate_response to perform the actual data processing.

        generate_response(self, prompt: str): Abstract method designed to be implemented
        by subclasses to generate a response based on the provided prompt. The 
        implementation should return a response relevant to the prompt. The implementation
        should return a Resume object.

    Note:
        As an abstract class, BaseMiner cannot be instantiated directly and must be 
        subclassed with an implementation of the generate_response method.
    """
    @endpoint
    def generate(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("Generating resume... ")
        logger.info(f"Job Description: {prompt}")
        resume = self.generate_response(prompt)

        resume_json = resume.to_json()

        logger.info(f"Generated Resume: {resume_json}")
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Responded in {execution_time} seconds")
        return { "answer": resume_json }

    @abstractmethod
    def generate_response(self, prompt: str) -> Resume:
        pass
