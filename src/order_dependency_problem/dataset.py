"""Dataset for order dependency problem."""
from abc import ABC, abstractmethod
import copy
import json
import random
from typing import List, Optional
import uuid

import pandas as pd
from pydantic import BaseModel, Field


class Choice(BaseModel):
    """Choice model."""
    
    text: str = Field(description="Choice text")
    label: Optional[str] = Field(default=None, description="Choice label")
    is_correct_answer: bool = Field(default=False, description="Is the correct answer")
    
    def __gt__(self, other):
        """Compare two choices based on their labels."""
        if isinstance(other, Choice):
            return self.label > other.label
        raise TypeError("Cannot compare Choice with non-Choice object.")


class MultipleChoiceQuestion(BaseModel):
    """Multiple choice question model."""
    
    id: str = Field(description="Question ID")
    question: str = Field(description="Question text")
    choices: List[Choice] = Field(description="Choices")
    

class BaseDataset(ABC):
    """Virtual class for dataset."""

    def __init__(self):
        self.questions: Optional[List[MultipleChoiceQuestion]] = None

    @classmethod
    @abstractmethod
    def load_from_file(
            cls,
            data_path: str,
            num_samples: Optional[int] = None,
            seed: Optional[int] = None,
        ):
        """Load the dataset from a file."""
        pass       
    
    def move_ground_truth_to_option(self, option: str):
        """Move the ground truth to the specified option.
        
        Params:
            option: The option label.
        Returns:
            A list of new questions with ground truths moved to the specified
            option.
        """
        result = []
        for question in self.questions:
            question = copy.deepcopy(question)
            ground_truth_idx = 0
            option_idx = 0
            for i, choice in enumerate(question.choices):
                if choice.is_correct_answer:
                    ground_truth_idx = i
                if choice.label == option:
                    option_idx = i

            if ground_truth_idx != option_idx:           
                (
                    question.choices[ground_truth_idx].text,
                    question.choices[option_idx].text,
                    question.choices[ground_truth_idx].is_correct_answer,
                    question.choices[option_idx].is_correct_answer,
                ) = (
                    question.choices[option_idx].text,
                    question.choices[ground_truth_idx].text,
                    question.choices[option_idx].is_correct_answer,
                    question.choices[ground_truth_idx].is_correct_answer,
                )
                        
            result.append(question)
        return result

    def generate_samples(
            self,
            shuffle_contents: bool = False,
            shuffle_labels: bool = False,
            seed: Optional[int] = None,
        ):
        """Generate samples from the dataset.
        
        Params:
            shuffle_contents: Shuffle the contents of the choices.
            shuffle_labels: Shuffle the labels of the choices.
            seed: Random seed.
        """
        if not (shuffle_contents or shuffle_labels):
            raise ValueError(
                "Either shuffle_contents or shuffle_labels must be True"
            )
            
        if seed is not None:
            random.seed(seed)
            
        result = []
        for question in self.questions:
            new_sample = copy.deepcopy(question)
            
            if shuffle_contents:
                random.shuffle(new_sample.choices)
                
            new_labels = [choice.label for choice in new_sample.choices]
            if shuffle_labels:
                random.shuffle(new_labels)
            else:
                new_labels.sort()

            for choice, new_label in zip(new_sample.choices, new_labels):
                choice.label = new_label
                
            result.append(new_sample)
                
        return result

class MmluDataset(BaseDataset):
    """MMLU dataset."""
    
    @classmethod
    def load_from_file(
            cls,
            data_path: str,
            num_samples: Optional[int] = None,
            seed: Optional[int] = None,
        ):
        """Load the dataset from a file.
        
        Params:
            data_path: Path to a csv file without header with the following
                columns:
                    - Question
                    - Option A
                    - Option B
                    - Option C
                    - Option D
                    - Ground truth answer
            num_samples: Number of samples to load. If None, load all samples.
            seed: Random seed.
        Returns:
            An ArcDataset object.
        """
        df = pd.read_csv(
            data_path,
            header=None,
            names=["Question", "A", "B", "C", "D", "Answer"],
        )
        
        if num_samples is not None:
            df = df.sample(
                n=num_samples,
                replace=False,
                random_state=seed,
            )
            
        result = cls()
        result.questions = []
        for _, row in df.iterrows():
            question = MultipleChoiceQuestion(
                id=str(uuid.uuid4()),
                question=row["Question"],
                choices=[
                    Choice(
                        text=row[label],
                        label=label,
                        is_correct_answer=label == row["Answer"],
                    )
                    for label in ["A", "B", "C", "D"]
                ]
            )
            result.questions.append(question)
        return result


class ArcDataset(BaseDataset):
    """ARC dataset."""
    
    @classmethod
    def load_from_file(
            cls,
            data_path: str,
            num_samples: Optional[int] = None,
            seed: Optional[int] = None,
        ):
        """Load the dataset from a file.
        
        Params:
            data_path: Path to a jsonl file. Each line is a json object with
                the following fields:
                    - id: str
                    - question: 
                        - stem: str (question text)
                        - choices: List[Dict[str, str]] (question choices)
                    - answerKey: str (correct answer label)
            num_samples: Number of samples to load. If None, load all samples.
            seed: Random seed.
        Returns:
            An ArcDataset object.
        """
        if seed is not None:
            random.seed(seed)
            
        result = cls()
        
        questions = []
        with open(data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                question = MultipleChoiceQuestion(
                    id=data["id"],
                    question=data["question"]["stem"],
                    choices=sorted([
                        Choice(
                            text=choice["text"],
                            label=choice["label"],
                            is_correct_answer=(
                                choice["label"] == data["answerKey"]
                            ),
                        )
                        for choice in data["question"]["choices"]
                    ])
                )
                questions.append(question)
        if num_samples is not None:
            questions = random.sample(questions, num_samples)
        result.questions = questions
        return result
