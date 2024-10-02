"""Evaluation module"""
import collections
from difflib import SequenceMatcher
from typing import List

from order_dependency_problem.dataset import MultipleChoiceQuestion


def find_the_answer_option_idx(
        question: MultipleChoiceQuestion,
        answer: str,
        label_removed: str = False,
    ):
    """Find the option index in the choices list for a given answer.
    
    Params:
        question: The question.
        answer: The answer.
        label_removed: Whether the labels are removed.
        
    Returns:
        The option index in the choices list.
    """
    
    right_option = 0
    if label_removed:
        # Find the option with the highest similarity to the answer
        max_similarity = 0.0
        for i, choice in enumerate(question.choices):
            similarity = SequenceMatcher(None, choice.text, answer).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                right_option = i
                
    else:
        for i, choice in enumerate(question.choices):
            # LLM answer may contain extra spaces or characters, therefore
            # we need to clean it up before comparing.
            if answer.strip().lower().startswith(choice.label.lower()):
                right_option = i
                break
    return right_option


def calculate_answer_prevalence(
        questions: List[MultipleChoiceQuestion],
        answers: List[str],
        label_removed: bool = False,
    ):
    """Calculate the prevalence of each answer.
    
    This is a biased metric for ODP, since it is sensitive to ground truths
    answer prevalance.
    
    Params:
        questions: List of questions.
        answers: List of answers.
        label_removed: Whether the labels are removed.
        
    Returns:
        Return a dictionary with the prevalence of each answer. If
        label_removed is False, the keys are the labels of the choices;
        otherwise, the keys are the indices of the choices.
    """
    answer_counts = collections.defaultdict(lambda: 0)
    for question, answer in zip(questions, answers):
        right_option = find_the_answer_option_idx(
            question=question,
            answer=answer,
            label_removed=label_removed,
        )
        if label_removed:
            answer_counts[right_option] += 1
        else:
            answer_counts[question.choices[right_option].label] += 1
            
    return {key: val / len(questions) for key, val in answer_counts.items()}


def calculate_accuracy(
        questions: List[MultipleChoiceQuestion],
        answers: List[str],
        label_removed: bool = False,
    ):
    """Calculate the accuracy of the answers.
    
    Params:
        questions: List of questions.
        answers: List of answers.
        label_removed: Whether the labels are removed.
        
    Returns:
        The accuracy of the answers.
    """
    correct_count = 0
    for question, answer in zip(questions, answers):
        idx = find_the_answer_option_idx(
            question=question,
            answer=answer,
            label_removed=label_removed,
        )
        if question.choices[idx].is_correct_answer:
            correct_count += 1
    return correct_count / len(questions)


def calculate_answer_recall(
        questions: List[MultipleChoiceQuestion],
        answers: List[str],
        label_removed: bool = False,
    ):
    """Calculate the recall of the answers.
    
    Params:
        questions: List of questions.
        answers: List of answers.
        label_removed: Whether the labels are removed.
        
    Returns:
        The recall of the answers, grouped by ground truths. If
        label_removed is True, group by the indices of the choices;
        otherwise, group by the labels of the choices.
    """
    correct_counts = collections.defaultdict(lambda: 0)
    total_counts = collections.defaultdict(lambda: 0)
    for question, answer in zip(questions, answers):
        gs_label = None
        for i, choice in enumerate(question.choices):
            if choice.is_correct_answer:
                gs_label = i if label_removed else choice.label
                break
        total_counts[gs_label] += 1
        
        idx = find_the_answer_option_idx(
            question=question,
            answer=answer,
            label_removed=label_removed,
        )
        
        if question.choices[idx].is_correct_answer:
            correct_counts[gs_label] += 1
        
    return {key: val / total_counts[key] for key, val in correct_counts.items()}