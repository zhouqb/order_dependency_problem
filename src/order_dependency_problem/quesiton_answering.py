"""Question answering module."""
import asyncio
from tqdm import tqdm
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from order_dependency_problem.dataset import MultipleChoiceQuestion


def create_chat_prompt(
        question: MultipleChoiceQuestion,
        label_removed: bool = False,
    ):
    """Create a chat prompt for the given question.
    
    Params:
        question: The question.
        label_removed: Whether the labels are removed.
        
    Returns:
        The chat prompt.
    """
    chat_prompt = ChatPromptTemplate([
        ("system", "The following is a multiple choice question. You should directly answer the question by choosing the correct option."),  # noqa
        ("user", """\
Question: {question}
OPtions:
{options}
Answer:""")])
    
    options = []
    for choice in question.choices:
        if label_removed:
            options.append(f"{choice.text}")
        else:
            options.append(f"{choice.label}. {choice.text}")
            
    return chat_prompt.invoke({
        "question": question.question,
        "options": "\n".join(options),
    })
    

async def answer_question(
    question: MultipleChoiceQuestion,
    model_name: str = "gpt-4o-mini",
    label_removed: bool = False,
):
    # TODO: Support models from other providers.
    model = ChatOpenAI(model_name=model_name, temperature=0.2)
    return await model.ainvoke(create_chat_prompt(question, label_removed))


async def answer_multiple_questions(
    questions: List[MultipleChoiceQuestion],
    model_name: str = "gpt-4o-mini",
    batch_size: int = 10,
    label_removed: bool = False,
    verbose: bool = False,
):
    """Answer multiple questions.
    
    Params:
        questions: List of questions.
        batch_size: Batch size.
        label_removed: Whether the labels are removed.
        verbose: Whether to show progress bar.
    
    Returns:
        List of answers.
    """
    answers = []
    
    num_samples = len(questions)
    
    if verbose:
        iterable = tqdm(range(0, num_samples, batch_size))
    else:
        iterable = range(0, num_samples, batch_size)
        
    for start_idx in iterable:
        batch = questions[start_idx:start_idx + batch_size]
        answers.extend(
            await asyncio.gather(
                *[answer_question(
                    question=question,
                    model_name=model_name,
                    label_removed=label_removed,
                  )
                  for question in batch]
            )
        )
    
    return answers